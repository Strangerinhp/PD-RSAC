"""
GCN-based Actor per paper Section 5.1.

Implements:
- Two-layer GCN encoder over hex graph (paper Eq. 15)
- Context encoder for vehicle features
- Masked softmax with temperature annealing (paper Eq. 16)
- Continuous charging power head (paper Eq. 17-18)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Optional, Tuple, List, Dict

from .gcn import GCNEncoder


class GCNActor(nn.Module):
    """
    GCN-based Actor following paper Section 5.1.
    
    Architecture:
    1. Hex-Graph Encoder: 2-layer GCN over hex adjacency
    2. Vehicle Context Encoder: MLP processing vehicle features + local hex context
    3. Action Type Head: Masked softmax over discrete actions
    4. Reposition Head: Softmax over hex targets
    5. Charging Power Head: Truncated Gaussian for continuous power (optional)
    """
    
    def __init__(
        self,
        num_hexes: int,
        hex_feature_dim: int,
        vehicle_feature_dim: int,
        context_dim: int,
        gcn_hidden_dim: int = 128,
        gcn_output_dim: int = 64,
        actor_hidden_dim: int = 256,
        action_dim: int = 3,
        dropout: float = 0.1,
        use_continuous_power: bool = False,
        max_trips: int = 2000,  # Configurable (was hardcoded 500)
    ):
        super().__init__()
        self.num_hexes = num_hexes
        self.action_dim = action_dim
        self.use_continuous_power = use_continuous_power
        self.max_trips = max_trips  # Store before use
        
        # ===== GCN Encoder (paper Eq. 15) =====
        # Two-layer GCN with symmetric normalization
        self.gcn = GCNEncoder(
            input_dim=hex_feature_dim,
            hidden_dims=[gcn_hidden_dim],
            output_dim=gcn_output_dim,
            dropout=dropout,
            use_batch_norm=True,
            activation='silu'  # Paper uses SiLU
        )
        
        # ===== Vehicle Context Encoder =====
        # Combines: vehicle features + local hex context + global context
        context_input_dim = vehicle_feature_dim + gcn_output_dim + context_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, actor_hidden_dim),
            nn.LayerNorm(actor_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.LayerNorm(actor_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # ===== Action Type Head =====
        # Discrete: SERVE=0, CHARGE=1, REPOSITION=2  (IDLE removed)
        self.action_type_head = nn.Linear(actor_hidden_dim, action_dim)
        
        # ===== Trip Selection Head =====
        # For SERVE action: score available trips per vehicle
        # Input: encoded_context + global hex embeddings
        # Output: trip scores [num_vehicles, max_trips]
        # NOTE: max_trips now configurable via constructor (removed hardcode)
        self.trip_selection_head = nn.Sequential(
            nn.Linear(actor_hidden_dim + gcn_output_dim, actor_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim, self.max_trips)
        )
        
        # ===== Reposition Head =====
        # Outputs logits for each hex target
        self.reposition_head = nn.Sequential(
            nn.Linear(actor_hidden_dim + gcn_output_dim, actor_hidden_dim),
            nn.SiLU(),
            nn.Linear(actor_hidden_dim, num_hexes)
        )
        
        # ===== Continuous Charging Power Head (paper Eq. 17-18) =====
        if use_continuous_power:
            self.power_head = nn.Sequential(
                nn.Linear(actor_hidden_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 2)  # (mu, log_sigma)
            )
            self.log_std_min = -5.0
            self.log_std_max = 2.0
        
        # Action head initialized without explicit SERVE bias
        
        # Store adjacency for GCN forward
        self._adj_cache = None
    
    def set_adjacency(self, adj: torch.Tensor):
        """Set the hex graph adjacency matrix for GCN."""
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        if adj.dim() == 2:
            # Ensure adjacency is non-negative (clamp for robustness)
            adj_clean = torch.clamp(adj, min=0.0)
            
            # Add self-loops
            adj_with_self = adj_clean + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
            
            # Degree matrix
            deg = adj_with_self.sum(dim=1)
            
            # Add epsilon to avoid division by zero
            deg = torch.clamp(deg, min=1e-6)
            
            deg_inv_sqrt = torch.pow(deg, -0.5)
            
            # Normalized adjacency using broadcasting
            norm_adj = deg_inv_sqrt.unsqueeze(1) * adj_with_self * deg_inv_sqrt.unsqueeze(0)
            
            # Final NaN check and replacement
            norm_adj = torch.nan_to_num(norm_adj, nan=0.0, posinf=1.0, neginf=0.0)
            
            self._adj_cache = norm_adj
        else:
            self._adj_cache = adj
    
    def forward(
        self,
        hex_features: torch.Tensor,      # φ_h: [batch, num_hexes, hex_feature_dim] or [num_hexes, hex_feature_dim]
        vehicle_features: torch.Tensor,  # [batch, num_vehicles, vehicle_feature_dim] or [num_vehicles, vehicle_feature_dim]
        vehicle_hex_ids: torch.Tensor,   # [batch, num_vehicles] or [num_vehicles]
        context_features: torch.Tensor,  # [batch, context_dim] or [context_dim]
        adj: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        trip_mask: Optional[torch.Tensor] = None,  # [batch, max_trips] or [max_trips] - mask valid trips
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing action distributions.
        
        Returns dict with:
            - action_type: Selected action per vehicle
            - action_log_prob: Log probability of action
            - reposition_target: Target hex for reposition
            - reposition_log_prob: Log probability of reposition target
            - action_entropy: Entropy of action distribution
        """
        # Handle batch dimension
        if hex_features.dim() == 2:
            hex_features = hex_features.unsqueeze(0)
            vehicle_features = vehicle_features.unsqueeze(0)
            vehicle_hex_ids = vehicle_hex_ids.unsqueeze(0)
            context_features = context_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = hex_features.size(0)
        num_vehicles = vehicle_features.size(1)
        device = hex_features.device

        # Use cached or provided adjacency
        if adj is None:
            adj = self._adj_cache
        if adj is None:
            raise ValueError("Adjacency matrix not set. Call set_adjacency() first.")
        
        # Adjacency is [num_hexes, num_hexes] - shared across batch
        # GCN will broadcast it automatically via einsum
        if adj.dim() == 3:
            # If someone passed batched adjacency, take first one
            adj = adj[0]
        
        # ===== GCN Encoding =====
        # hex_embeddings: [batch, num_hexes, gcn_output_dim]
        hex_embeddings = self.gcn(hex_features, adj)
        
        # ===== Get local hex context for each vehicle =====
        # Gather hex embedding for each vehicle's current hex
        # vehicle_hex_ids: [batch, num_vehicles]
        batch_indices = torch.arange(batch_size, device=hex_features.device).unsqueeze(1).expand(-1, num_vehicles)
        local_hex_context = hex_embeddings[batch_indices, vehicle_hex_ids]  # [batch, num_vehicles, gcn_output_dim]
        
        # ===== Global hex context (pooled) =====
        global_hex_context = hex_embeddings.mean(dim=1)  # [batch, gcn_output_dim]
        
        # ===== Vehicle Context Encoding =====
        # Concatenate: vehicle features + local hex context + global context
        context_expanded = context_features.unsqueeze(1).expand(-1, num_vehicles, -1)
        vehicle_context = torch.cat([
            vehicle_features,
            local_hex_context,
            context_expanded
        ], dim=-1)  # [batch, num_vehicles, context_input_dim]
        
        # Encode context
        encoded_context = self.context_encoder(vehicle_context)  # [batch, num_vehicles, actor_hidden_dim]
        
        # ===== Action Type Head =====
        action_logits = self.action_type_head(encoded_context)  # [batch, num_vehicles, action_dim]
        
        # Sanitize logits
        action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        
        # Apply mask
        if action_mask is not None:
            if action_mask.dim() == 2:
                action_mask = action_mask.unsqueeze(0).expand(batch_size, -1, -1)
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))
        
        # Temperature-annealed softmax (paper Eq. 16)
        action_log_softmax = F.log_softmax(action_logits / temperature, dim=-1)
        action_probs = torch.exp(action_log_softmax)
        
        # Handle all-masked case: fallback to uniform over valid (non-masked) actions
        all_masked = action_probs.isnan().any(dim=-1) | (action_probs.sum(dim=-1) == 0)  # [batch, num_vehicles]
        if all_masked.any():
            if action_mask is not None:
                mask_exp = action_mask if action_mask.dim() == 3 else action_mask.unsqueeze(0).expand(batch_size, -1, -1)
                valid_count = mask_exp.sum(dim=-1, dtype=torch.float32).clamp(min=1.0)  # [batch, num_vehicles]
                default_probs = torch.where(mask_exp, (1.0 / valid_count).unsqueeze(-1), torch.zeros_like(action_probs))
                default_log_probs = torch.where(
                    mask_exp,
                    (-torch.log(valid_count + 1e-8)).unsqueeze(-1),
                    torch.full_like(action_log_softmax, float('-inf'))
                )
            else:
                k = action_probs.shape[-1]
                default_probs = torch.ones_like(action_probs) / k
                default_log_probs = torch.full_like(action_log_softmax, -torch.log(torch.tensor(k, dtype=torch.float32, device=action_probs.device)))
            action_probs = torch.where(all_masked.unsqueeze(-1), default_probs, action_probs)
            action_log_softmax = torch.where(all_masked.unsqueeze(-1), default_log_probs, action_log_softmax)
        
        # Sample or select action
        if deterministic:
            action_type = action_probs.argmax(dim=-1)
        else:
            # Memory-efficient Gumbel-Max trick avoids torch.multinomial OOM
            u = torch.rand_like(action_log_softmax)
            gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            action_type = (action_log_softmax + gumbel).argmax(dim=-1)
        
        # Memory-efficient: compute log_prob without full distribution
        action_log_prob = torch.gather(
            action_log_softmax,
            dim=-1,
            index=action_type.unsqueeze(-1)
        ).squeeze(-1)  # [batch, num_vehicles]
        
        # Clamp to prevent extreme values
        action_log_prob = torch.clamp(action_log_prob, min=-20.0, max=0.0)
        
        # Entropy: -sum(p * log(p))
        # Compute true entropy BEFORE nan_to_num (Bug 3 fix: clamping preserves magnitude)
        action_entropy = -(action_probs * torch.clamp(action_log_softmax, min=-20.0)).sum(dim=-1)  # [batch, num_vehicles]
        # safe_log_softmax only for gradient path (zero out -inf positions to avoid 0*-inf=NaN)
        safe_log_softmax = torch.nan_to_num(action_log_softmax, nan=0.0, neginf=0.0, posinf=0.0)
        
        # ===== Reposition Head (MEMORY-EFFICIENT) =====
        # Only compute reposition for vehicles that chose REPOSITION (action_type == 2)
        # Instead of [batch, ALL_vehicles, num_hexes], compute only for reposition vehicles
        reposition_target = torch.zeros(batch_size, num_vehicles, dtype=torch.long, device=device)
        reposition_log_prob = torch.zeros(batch_size, num_vehicles, device=device)

        reposition_mask_action = (action_type == 2)  # REPOSITION=2 in new scheme

        if reposition_mask_action.any():
            # Vectorized: flatten all repositioning vehicles across batch
            flat_indices = reposition_mask_action.nonzero(as_tuple=False)  # [total_repos, 2] (batch_idx, veh_idx)
            batch_idx = flat_indices[:, 0]  # [total_repos]
            veh_idx = flat_indices[:, 1]    # [total_repos]

            repos_context = encoded_context[batch_idx, veh_idx]  # [total_repos, hidden]
            repos_global = global_hex_context[batch_idx]  # [total_repos, gcn_out]
            repos_input = torch.cat([repos_context, repos_global], dim=-1)  # [total_repos, hidden + gcn_out]

            repos_logits = self.reposition_head(repos_input)  # [total_repos, num_hexes]
            repos_logits = torch.nan_to_num(repos_logits, nan=0.0, posinf=20.0, neginf=-20.0)

            if reposition_mask is not None:
                if reposition_mask.dim() == 2:
                    repos_mask_flat = reposition_mask[veh_idx]  # [total_repos, num_hexes]
                elif reposition_mask.dim() == 3:
                    repos_mask_flat = reposition_mask[batch_idx, veh_idx]  # [total_repos, num_hexes]
                else:
                    repos_mask_flat = None
                if repos_mask_flat is not None:
                    repos_logits = repos_logits.masked_fill(~repos_mask_flat, float('-inf'))

            repos_log_softmax = F.log_softmax(repos_logits / temperature, dim=-1)
            repos_probs = torch.exp(repos_log_softmax)
            
            # Handle NaNs from all-masked (fallback to uniform)
            all_masked_repos = repos_probs.isnan().any(dim=-1) | (repos_probs.sum(dim=-1) == 0)
            if all_masked_repos.any():
                repos_probs[all_masked_repos] = 1.0 / self.num_hexes
                repos_log_softmax[all_masked_repos] = -torch.log(torch.tensor(self.num_hexes, dtype=torch.float32, device=device))

            if deterministic:
                repos_target = repos_probs.argmax(dim=-1)
                
                # FIX: If deterministic, but completely masked (uniform fallback), argmax returns 0 for everyone.
                # This causes massive collisions in evaluation. We must randomly scatter these impossible choices.
                if all_masked_repos.any():
                    num_masked = all_masked_repos.sum().item()
                    random_targets = torch.randint(0, self.num_hexes, (num_masked,), device=device)
                    repos_target[all_masked_repos] = random_targets
            else:
                # Reposition target is execution detail — use deterministic argmax
                repos_target = repos_log_softmax.argmax(dim=-1)
                if all_masked_repos.any():
                    num_masked = all_masked_repos.sum().item()
                    random_targets = torch.randint(0, self.num_hexes, (num_masked,), device=device)
                    repos_target[all_masked_repos] = random_targets

            repos_lp = torch.gather(repos_log_softmax, dim=-1, index=repos_target.unsqueeze(-1)).squeeze(-1)
            repos_lp = torch.clamp(repos_lp, min=-20.0, max=0.0)

            reposition_target[batch_idx, veh_idx] = repos_target
            reposition_log_prob[batch_idx, veh_idx] = repos_lp

        # ===== Trip Selection Head (MEMORY-EFFICIENT) =====
        # Only compute for vehicles that chose SERVE (action_type == 0)
        selected_trip_init = torch.zeros(batch_size, num_vehicles, dtype=torch.long, device=device)
        trip_log_prob_init = torch.zeros(batch_size, num_vehicles, device=device)

        serve_mask_action = (action_type == 0)  # SERVE=0 in new scheme

        if serve_mask_action.any():
            # Vectorized: flatten all serving vehicles across batch
            flat_indices = serve_mask_action.nonzero(as_tuple=False)  # [total_serve, 2]
            batch_idx_s = flat_indices[:, 0]
            veh_idx_s = flat_indices[:, 1]

            serve_context = encoded_context[batch_idx_s, veh_idx_s]  # [total_serve, hidden]
            serve_global = global_hex_context[batch_idx_s]  # [total_serve, gcn_out]
            serve_input = torch.cat([serve_context, serve_global], dim=-1)

            serve_trip_logits = self.trip_selection_head(serve_input)  # [total_serve, max_trips]
            serve_trip_logits = torch.nan_to_num(serve_trip_logits, nan=0.0, posinf=20.0, neginf=-20.0)

            if trip_mask is not None:
                if trip_mask.dim() == 1:
                    # [max_trips] — shared mask for all vehicles
                    serve_trip_logits = serve_trip_logits.masked_fill(~trip_mask.unsqueeze(0).expand(serve_trip_logits.shape[0], -1), float('-inf'))
                elif trip_mask.dim() == 2:
                    # [num_vehicles, max_trips] — per-vehicle reachability mask
                    serve_trip_logits = serve_trip_logits.masked_fill(~trip_mask[veh_idx_s], float('-inf'))

            serve_trip_log_softmax = F.log_softmax(serve_trip_logits / temperature, dim=-1)
            serve_trip_probs = torch.exp(serve_trip_log_softmax)

            # Handle NaNs from all-masked (fallback to uniform)
            all_masked_serve = serve_trip_probs.isnan().any(dim=-1) | (serve_trip_probs.sum(dim=-1) == 0)
            if all_masked_serve.any():
                serve_trip_probs[all_masked_serve] = 1.0 / self.max_trips
                serve_trip_log_softmax[all_masked_serve] = -torch.log(torch.tensor(self.max_trips, dtype=torch.float32, device=device))

            if deterministic:
                serve_trip_target = serve_trip_probs.argmax(dim=-1)

                # FIX: If deterministic, but completely masked (uniform fallback), argmax returns 0 for everyone.
                # This causes massive collisions in evaluation. We must randomly scatter these impossible choices.
                if all_masked_serve.any():
                    num_masked = all_masked_serve.sum().item()
                    random_targets = torch.randint(0, self.max_trips, (num_masked,), device=device)
                    serve_trip_target[all_masked_serve] = random_targets
            else:
                # Serve trip selection is execution detail — use deterministic argmax
                serve_trip_target = serve_trip_log_softmax.argmax(dim=-1)

            serve_trip_lp = torch.gather(serve_trip_log_softmax, dim=-1, index=serve_trip_target.unsqueeze(-1)).squeeze(-1)
            serve_trip_lp = torch.clamp(serve_trip_lp, min=-20.0, max=0.0)

            selected_trip_init[batch_idx_s, veh_idx_s] = serve_trip_target
            trip_log_prob_init[batch_idx_s, veh_idx_s] = serve_trip_lp

        selected_trip = selected_trip_init
        trip_log_prob = trip_log_prob_init

        result = {
            'action_type': action_type,
            'action_log_prob': action_log_prob,
            'action_probs': action_probs,                  # [batch, V, 3] full distribution (differentiable)
            'action_log_probs_all': safe_log_softmax,      # [batch, V, 3] log probs all actions (no -inf)
            'reposition_target': reposition_target,
            'reposition_log_prob': reposition_log_prob,
            'selected_trip': selected_trip,
            'trip_log_prob': trip_log_prob,
            'action_entropy': action_entropy,
            'hex_embeddings': hex_embeddings,
            '_encoded_context': encoded_context,            # [batch, V, hidden] for aux loss
            '_global_hex_context': global_hex_context,      # [batch, gcn_out] for aux loss
        }
        
        # ===== Continuous Charging Power (optional) =====
        if self.use_continuous_power:
            power_params = self.power_head(encoded_context)  # [batch, num_vehicles, 2]
            mu = power_params[..., 0]
            log_std = torch.clamp(power_params[..., 1], self.log_std_min, self.log_std_max)
            std = log_std.exp()
            
            if deterministic:
                charging_power = torch.sigmoid(mu)  # Squash to [0, 1], scale to P_max externally
            else:
                power_dist = Normal(mu, std)
                u = power_dist.rsample()
                charging_power = torch.sigmoid(u)
                
                # Log prob with Jacobian correction (paper Eq. 18)
                power_log_prob = power_dist.log_prob(u) - torch.log(charging_power * (1 - charging_power) + 1e-6)
                result['charging_power'] = charging_power
                result['power_log_prob'] = power_log_prob
        
        if squeeze_output:
            for key in result:
                if isinstance(result[key], torch.Tensor) and result[key].dim() > 0:
                    result[key] = result[key].squeeze(0)
        
        return result
    
    def get_backbone_output(self, state: torch.Tensor) -> torch.Tensor:
        """Compatibility method for episode_collector that accesses backbone directly."""
        # This is a simplified version that works with flat states
        # In practice, we should parse the flat state back into components
        return self.context_encoder[0:4](state)  # Return after first two layers


class GCNActorWrapper(nn.Module):
    """
    Wrapper that provides backward compatibility with the original Actor interface.
    
    Parses flat state tensor into hex/vehicle/context components for GCNActor.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_hexes: int,
        num_vehicles: int,
        vehicle_feature_dim: int = 16,
        hex_feature_dim: int = 8,
        context_dim: int = 32,
        hidden_dims: List[int] = [256, 256],
        dropout: float = 0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_hexes = num_hexes
        self.num_vehicles = num_vehicles
        self.vehicle_feature_dim = vehicle_feature_dim
        self.hex_feature_dim = hex_feature_dim
        self.context_dim = context_dim
        
        # Initialize GCN Actor
        self.gcn_actor = GCNActor(
            num_hexes=num_hexes,
            hex_feature_dim=hex_feature_dim,
            vehicle_feature_dim=vehicle_feature_dim,
            context_dim=context_dim,
            gcn_hidden_dim=hidden_dims[0] // 2,
            gcn_output_dim=64,
            actor_hidden_dim=hidden_dims[-1],
            action_dim=action_dim,
            dropout=dropout
        )
        
        # Fallback MLP backbone for compatibility
        dims = [state_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        
        self.action_type_head = nn.Linear(hidden_dims[-1], action_dim)
        self.reposition_head = nn.Linear(hidden_dims[-1], num_hexes)
        
        # Emtpy action bias
    
    def _parse_flat_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse flat state back into components."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.size(0)
        vehicle_size = self.num_vehicles * self.vehicle_feature_dim
        hex_size = self.num_hexes * self.hex_feature_dim
        
        vehicle_features = state[:, :vehicle_size].view(batch_size, self.num_vehicles, self.vehicle_feature_dim)
        hex_features = state[:, vehicle_size:vehicle_size + hex_size].view(batch_size, self.num_hexes, self.hex_feature_dim)
        context_features = state[:, vehicle_size + hex_size:]
        
        # Extract vehicle hex IDs from vehicle features (assume first feature is hex_id or position)
        # This is a simplification - in practice we'd have this info from environment
        vehicle_hex_ids = torch.zeros(batch_size, self.num_vehicles, dtype=torch.long, device=state.device)
        
        return hex_features, vehicle_features, vehicle_hex_ids, context_features
    
    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass using MLP backbone (for compatibility)."""
        features = self.backbone(state)
        
        action_logits = self.action_type_head(features)
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))
        
        action_log_softmax = F.log_softmax(action_logits, dim=-1)
        action_probs = torch.exp(action_log_softmax)
        action_dist = Categorical(probs=action_probs)
        
        if deterministic:
            action_type = action_probs.argmax(dim=-1)
        else:
            u = torch.rand_like(action_log_softmax)
            gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            action_type = (action_log_softmax + gumbel).argmax(dim=-1)
        
        action_log_prob = torch.gather(action_log_softmax, dim=-1, index=action_type.unsqueeze(-1)).squeeze(-1)
        
        reposition_logits = self.reposition_head(features)
        if reposition_mask is not None:
            reposition_logits = reposition_logits.masked_fill(~reposition_mask, float('-inf'))
        
        reposition_log_softmax = F.log_softmax(reposition_logits, dim=-1)
        reposition_probs = torch.exp(reposition_log_softmax)
        
        if deterministic:
            reposition_target = reposition_probs.argmax(dim=-1)
        else:
            u = torch.rand_like(reposition_log_softmax)
            gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            reposition_target = (reposition_log_softmax + gumbel).argmax(dim=-1)
        
        reposition_log_prob = torch.gather(reposition_log_softmax, dim=-1, index=reposition_target.unsqueeze(-1)).squeeze(-1)
        
        return action_type, action_log_prob, reposition_target, reposition_log_prob
