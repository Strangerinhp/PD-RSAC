import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import copy

from .gcn_actor import GCNActor
from .critic import TwinCritic  # Legacy flat critic
from .gcn_critic import GCNTwinCritic  # NEW: GCN-based critic matching Actor


@dataclass
class SACOutput:
    action_type: torch.Tensor
    reposition_target: torch.Tensor
    action_log_prob: torch.Tensor
    reposition_log_prob: torch.Tensor
    selected_trip: Optional[torch.Tensor] = None  # NEW: for GCN trip selection
    trip_log_prob: Optional[torch.Tensor] = None  # NEW: log prob of trip selection
    serve_scores: Optional[torch.Tensor] = None  # For assignment
    charge_scores: Optional[torch.Tensor] = None  # For assignment
    q1: Optional[torch.Tensor] = None
    q2: Optional[torch.Tensor] = None
    action_entropy: Optional[torch.Tensor] = None  # For GCN actor
    hex_embeddings: Optional[torch.Tensor] = None  # For GCN actor
    action_probs: Optional[torch.Tensor] = None         # [batch, V, 3] full distribution
    action_log_probs_all: Optional[torch.Tensor] = None  # [batch, V, 3] log probs all actions
    _encoded_context: Optional[torch.Tensor] = None      # [batch, V, hidden] for aux loss
    _global_hex_context: Optional[torch.Tensor] = None   # [batch, gcn_out] for aux loss


class SACAgent(nn.Module):
    """
    Soft Actor-Critic Agent per paper.
    
    Features:
    - Twin critics with min(Q1, Q2)
    - Automatic entropy tuning
    - Semi-MDP duration discounting (γ^Δ) - ENABLED BY DEFAULT
    - Reward normalization
    - GCNActor for per-vehicle spatial reasoning (paper Section 5.1.2)
    - Action space: SERVE=0, CHARGE=1, REPOSITION=2 (IDLE removed)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_hexes: int,
        actor_hidden_dims: List[int] = [256, 256],
        critic_hidden_dims: List[int] = [256, 256],
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        dropout: float = 0.1,
        device: str = 'cuda',
        num_vehicles: int = 1000,
        use_semi_mdp: bool = True,  # Semi-MDP enabled by default per paper
        use_gcn_critic: bool = True,  # NEW: Use GCN critic matching Actor architecture
        # GCN options
        vehicle_feature_dim: int = 16,  # FIXED: Matches feature builder output
        hex_feature_dim: int = 5,
        context_dim: int = 9,
        # Configurable limits (removed hardcoding)
        max_trips: int = 2000,
        num_stations: int = 150,
        max_serve: int = 500,
        max_charge: int = 200,
        min_alpha: float = 0.05,
        max_alpha: float = 1.0,
        repos_aux_weight: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_hexes = num_hexes
        self.num_vehicles = num_vehicles
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.device = device
        self.use_semi_mdp = use_semi_mdp
        self.use_gcn_critic = use_gcn_critic
        self.repos_aux_weight = repos_aux_weight
        
        # Store feature dimensions for state parsing
        self.vehicle_feature_dim = vehicle_feature_dim
        self.hex_feature_dim = hex_feature_dim
        self.context_dim = context_dim
        
        # Store configurable limits BEFORE use in Actor/Critic construction
        self._max_trips = max_trips
        self._num_stations = num_stations
        self._max_serve = max_serve
        self._max_charge = max_charge
        
        # GCNActor for per-vehicle spatial reasoning (paper Section 5.1.2)
        # Paper spec: GCN hidden_dims=(128, 64), actor_hidden=128
        # actor_hidden_dims[0] = GCN first layer output
        # actor_hidden_dims[1] = actor context encoder hidden
        self.actor = GCNActor(
            num_hexes=num_hexes,
            hex_feature_dim=hex_feature_dim,
            vehicle_feature_dim=vehicle_feature_dim,
            context_dim=context_dim,
            gcn_hidden_dim=actor_hidden_dims[0],  # Paper: 128
            gcn_output_dim=64,  # Paper: d_h=64
            actor_hidden_dim=actor_hidden_dims[1] if len(actor_hidden_dims) > 1 else actor_hidden_dims[0],
            action_dim=action_dim,
            dropout=dropout,
            use_continuous_power=False,
            max_trips=self._max_trips,  # Use stored value
        ).to(device)
        # Adjacency matrix cache (set by environment)
        self._adjacency_matrix: Optional[torch.Tensor] = None

        # Create Critic (GCN-based or legacy flat)
        if use_gcn_critic:
            print("[SAC] Using GCN-based Critic (matching Actor architecture)")
            self.critic = GCNTwinCritic(
                num_hexes=num_hexes,
                hex_feature_dim=hex_feature_dim,
                vehicle_feature_dim=vehicle_feature_dim,
                context_dim=context_dim,
                action_dim=action_dim,
                gcn_hidden_dim=actor_hidden_dims[0],  # Match Actor GCN dim
                gcn_output_dim=64,  # Match Actor
                critic_hidden_dim=critic_hidden_dims[0] if critic_hidden_dims else 256,
                dropout=dropout,
                aggregation='mean',  # Fleet Q = mean of per-vehicle Q
            ).to(device)
        else:
            print("[SAC] Using legacy flat Critic (no spatial reasoning)")
            self.critic = TwinCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=critic_hidden_dims,
                dropout=dropout,
                num_vehicles=num_vehicles,
                max_trips=self._max_trips,
                num_stations=self._num_stations,
                max_serve_per_step=self._max_serve,
                max_charge_per_step=self._max_charge,
                use_assignment_encoding=True,
            ).to(device)

        self.critic_target = copy.deepcopy(self.critic)
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        if auto_alpha:
            if target_entropy is None:
                # For discrete actions: target entropy = -0.98 * log(|A|)
                # Standard SAC-Discrete formulation from paper
                # For 3 actions (SERVE/CHARGE/REPOSITION): -0.98 * log(3) = -1.076
                # This allows policy to become nearly deterministic when optimal
                target_entropy = -0.98 * torch.log(torch.tensor(float(action_dim))).item()
            self.target_entropy = target_entropy
            # Initialize log_alpha from provided alpha value (respects config)
            self.log_alpha = nn.Parameter(torch.tensor(alpha).log())
            # Minimum and maximum alpha bounds - honored from config
            self.min_alpha = min_alpha
            self.max_alpha = max_alpha
        else:
            self.register_buffer('log_alpha', torch.tensor(alpha).log())
            self.target_entropy = None
            self.min_alpha = 0.0
            self.max_alpha = float('inf')
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        if auto_alpha:
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        
        # Reward normalization - running statistics
        # Initialize with reasonable estimates based on NYC taxi data analysis:
        # - Per-step reward range: [-994, +86] → mean ≈ -230, std ≈ 150
        # - This prevents clipping in early batches and stabilizes training
        self.reward_mean = -200.0  # Conservative estimate of typical mean
        self.reward_std = 150.0    # Based on actual std ≈ 155
        self.reward_count = 1000   # Pretend we've seen some data already
        self._reward_scale = 1.0   # Don't scale rewards, just normalize
    
    def call_critic(
        self,
        critic,
        states_flat: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Call critic (GCN or flat) with correct interface.

        Use this instead of calling critic directly to handle both GCN and flat critics.
        """
        if self.use_gcn_critic:
            hex_features, vehicle_features, vehicle_hex_ids, context_features = \
                self._parse_flat_state(states_flat)
            return critic(
                hex_features=hex_features,
                vehicle_features=vehicle_features,
                vehicle_hex_ids=vehicle_hex_ids,
                context_features=context_features,
                adjacency=self._adjacency_matrix,
                actions=actions,
            )
        else:
            # Legacy: aggregate actions for flat critic
            if actions.dim() == 2 and actions.size(1) > 1:
                action_agg = torch.mode(actions, dim=1).values
            else:
                action_agg = actions.squeeze(-1) if actions.dim() > 1 else actions
            return critic(states_flat, action_agg)

    def set_adjacency(self, adj: torch.Tensor):
        """Set the hex graph adjacency matrix for GCN actor.
        
        Args:
            adj: [num_hexes, num_hexes] adjacency matrix (can be sparse or dense)
        """
        self._adjacency_matrix = adj
        
        # Handle DDP/DataParallel wrapping
        actor_module = self.actor.module if hasattr(self.actor, 'module') else self.actor
        actor_module.set_adjacency(adj)
    
    def _parse_flat_state(self, state: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse flat state tensor back into components for GCN actor.
        
        Returns:
            hex_features: [batch, num_hexes, hex_feature_dim]
            vehicle_features: [batch, num_vehicles, vehicle_feature_dim]
            vehicle_hex_ids: [batch, num_vehicles] - hex index for each vehicle
            context_features: [batch, context_dim]
        """
        if isinstance(state, dict):
            # Already parsed structure from replay buffer
            vehicle_features = state['vehicle']
            hex_features = state['hex']
            context_features = state['context']
            
            # Add batch dimension if missing
            if vehicle_features.dim() == 2:
                vehicle_features = vehicle_features.unsqueeze(0)
            if hex_features.dim() == 2:
                hex_features = hex_features.unsqueeze(0)
            if context_features.dim() == 1:
                context_features = context_features.unsqueeze(0)
        else:
            # Parse from flat tensor
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            batch_size = state.size(0)
            vehicle_size = self.num_vehicles * self.vehicle_feature_dim
            hex_size = self.num_hexes * self.hex_feature_dim
            
            # Extract components
            vehicle_features = state[:, :vehicle_size].view(batch_size, self.num_vehicles, self.vehicle_feature_dim)
            hex_features = state[:, vehicle_size:vehicle_size + hex_size].view(batch_size, self.num_hexes, self.hex_feature_dim)
            context_features = state[:, vehicle_size + hex_size:]
        
        # Extract vehicle positions (hex IDs) from vehicle features
        # The 1st feature (index 0) is normalized position, denormalize to get hex_id
        # Based on feature builder: vehicle_features[:, 0] = position / num_hexes
        vehicle_positions_normalized = vehicle_features[:, :, 0]  # [batch, num_vehicles]
        vehicle_hex_ids = (vehicle_positions_normalized * self.num_hexes).long().clamp(0, self.num_hexes - 1)
        
        return hex_features, vehicle_features, vehicle_hex_ids, context_features
    
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running statistics for stable training."""
        # Update running statistics
        batch_mean = rewards.mean().item()
        batch_std = rewards.std().item() + 1e-8
        batch_size = rewards.numel()
        
        # Welford's online algorithm for running mean/std
        new_count = self.reward_count + batch_size
        delta = batch_mean - self.reward_mean
        self.reward_mean += delta * batch_size / new_count
        self.reward_std = 0.9 * self.reward_std + 0.1 * batch_std  # EMA for std
        self.reward_count = new_count
        
        # Normalize to zero mean, unit variance
        normalized = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        # Clip to prevent extreme values but allow reasonable range
        normalized = torch.clamp(normalized, -5.0, 5.0)
        
        return normalized
    
    @property
    def alpha(self) -> torch.Tensor:
        # Ensure alpha stays within [min_alpha, max_alpha] bounds
        raw_alpha = self.log_alpha.exp()
        min_a = self.min_alpha if hasattr(self, 'min_alpha') else 0.0
        max_a = self.max_alpha if hasattr(self, 'max_alpha') else float('inf')
        return torch.clamp(raw_alpha, min=min_a, max=max_a)
    
    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        trip_features: Optional[torch.Tensor] = None,
        trip_mask: Optional[torch.Tensor] = None,
        station_features: Optional[torch.Tensor] = None,
        station_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
        vehicle_hex_ids: Optional[torch.Tensor] = None,
    ) -> SACOutput:
        """Forward pass through GCNActor.
        
        Args:
            state: Flat state tensor [batch, state_dim] or [state_dim]
            action_mask: [num_vehicles, action_dim] or [batch, num_vehicles, action_dim]
            reposition_mask: [num_vehicles, num_hexes] or [batch, num_vehicles, num_hexes]
            trip_mask: Optional trip mask for GCN trip selection
            deterministic: Whether to use deterministic action selection
            temperature: Temperature for softmax (GCN actor)
            vehicle_hex_ids: [num_vehicles] or [batch, num_vehicles] current hex for each vehicle
        """
        # Parse flat state into components for GCN
        hex_features, vehicle_features, parsed_hex_ids, context_features = self._parse_flat_state(state)
        
        # Use provided vehicle_hex_ids if available, else use parsed ones
        if vehicle_hex_ids is None:
            vehicle_hex_ids = parsed_hex_ids
        
        # GCNActor forward
        output = self.actor(
            hex_features=hex_features,
            vehicle_features=vehicle_features,
            vehicle_hex_ids=vehicle_hex_ids,
            context_features=context_features,
            adj=self._adjacency_matrix,
            action_mask=action_mask,
            reposition_mask=reposition_mask,
            trip_mask=trip_mask,
            temperature=temperature,
            deterministic=deterministic,
        )
        
        return SACOutput(
            action_type=output['action_type'],
            reposition_target=output['reposition_target'],
            action_log_prob=output['action_log_prob'],
            reposition_log_prob=output['reposition_log_prob'],
            selected_trip=output.get('selected_trip'),
            trip_log_prob=output.get('trip_log_prob'),
            serve_scores=None,
            charge_scores=None,
            action_entropy=output.get('action_entropy'),
            hex_embeddings=output.get('hex_embeddings'),
            action_probs=output.get('action_probs'),          # full distribution
            action_log_probs_all=output.get('action_log_probs_all'),  # log probs all actions
            _encoded_context=output.get('_encoded_context'),
            _global_hex_context=output.get('_global_hex_context'),
        )
    
    def select_action(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        trip_features: Optional[torch.Tensor] = None,
        trip_mask: Optional[torch.Tensor] = None,
        station_features: Optional[torch.Tensor] = None,
        station_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
        vehicle_hex_ids: Optional[torch.Tensor] = None,
    ) -> SACOutput:
        """Select action with full output including serve/charge scores."""
        with torch.no_grad():
            output = self.forward(
                state, action_mask, reposition_mask,
                trip_features, trip_mask,
                station_features, station_mask,
                deterministic,
                temperature=temperature,
                vehicle_hex_ids=vehicle_hex_ids
            )
        return output  # Return full SACOutput for enhanced actor support
    
    def select_action_gcn(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        trip_mask: Optional[torch.Tensor] = None,
        vehicle_hex_ids: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> SACOutput:
        """Select action specifically for GCN actor.
        
        Returns per-vehicle actions [num_vehicles] instead of single action.
        """
        with torch.no_grad():
            output = self.forward(
                state, action_mask, reposition_mask,
                trip_mask=trip_mask,
                deterministic=deterministic,
                temperature=temperature,
                vehicle_hex_ids=vehicle_hex_ids
            )
        return output
    
    def select_action_simple(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple action selection returning only action_type and reposition_target."""
        with torch.no_grad():
            output = self.forward(state, action_mask, reposition_mask, deterministic=deterministic)
        return output.action_type, output.reposition_target
    
    def compute_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,  # Semi-MDP durations
        serve_vehicle_idx: Optional[torch.Tensor] = None,
        serve_trip_idx: Optional[torch.Tensor] = None,
        num_served: Optional[torch.Tensor] = None,
        charge_vehicle_idx: Optional[torch.Tensor] = None,
        charge_station_idx: Optional[torch.Tensor] = None,
        num_charged: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Extract action types for critic - actions shape: [batch, num_vehicles, 2]
        # Use aggregated action (mean action type per batch)
        if actions.dim() == 3:
            # actions[:, :, 0] = action_type, actions[:, :, 1] = reposition_target
            action_type_agg = actions[:, :, 0].float().mean(dim=1).long()
        elif actions.dim() == 2:
            action_type_agg = actions[:, 0].long()
        else:
            action_type_agg = actions.long()
        
        # Handle critic based on type (GCN vs flat)
        if self.use_gcn_critic:
            # === GCN Critic: Use structured state + per-vehicle actions ===

            # Parse states to structured format
            hex_features, vehicle_features, vehicle_hex_ids, context_features = self._parse_flat_state(states)
            next_hex_features, next_vehicle_features, next_vehicle_hex_ids, next_context_features = self._parse_flat_state(next_states)

            # Per-vehicle actions (no aggregation!)
            if actions.dim() == 3:
                per_vehicle_actions = actions[:, :, 0].long()  # [batch, num_vehicles]
            elif actions.dim() == 2:
                per_vehicle_actions = actions[:, 0].unsqueeze(1).expand(-1, self.num_vehicles).long()
            else:
                per_vehicle_actions = actions.unsqueeze(0).unsqueeze(1).expand(1, self.num_vehicles).long()

            with torch.no_grad():
                # Actor forward for next state
                next_output = self.forward(next_states, action_mask, reposition_mask)

                next_probs = next_output.action_probs          # [batch, V, 4]
                next_log_probs = next_output.action_log_probs_all  # [batch, V, 4]

                if next_probs is not None and next_log_probs is not None:
                    # === Full-distribution target (exact, no sampling variance) ===
                    # V(s') = Σ_a π(a|s') · [Q(s',a) − α·log π(a|s')]
                    target_module = self.critic_target.module if hasattr(self.critic_target, 'module') else self.critic_target
                    hex_emb_t1 = target_module.critic1.gcn(next_hex_features, self._adjacency_matrix)
                    hex_emb_t2 = target_module.critic2.gcn(next_hex_features, self._adjacency_matrix)

                    Q_next_all = []
                    for a_idx in range(self.action_dim):
                        a_tensor = torch.full_like(next_output.action_type, a_idx)
                        nq1, nq2 = target_module(
                            hex_features=next_hex_features,
                            vehicle_features=next_vehicle_features,
                            vehicle_hex_ids=next_vehicle_hex_ids,
                            context_features=next_context_features,
                            adjacency=self._adjacency_matrix,
                            actions=a_tensor,
                            hex_embeddings_q1=hex_emb_t1,
                            hex_embeddings_q2=hex_emb_t2,
                        )
                        Q_next_all.append(torch.min(nq1, nq2))
                    Q_next_all = torch.stack(Q_next_all, dim=-1)  # [batch, 4]

                    mean_next_probs = next_probs.mean(dim=1)        # [batch, 4]
                    mean_next_log_probs = next_log_probs.mean(dim=1)  # [batch, 4]

                    next_v = (mean_next_probs * (Q_next_all - self.alpha * mean_next_log_probs)).sum(dim=-1)
                else:
                    # Fallback: sampled action (if actor didn't return full distribution)
                    next_action_type = next_output.action_type
                    next_action_log_prob = next_output.action_log_prob.mean(dim=-1) if next_output.action_log_prob.dim() > 1 else next_output.action_log_prob
                    next_q1, next_q2 = self.critic_target(
                        hex_features=next_hex_features,
                        vehicle_features=next_vehicle_features,
                        vehicle_hex_ids=next_vehicle_hex_ids,
                        context_features=next_context_features,
                        adjacency=self._adjacency_matrix,
                        actions=next_action_type,
                    )
                    next_v = torch.min(next_q1, next_q2) - self.alpha * next_action_log_prob

                # Semi-MDP discounting
                if durations is not None:
                    discount = torch.pow(self.gamma, durations)
                else:
                    discount = self.gamma

                target_q = rewards + (1 - dones.float()) * discount * next_v

            # Current Q-value
            q1, q2 = self.critic(
                hex_features=hex_features,
                vehicle_features=vehicle_features,
                vehicle_hex_ids=vehicle_hex_ids,
                context_features=context_features,
                adjacency=self._adjacency_matrix,
                actions=per_vehicle_actions,
            )
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        else:
            # === Legacy Flat Critic: Use aggregated actions + assignment encoding ===

            assignment_kwargs = {
                'serve_vehicle_idx': serve_vehicle_idx,
                'serve_trip_idx': serve_trip_idx,
                'num_served': num_served,
                'charge_vehicle_idx': charge_vehicle_idx,
                'charge_station_idx': charge_station_idx,
                'num_charged': num_charged,
            }

            with torch.no_grad():
                next_output = self.forward(next_states, action_mask, reposition_mask)

                next_probs = next_output.action_probs          # [batch, V, 4] or None
                next_log_probs = next_output.action_log_probs_all  # [batch, V, 4] or None

                if next_probs is not None and next_log_probs is not None:
                    # Full-distribution target for legacy critic
                    batch_size_l = next_states.size(0) if next_states.dim() > 1 else 1
                    Q_next_all_l = []
                    for a_idx in range(self.action_dim):
                        a_tensor = torch.full((batch_size_l,), a_idx, device=next_states.device, dtype=torch.long)
                        nq1, nq2 = self.critic_target(next_states, a_tensor, **assignment_kwargs)
                        Q_next_all_l.append(torch.min(nq1, nq2))
                    Q_next_all_l = torch.stack(Q_next_all_l, dim=-1)  # [batch, 4]

                    mean_next_probs = next_probs.mean(dim=1)        # [batch, 4]
                    mean_next_log_probs = next_log_probs.mean(dim=1)  # [batch, 4]
                    next_v = (mean_next_probs * (Q_next_all_l - self.alpha * mean_next_log_probs)).sum(dim=-1)
                else:
                    # Fallback: sampled action
                    next_action_type = next_output.action_type
                    if next_action_type.dim() > 1:
                        next_action_type_agg = next_action_type.float().mean(dim=-1).long()
                    else:
                        next_action_type_agg = next_action_type
                    next_action_log_prob = next_output.action_log_prob.mean(dim=-1) if next_output.action_log_prob.dim() > 1 else next_output.action_log_prob
                    next_q1, next_q2 = self.critic_target(next_states, next_action_type_agg, **assignment_kwargs)
                    next_v = torch.min(next_q1, next_q2) - self.alpha * next_action_log_prob

                if durations is not None:
                    discount = torch.pow(self.gamma, durations)
                else:
                    discount = self.gamma

                target_q = rewards + (1 - dones.float()) * discount * next_v

            q1, q2 = self.critic(states, action_type_agg, **assignment_kwargs)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        return critic_loss
    
    def compute_actor_loss(
        self,
        states: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        serve_vehicle_idx: Optional[torch.Tensor] = None,
        serve_trip_idx: Optional[torch.Tensor] = None,
        num_served: Optional[torch.Tensor] = None,
        charge_vehicle_idx: Optional[torch.Tensor] = None,
        charge_station_idx: Optional[torch.Tensor] = None,
        num_charged: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Compute actor loss with REINFORCE for trip selection.
        
        Trip selection uses REINFORCE algorithm:
        - Sample trip from categorical distribution
        - Compute Q-value for that specific assignment
        - Use Q-value as reward signal: ∇J = E[∇log π(trip|state) * Q(state, action, trip)]
        
        Returns:
            Tuple of (total_actor_loss, negative_entropy, aux_loss_dict)
        """
        # Extract device - states can be dict or tensor depending on training mode
        if isinstance(states, dict):
            _device = next(iter(states.values())).device
        else:
            _device = states.device

        # FIX: Parse state ONCE here so the critic path reuses it without a second parse
        if self.use_gcn_critic:
            hex_features, vehicle_features, vehicle_hex_ids, context_features = self._parse_flat_state(states)

        actor_output = self.forward(
            states,
            action_mask,
            reposition_mask,
        )

        action_type = actor_output.action_type
        action_log_prob = actor_output.action_log_prob

        # ===== SAC-Discrete policy loss (Christodoulou 2019) =====
        # Full distribution: L = Σ_a π(a|s) · (α·log_π(a|s) − Q(s,a))
        action_probs_full = actor_output.action_probs          # [batch, V, 4] or None
        action_log_probs_full = actor_output.action_log_probs_all  # [batch, V, 4] or None
        q_stats = {}  # populated below for monitoring

        if self.use_gcn_critic:
            # hex_features etc. already parsed at top of this method — no second parse
            critic_module = self.critic.module if hasattr(self.critic, 'module') else self.critic

            if action_probs_full is not None and action_log_probs_full is not None:
                # Pre-compute GCN hex embeddings ONCE, reuse for all 4 action evaluations.
                # Saves 3/4 of the GCN cost (most expensive step in critic forward).
                with torch.no_grad():
                    hex_emb_q1 = critic_module.critic1.gcn(hex_features, self._adjacency_matrix)
                    hex_emb_q2 = critic_module.critic2.gcn(hex_features, self._adjacency_matrix)

                    # Per-vehicle Q for each action type [batch, V] per action
                    Q_pv_all = []
                    for a_idx in range(self.action_dim):
                        a_tensor = torch.full_like(action_type, a_idx)  # [batch, num_vehicles]
                        q1_pv, q2_pv = critic_module(
                            hex_features=hex_features,
                            vehicle_features=vehicle_features,
                            vehicle_hex_ids=vehicle_hex_ids,
                            context_features=context_features,
                            adjacency=self._adjacency_matrix,
                            actions=a_tensor,
                            hex_embeddings_q1=hex_emb_q1,
                            hex_embeddings_q2=hex_emb_q2,
                            return_per_vehicle=True,
                        )
                        Q_pv_all.append(torch.min(q1_pv, q2_pv))  # [batch, V]
                    Q_pv_all = torch.stack(Q_pv_all, dim=-1)  # [batch, V, 4]

                # Q-stats for monitoring (avoids extra forward pass in update())
                q_stats = {'q_mean': Q_pv_all.mean().item(), 'q_std': Q_pv_all.std().item()}

                # Per-vehicle SAC-discrete loss (Christodoulou 2019):
                # L_i = Σ_a π_i(a|s) · (α·log π_i(a|s) − Q_i(s,a))
                # Each vehicle gets its OWN gradient based on its features and Q-values.
                # action_probs_full: [batch, V, 4], action_log_probs_full: [batch, V, 4]
                per_vehicle_loss = (action_probs_full * (
                    self.alpha.detach() * action_log_probs_full - Q_pv_all
                )).sum(dim=-1)  # [batch, V]
                policy_loss = per_vehicle_loss.mean()  # mean over vehicles and batch
            else:
                # Fallback (actor didn't return full distribution): sampled loss, no Q-gradient
                action_log_prob_avg = action_log_prob.mean(dim=-1) if action_log_prob.dim() > 1 else action_log_prob
                with torch.no_grad():
                    q1, q2 = critic_module(
                        hex_features=hex_features,
                        vehicle_features=vehicle_features,
                        vehicle_hex_ids=vehicle_hex_ids,
                        context_features=context_features,
                        adjacency=self._adjacency_matrix,
                        actions=action_type,
                    )
                min_q = torch.min(q1, q2)
                q_stats = {'q_mean': min_q.mean().item(), 'q_std': min_q.std().item()}
                policy_loss = (self.alpha.detach() * action_log_prob_avg - min_q).mean()
        else:
            # Legacy Flat Critic path — needs aggregation and assignment building
            if action_type.dim() > 1:
                action_type_for_critic = action_type.float().mean(dim=-1).long()
                action_log_prob_avg = action_log_prob.mean(dim=-1)
            else:
                action_type_for_critic = action_type
                action_log_prob_avg = action_log_prob

            # Build assignments from current policy output
            selected_trip = actor_output.selected_trip
            current_serve_vehicle_idx = serve_vehicle_idx
            current_serve_trip_idx = serve_trip_idx

            if selected_trip is not None and action_type.dim() > 1:
                batch_size = action_type.size(0)
                max_serve = min(200, self.num_vehicles)
                serve_veh_list = torch.full((batch_size, max_serve), -1, dtype=torch.long, device=_device)
                serve_trip_list = torch.full((batch_size, max_serve), -1, dtype=torch.long, device=_device)

                serve_mask_all = (action_type == 1)
                serve_counts = serve_mask_all.sum(dim=1)
                num_served = serve_counts.clamp(max=max_serve)

                serve_ranking = serve_mask_all.float().cumsum(dim=1) * serve_mask_all.float()
                valid = (serve_ranking > 0) & (serve_ranking <= max_serve)
                ranking_idx = (serve_ranking - 1).long().clamp(min=0)

                batch_indices = torch.arange(batch_size, device=_device).unsqueeze(1).expand_as(action_type)
                veh_indices = torch.arange(self.num_vehicles, device=_device).unsqueeze(0).expand_as(action_type)

                serve_veh_list[batch_indices[valid], ranking_idx[valid]] = veh_indices[valid]
                serve_trip_list[batch_indices[valid], ranking_idx[valid]] = selected_trip[batch_indices[valid], veh_indices[valid]]

                current_serve_vehicle_idx = serve_veh_list
                current_serve_trip_idx = serve_trip_list

            assignment_kwargs = {
                'serve_vehicle_idx': current_serve_vehicle_idx if isinstance(current_serve_vehicle_idx, torch.Tensor) else serve_vehicle_idx,
                'serve_trip_idx': current_serve_trip_idx if isinstance(current_serve_trip_idx, torch.Tensor) else serve_trip_idx,
                'num_served': num_served,
                'charge_vehicle_idx': charge_vehicle_idx,
                'charge_station_idx': charge_station_idx,
                'num_charged': num_charged,
            }
            with torch.no_grad():
                q1, q2 = self.critic(states, action_type_for_critic, **assignment_kwargs)
            min_q = torch.min(q1, q2)
            q_stats = {'q_mean': min_q.mean().item(), 'q_std': min_q.std().item()}
            policy_loss = (self.alpha.detach() * action_log_prob_avg - min_q).mean()
        
        total_actor_loss = policy_loss
        
        # ===== Demand-based auxiliary loss for reposition head =====
        # The reposition head has NO gradient from SAC-discrete loss (disconnected).
        # Without this, the head stays at random init → random targets → bad outcomes
        # → Q(REPOSITION) stays low → death spiral to 0% reposition.
        # Fix: train the head to prefer high-demand hexes using hex_features demand.
        if (self.repos_aux_weight > 0 and self.use_gcn_critic
                and actor_output._encoded_context is not None
                and actor_output._global_hex_context is not None):
            encoded_ctx = actor_output._encoded_context       # [batch, V, hidden]
            global_ctx = actor_output._global_hex_context      # [batch, gcn_out]
            batch_sz = encoded_ctx.size(0)
            num_veh = encoded_ctx.size(1)

            # Subsample vehicles for memory (full [batch, V, num_hexes] is too large)
            n_aux = min(32, num_veh)
            aux_veh_idx = torch.randint(0, num_veh, (batch_sz, n_aux), device=_device)
            batch_range = torch.arange(batch_sz, device=_device).unsqueeze(1).expand(-1, n_aux)
            aux_ctx = encoded_ctx[batch_range, aux_veh_idx]    # [batch, n_aux, hidden]
            aux_global = global_ctx.unsqueeze(1).expand(-1, n_aux, -1)  # [batch, n_aux, gcn_out]
            aux_input = torch.cat([aux_ctx, aux_global], dim=-1)

            # Get reposition head (unwrap DDP if needed)
            actor_module = self.actor.module if hasattr(self.actor, 'module') else self.actor
            aux_repos_logits = actor_module.reposition_head(aux_input)  # [batch, n_aux, num_hexes]

            # Demand target from hex_features (already parsed above for GCN critic path)
            # hex_features[:, :, 2] = norm_demand (trip demand per hex, clamped [0,1])
            demand = hex_features[:, :, 2]  # [batch, num_hexes]
            # Temperature-scaled softmax: sharp target on high-demand hexes
            demand_target = F.softmax(demand * 10.0, dim=-1)  # [batch, num_hexes]

            # Soft cross-entropy: -sum(target * log_softmax(logits))
            aux_log_probs = F.log_softmax(aux_repos_logits, dim=-1)  # [batch, n_aux, num_hexes]
            demand_expanded = demand_target.unsqueeze(1).expand(-1, n_aux, -1)  # [batch, n_aux, num_hexes]
            repos_aux_loss = -(demand_expanded * aux_log_probs).sum(dim=-1).mean()

            total_actor_loss = total_actor_loss + self.repos_aux_weight * repos_aux_loss
            q_stats['repos_aux_loss'] = repos_aux_loss.item()

        # Get actual entropy of action distribution (more stable than sampled log_prob)
        if actor_output.action_entropy is not None:
            action_entropy = actor_output.action_entropy.mean()
        else:
            action_entropy = self.get_action_entropy(states, action_mask)
        
        # Return negative entropy for alpha tuning (entropy is positive, log_prob is negative)
        return total_actor_loss, -action_entropy, q_stats
    
    def get_action_entropy(
        self,
        states: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get action entropy for alpha tuning (mean entropy across vehicles)."""
        # Forward to get action entropy
        output = self.forward(states, action_mask)
        if output.action_entropy is not None:
            # Mean across vehicles and batch
            return output.action_entropy.mean()
        else:
            # Fallback: compute from log probs
            return -output.action_log_prob.mean()

    def compute_alpha_loss(self, neg_entropy: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha loss for automatic entropy tuning.
        
        neg_entropy = -H = E[log_prob] (negative of actual entropy)
        target_entropy = -ratio * log(|A|) (negative, e.g., -0.42 for ratio=0.30)
        
        SAC formula: L(α) = -E[α * (log_prob - target_entropy)]
                          = -E[α * (neg_entropy - target_entropy)]
        
        The NEGATIVE sign is crucial for gradient descent direction:
        - When entropy LOW (need more exploration): gradient < 0 → α increases
        - When entropy HIGH (need more exploitation): gradient > 0 → α decreases
        
        Without the negative sign, gradient descent goes the wrong direction!
        """
        alpha_loss = -(self.log_alpha * (neg_entropy - self.target_entropy).detach()).mean()
        return alpha_loss
    
    def _flatten_states(self, states) -> torch.Tensor:
        """Flatten dict states to tensor if needed."""
        if isinstance(states, dict):
            # states = {'vehicle': [B, V, F], 'hex': [B, H, F], 'context': [B, C]}
            batch_size = states['vehicle'].shape[0]
            vehicle_flat = states['vehicle'].view(batch_size, -1)
            hex_flat = states['hex'].view(batch_size, -1)
            context_flat = states['context']
            if context_flat.dim() == 1:
                context_flat = context_flat.unsqueeze(0).expand(batch_size, -1)
            return torch.cat([vehicle_flat, hex_flat, context_flat], dim=-1)
        return states
    
    def update(
        self,
        states,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states,
        dones: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        reposition_mask: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        # Assignment info for auxiliary loss
        serve_vehicle_idx: Optional[torch.Tensor] = None,
        serve_trip_idx: Optional[torch.Tensor] = None,
        num_served: Optional[torch.Tensor] = None,
        charge_vehicle_idx: Optional[torch.Tensor] = None,
        charge_station_idx: Optional[torch.Tensor] = None,
        num_charged: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update agent with a batch of transitions.
        
        Args:
            states: Current states
            actions: Actions taken
            rewards: Rewards received
            next_states: Next states
            dones: Episode done flags
            action_mask: Valid action mask
            reposition_mask: Valid reposition target mask
            durations: Action durations for Semi-MDP (paper Eq. 13)
            serve_vehicle_idx: Vehicle indices that served [batch, max_serve]
            serve_trip_idx: Trip indices they were assigned [batch, max_serve]
            num_served: Number of vehicles that served [batch]
            charge_vehicle_idx: Vehicle indices that charged [batch, max_charge]
            charge_station_idx: Station indices they were assigned [batch, max_charge]
            num_charged: Number of vehicles that charged [batch]
        """
        # Flatten dict states if needed
        states_flat = self._flatten_states(states)
        next_states_flat = self._flatten_states(next_states)
        
        # Normalize rewards for stable training
        rewards_normalized = self._normalize_rewards(rewards)
        
        # Handle durations for Semi-MDP
        if durations is None and self.use_semi_mdp:
            # Default duration of 1 step if not provided
            durations = torch.ones_like(rewards)
        
        critic_loss = self.compute_critic_loss(
            states_flat, actions, rewards_normalized, next_states_flat, dones,
            action_mask, reposition_mask, durations,
            # Pass assignments to critic
            serve_vehicle_idx, serve_trip_idx, num_served,
            charge_vehicle_idx, charge_station_idx, num_charged,
        )
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Compute actor loss with auxiliary assignment loss
        actor_loss, log_prob, aux_losses = self.compute_actor_loss(
            states_flat, action_mask, reposition_mask,
            serve_vehicle_idx, serve_trip_idx, num_served,
            charge_vehicle_idx, charge_station_idx, num_charged,
        )
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping for actor
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = self.compute_alpha_loss(log_prob)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        self.soft_update_target()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'log_prob': log_prob.mean().item(),
            'q_mean': aux_losses.get('q_mean', 0.0),
            'q_std': aux_losses.get('q_std', 0.0),
            'repos_aux_loss': aux_losses.get('repos_aux_loss', 0.0),
        }
    
    def soft_update_target(self):
        critic_module = self.critic.module if hasattr(self.critic, 'module') else self.critic
        target_module = self.critic_target.module if hasattr(self.critic_target, 'module') else self.critic_target
        
        for target_param, param in zip(target_module.parameters(), critic_module.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def hard_update_target(self):
        critic_module = self.critic.module if hasattr(self.critic, 'module') else self.critic
        target_module = self.critic_target.module if hasattr(self.critic_target, 'module') else self.critic_target
        
        target_module.load_state_dict(critic_module.state_dict())
    
    def save(self, path: str):
        actor_module = self.actor.module if hasattr(self.actor, 'module') else self.actor
        critic_module = self.critic.module if hasattr(self.critic, 'module') else self.critic
        target_module = self.critic_target.module if hasattr(self.critic_target, 'module') else self.critic_target
        
        torch.save({
            'actor': actor_module.state_dict(),
            'critic': critic_module.state_dict(),
            'critic_target': target_module.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_alpha else None,
        }, path)
    
    def load(self, path: str):
        actor_module = self.actor.module if hasattr(self.actor, 'module') else self.actor
        critic_module = self.critic.module if hasattr(self.critic, 'module') else self.critic
        target_module = self.critic_target.module if hasattr(self.critic_target, 'module') else self.critic_target
        
        checkpoint = torch.load(path, map_location=self.device)
        actor_module.load_state_dict(checkpoint['actor'])
        critic_module.load_state_dict(checkpoint['critic'])
        target_module.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.auto_alpha and checkpoint['alpha_optimizer'] is not None:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])


class DistributedSACAgent(SACAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_hexes: int,
        local_rank: int = 0,
        world_size: int = 1,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, num_hexes, **kwargs)
        self.local_rank = local_rank
        self.world_size = world_size
    
    def setup_distributed(self):
        if self.world_size > 1:
            self.actor = nn.parallel.DistributedDataParallel(
                self.actor,
                device_ids=[self.local_rank]
            )
            self.critic = nn.parallel.DistributedDataParallel(
                self.critic,
                device_ids=[self.local_rank]
            )
    
    def sync_gradients(self):
        if self.world_size > 1:
            for param in self.actor.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(
                        param.grad,
                        op=torch.distributed.ReduceOp.AVG
                    )
            for param in self.critic.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(
                        param.grad,
                        op=torch.distributed.ReduceOp.AVG
                    )
