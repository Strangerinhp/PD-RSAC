"""
GCN-based Critic to match Actor's spatial structure.

FIXES:
1. Uses GCN to encode spatial hex features (like Actor)
2. Per-vehicle action encoding (no aggregation loss)
3. Consistent state representation with Actor
4. Coordination-aware Q-value estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .gcn import GCNEncoder


class GCNCritic(nn.Module):
    """
    GCN-based Critic matching Actor's spatial architecture.

    Key improvements over flat Critic:
    1. GCN backbone for hex embeddings (same as Actor)
    2. Per-vehicle Q-value computation (no aggregation before encoding)
    3. Spatial structure preservation
    4. Compatible with structured state dict
    """

    def __init__(
        self,
        num_hexes: int,
        hex_feature_dim: int,
        vehicle_feature_dim: int,
        context_dim: int,
        action_dim: int = 4,
        gcn_hidden_dim: int = 128,
        gcn_output_dim: int = 64,
        critic_hidden_dim: int = 256,
        dropout: float = 0.1,
        aggregation: str = 'mean',  # 'mean', 'sum', or 'weighted'
    ):
        """
        Initialize GCN-based Critic.

        Args:
            num_hexes: Number of hexagons
            hex_feature_dim: Hex feature dimension
            vehicle_feature_dim: Vehicle feature dimension
            context_dim: Global context dimension
            action_dim: Action space dimension
            gcn_hidden_dim: GCN hidden dimension
            gcn_output_dim: GCN output (hex embedding) dimension
            critic_hidden_dim: Critic MLP hidden dimension
            dropout: Dropout rate
            aggregation: How to aggregate per-vehicle Q-values to fleet Q
        """
        super().__init__()
        self.num_hexes = num_hexes
        self.action_dim = action_dim
        self.gcn_output_dim = gcn_output_dim
        self.aggregation = aggregation

        # ===== GCN Encoder (SHARED STRUCTURE WITH ACTOR) =====
        # Two-layer GCN with symmetric normalization
        self.gcn = GCNEncoder(
            input_dim=hex_feature_dim,
            hidden_dims=[gcn_hidden_dim],
            output_dim=gcn_output_dim,
            dropout=dropout,
            use_batch_norm=True,
            activation='silu'
        )

        # ===== Vehicle-Action Context Encoder =====
        # Combines: vehicle features + local hex context + action encoding
        # Input: vehicle_features (14) + local_hex (64) + context (9) + action_one_hot (4)
        context_input_dim = vehicle_feature_dim + gcn_output_dim + context_dim + action_dim

        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, critic_hidden_dim),
            nn.LayerNorm(critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.LayerNorm(critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ===== Q-Value Head =====
        # Outputs Q-value per vehicle
        self.q_head = nn.Sequential(
            nn.Linear(critic_hidden_dim, critic_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim // 2, 1)
        )

        # ===== Weighted Aggregation (optional) =====
        if aggregation == 'weighted':
            self.importance_head = nn.Sequential(
                nn.Linear(critic_hidden_dim, critic_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(critic_hidden_dim // 2, 1)
            )

    def forward(
        self,
        hex_features: torch.Tensor,  # [batch, num_hexes, hex_feature_dim]
        vehicle_features: torch.Tensor,  # [batch, num_vehicles, vehicle_feature_dim]
        vehicle_hex_ids: torch.Tensor,  # [batch, num_vehicles] hex indices
        context_features: torch.Tensor,  # [batch, context_dim]
        adjacency: torch.Tensor,  # [num_hexes, num_hexes]
        actions: torch.Tensor,  # [batch, num_vehicles] action types
        hex_embeddings: Optional[torch.Tensor] = None,  # pre-computed [batch, num_hexes, gcn_out_dim]
        return_per_vehicle: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with structured state.

        Args:
            hex_features: Hex spatial features
            vehicle_features: Per-vehicle features
            vehicle_hex_ids: Vehicle positions (hex indices)
            context_features: Global context
            adjacency: Hex graph adjacency matrix
            actions: Per-vehicle action types
            return_per_vehicle: If True, return [batch, V] per-vehicle Q instead of [batch] fleet Q

        Returns:
            q_values: [batch] fleet Q-values or [batch, V] per-vehicle Q-values
        """
        batch_size, num_vehicles = vehicle_features.shape[0], vehicle_features.shape[1]

        # ===== 1. GCN Encoding (SAME AS ACTOR) =====
        if hex_embeddings is None:
            hex_embeddings = self.gcn(hex_features, adjacency)  # [batch, num_hexes, 64]

        # ===== 2. Gather Local Hex Context per Vehicle =====
        # Create batch indices for gathering
        batch_indices = torch.arange(batch_size, device=hex_features.device).unsqueeze(1).expand(-1, num_vehicles)

        # Gather local hex embeddings for each vehicle
        local_hex_context = hex_embeddings[batch_indices, vehicle_hex_ids]  # [batch, num_vehicles, 64]

        # ===== 3. Expand Global Context =====
        context_expanded = context_features.unsqueeze(1).expand(-1, num_vehicles, -1)  # [batch, num_vehicles, 9]

        # ===== 4. Encode Actions =====
        action_one_hot = F.one_hot(actions.long(), self.action_dim).float()  # [batch, num_vehicles, 4]

        # ===== 5. Combine All Features =====
        vehicle_action_context = torch.cat([
            vehicle_features,      # [batch, num_vehicles, 14]
            local_hex_context,     # [batch, num_vehicles, 64] - FROM GCN!
            context_expanded,      # [batch, num_vehicles, 9]
            action_one_hot        # [batch, num_vehicles, 4]
        ], dim=-1)  # [batch, num_vehicles, 91]

        # ===== 6. Encode Context =====
        encoded_context = self.context_encoder(vehicle_action_context)  # [batch, num_vehicles, 256]

        # ===== 7. Compute Per-Vehicle Q-Values =====
        q_per_vehicle = self.q_head(encoded_context).squeeze(-1)  # [batch, num_vehicles]

        if return_per_vehicle:
            return q_per_vehicle  # [batch, V]

        # ===== 8. Aggregate to Fleet Q-Value =====
        if self.aggregation == 'mean':
            q_fleet = q_per_vehicle.mean(dim=1)  # [batch]
        elif self.aggregation == 'sum':
            q_fleet = q_per_vehicle.sum(dim=1)  # [batch]
        elif self.aggregation == 'weighted':
            # Importance-weighted aggregation
            importance_logits = self.importance_head(encoded_context).squeeze(-1)  # [batch, num_vehicles]
            importance_weights = F.softmax(importance_logits, dim=1)  # [batch, num_vehicles]
            q_fleet = (q_per_vehicle * importance_weights).sum(dim=1)  # [batch]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return q_fleet

    def forward_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience method accepting state dict directly.

        Args:
            state_dict: Dict with keys: hex_features, vehicle_features,
                        vehicle_hex_ids, context_features, adjacency
            actions: [batch, num_vehicles] action types

        Returns:
            q_values: [batch] fleet Q-values
        """
        return self.forward(
            hex_features=state_dict['hex_features'],
            vehicle_features=state_dict['vehicle_features'],
            vehicle_hex_ids=state_dict['vehicle_hex_ids'],
            context_features=state_dict['context_features'],
            adjacency=state_dict['adjacency'],
            actions=actions
        )


class GCNTwinCritic(nn.Module):
    """
    Twin GCN Critics for SAC (reduce overestimation bias).
    """

    def __init__(
        self,
        num_hexes: int,
        hex_feature_dim: int,
        vehicle_feature_dim: int,
        context_dim: int,
        action_dim: int = 4,
        gcn_hidden_dim: int = 128,
        gcn_output_dim: int = 64,
        critic_hidden_dim: int = 256,
        dropout: float = 0.1,
        aggregation: str = 'mean',
    ):
        super().__init__()

        # Twin critics
        self.critic1 = GCNCritic(
            num_hexes=num_hexes,
            hex_feature_dim=hex_feature_dim,
            vehicle_feature_dim=vehicle_feature_dim,
            context_dim=context_dim,
            action_dim=action_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            gcn_output_dim=gcn_output_dim,
            critic_hidden_dim=critic_hidden_dim,
            dropout=dropout,
            aggregation=aggregation
        )

        self.critic2 = GCNCritic(
            num_hexes=num_hexes,
            hex_feature_dim=hex_feature_dim,
            vehicle_feature_dim=vehicle_feature_dim,
            context_dim=context_dim,
            action_dim=action_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            gcn_output_dim=gcn_output_dim,
            critic_hidden_dim=critic_hidden_dim,
            dropout=dropout,
            aggregation=aggregation
        )

    def forward(
        self,
        hex_features: torch.Tensor,
        vehicle_features: torch.Tensor,
        vehicle_hex_ids: torch.Tensor,
        context_features: torch.Tensor,
        adjacency: torch.Tensor,
        actions: torch.Tensor,
        hex_embeddings_q1: Optional[torch.Tensor] = None,  # pre-computed for critic1
        hex_embeddings_q2: Optional[torch.Tensor] = None,  # pre-computed for critic2
        return_per_vehicle: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both critics.

        Returns:
            q1, q2: [batch] Q-values from both critics (or [batch, V] if return_per_vehicle)
        """
        q1 = self.critic1(hex_features, vehicle_features, vehicle_hex_ids,
                         context_features, adjacency, actions,
                         hex_embeddings=hex_embeddings_q1,
                         return_per_vehicle=return_per_vehicle)
        q2 = self.critic2(hex_features, vehicle_features, vehicle_hex_ids,
                         context_features, adjacency, actions,
                         hex_embeddings=hex_embeddings_q2,
                         return_per_vehicle=return_per_vehicle)
        return q1, q2

    def forward_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method with state dict."""
        q1 = self.critic1.forward_dict(state_dict, actions)
        q2 = self.critic2.forward_dict(state_dict, actions)
        return q1, q2

    def q1(
        self,
        hex_features: torch.Tensor,
        vehicle_features: torch.Tensor,
        vehicle_hex_ids: torch.Tensor,
        context_features: torch.Tensor,
        adjacency: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Get Q-value from first critic only."""
        return self.critic1(hex_features, vehicle_features, vehicle_hex_ids,
                           context_features, adjacency, actions)

    def min_q(
        self,
        hex_features: torch.Tensor,
        vehicle_features: torch.Tensor,
        vehicle_hex_ids: torch.Tensor,
        context_features: torch.Tensor,
        adjacency: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Get minimum Q-value from twin critics."""
        q1, q2 = self.forward(hex_features, vehicle_features, vehicle_hex_ids,
                             context_features, adjacency, actions)
        return torch.min(q1, q2)

    def min_q_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Get min Q-value with state dict."""
        q1, q2 = self.forward_dict(state_dict, actions)
        return torch.min(q1, q2)


# Backward compatibility wrapper
class GCNCriticWrapper(nn.Module):
    """
    Wrapper to make GCNCritic compatible with flat state interface.

    This allows gradual migration from flat to structured state.

    DEPRECATED: This wrapper is for backward compatibility only.
    Use GCNTwinCritic directly with structured state instead.
    """

    def __init__(
        self,
        num_hexes: int,
        num_vehicles: int,
        hex_feature_dim: int = 5,
        vehicle_feature_dim: int = 16,  # FIX: Match feature builder output
        context_dim: int = 9,
        adjacency: Optional[torch.Tensor] = None,
        **gcn_critic_kwargs
    ):
        super().__init__()
        self.num_hexes = num_hexes
        self.num_vehicles = num_vehicles
        self.hex_feature_dim = hex_feature_dim
        self.vehicle_feature_dim = vehicle_feature_dim
        self.context_dim = context_dim

        # Store adjacency matrix (must be set via set_adjacency() if not provided)
        self._adjacency = adjacency

        self.gcn_critic = GCNTwinCritic(
            num_hexes=num_hexes,
            hex_feature_dim=hex_feature_dim,
            vehicle_feature_dim=vehicle_feature_dim,
            context_dim=context_dim,
            **gcn_critic_kwargs
        )

    def set_adjacency(self, adjacency: torch.Tensor):
        """Set adjacency matrix for GCN."""
        self._adjacency = adjacency

    def _parse_flat_state(self, state_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parse flat state back to structured format.

        Flat state format:
        [vehicle_features (V×14), hex_features (H×5), context (9)]
        """
        batch_size = state_flat.shape[0]

        vehicle_size = self.num_vehicles * self.vehicle_feature_dim
        hex_size = self.num_hexes * self.hex_feature_dim

        # Parse components
        vehicle_features = state_flat[:, :vehicle_size].view(batch_size, self.num_vehicles, self.vehicle_feature_dim)
        hex_features = state_flat[:, vehicle_size:vehicle_size + hex_size].view(batch_size, self.num_hexes, self.hex_feature_dim)
        context_features = state_flat[:, vehicle_size + hex_size:vehicle_size + hex_size + self.context_dim]

        # Extract vehicle_hex_ids from vehicle_features (heuristic)
        # CRITICAL FIX: Index 0 is normalized position (not index 1 which is SOC!)
        vehicle_hex_ids = (vehicle_features[:, :, 0] * self.num_hexes).long().clamp(0, self.num_hexes - 1)

        # Use stored adjacency matrix
        if self._adjacency is None:
            raise RuntimeError(
                "Adjacency matrix not set! Call set_adjacency() before using GCNCriticWrapper. "
                "This typically happens if the wrapper is used before environment initialization."
            )
        adjacency = self._adjacency

        return {
            'hex_features': hex_features,
            'vehicle_features': vehicle_features,
            'vehicle_hex_ids': vehicle_hex_ids,
            'context_features': context_features,
            'adjacency': adjacency
        }

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with flat state (backward compatible).

        Args:
            state: [batch, state_dim] flat state
            action: [batch, num_vehicles] actions

        Returns:
            q1, q2: Twin Q-values
        """
        state_dict = self._parse_flat_state(state)
        return self.gcn_critic.forward_dict(state_dict, action)
