"""
Structured State Builder for GCN Actor and Critic.

Provides consistent state format matching paper's spatial reasoning approach:
- vehicle_features: [batch, num_vehicles, vehicle_feature_dim]
- hex_features: [batch, num_hexes, hex_feature_dim]
- vehicle_hex_ids: [batch, num_vehicles] - hex indices for gathering
- context_features: [batch, context_dim]
- adjacency: [num_hexes, num_hexes] - normalized GCN adjacency
"""

import torch
from typing import Dict, Optional, List

from ..state import TensorFleetState, TensorTripState, TensorStationState
from ..spatial import HexGrid
from .builder import FeatureBuilder


class StructuredStateBuilder:
    """
    Builds structured state dicts for GCN-based Actor and Critic.

    Ensures consistent state representation across:
    - Environment (observation generation)
    - Replay buffer (storage/retrieval)
    - Actor (policy inference)
    - Critic (Q-value estimation)
    """

    def __init__(
        self,
        hex_grid: HexGrid,
        num_vehicles: int,
        max_soc: float = 100.0,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.hex_grid = hex_grid
        self.num_vehicles = num_vehicles
        self.num_hexes = hex_grid.num_hexes

        # Underlying feature builder
        self.feature_builder = FeatureBuilder(
            hex_grid=hex_grid,
            num_vehicles=num_vehicles,
            max_soc=max_soc,
            device=device
        )

        # Cache adjacency matrix (same across all states)
        self._adjacency: Optional[torch.Tensor] = None

    @property
    def adjacency(self) -> torch.Tensor:
        """Get normalized adjacency matrix for GCN."""
        if self._adjacency is None:
            self._adjacency = self.hex_grid.get_normalized_adjacency()
        return self._adjacency

    def build_state(
        self,
        fleet: TensorFleetState,
        trips: TensorTripState,
        stations: TensorStationState,
        current_step: int,
        max_steps: int,
        max_pickup_distance: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Build structured state dict for single timestep.

        Returns:
            Dict with keys:
                - hex_features: [num_hexes, hex_feature_dim]
                - vehicle_features: [num_vehicles, vehicle_feature_dim]
                - vehicle_hex_ids: [num_vehicles] - vehicle positions (hex indices)
                - context_features: [context_dim]
                - adjacency: [num_hexes, num_hexes]
        """
        # Build features using underlying builder
        vehicle_features = self.feature_builder.build_vehicle_features(
            fleet, current_step, max_steps,
            trips=trips, stations=stations,
            max_pickup_distance=max_pickup_distance
        )

        hex_features = self.feature_builder.build_hex_features(
            fleet, trips, stations, current_step
        )

        context_features = self.feature_builder.build_context_features(
            current_step, max_steps, fleet, trips
        )

        # Extract vehicle positions as hex indices
        vehicle_hex_ids = fleet.positions  # [num_vehicles]

        return {
            "hex_features": hex_features,  # [H, 5]
            "vehicle_features": vehicle_features,  # [N, 16]
            "vehicle_hex_ids": vehicle_hex_ids,  # [N]
            "context_features": context_features,  # [9]
            "adjacency": self.adjacency,  # [H, H]
        }

    def build_batch(
        self,
        states: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Stack list of single states into batched format.

        Args:
            states: List of state dicts from build_state()

        Returns:
            Batched state dict with [batch_size, ...] tensors
        """
        if len(states) == 0:
            raise ValueError("Cannot batch empty state list")

        return {
            "hex_features": torch.stack([s["hex_features"] for s in states]),  # [B, H, 5]
            "vehicle_features": torch.stack([s["vehicle_features"] for s in states]),  # [B, N, 16]
            "vehicle_hex_ids": torch.stack([s["vehicle_hex_ids"] for s in states]),  # [B, N]
            "context_features": torch.stack([s["context_features"] for s in states]),  # [B, 9]
            "adjacency": states[0]["adjacency"],  # [H, H] - same for all
        }

    def to_flat_state(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Convert structured state to flat tensor (for backward compatibility).

        Args:
            state_dict: Structured state from build_state()

        Returns:
            Flat state tensor [state_dim]
        """
        vehicle_flat = state_dict["vehicle_features"].view(-1)
        hex_flat = state_dict["hex_features"].view(-1)
        context_flat = state_dict["context_features"].view(-1)

        return torch.cat([vehicle_flat, hex_flat, context_flat])

    def from_flat_state(
        self,
        flat_state: torch.Tensor,
        vehicle_hex_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parse flat state tensor back to structured format.

        Args:
            flat_state: Flattened state [state_dim] or [batch, state_dim]
            vehicle_hex_ids: Optional vehicle positions [num_vehicles] or [batch, num_vehicles]

        Returns:
            Structured state dict
        """
        if flat_state.dim() == 1:
            # Single state
            vehicle_feature_dim = 16
            hex_feature_dim = 5
            context_dim = 9

            vehicle_size = self.num_vehicles * vehicle_feature_dim
            hex_size = self.num_hexes * hex_feature_dim

            vehicle_features = flat_state[:vehicle_size].view(self.num_vehicles, vehicle_feature_dim)
            hex_features = flat_state[vehicle_size:vehicle_size + hex_size].view(self.num_hexes, hex_feature_dim)
            context_features = flat_state[vehicle_size + hex_size:vehicle_size + hex_size + context_dim]

            # Extract vehicle_hex_ids from vehicle_features if not provided
            if vehicle_hex_ids is None:
                # Heuristic: use normalized position (first feature) * num_hexes
                vehicle_hex_ids = (vehicle_features[:, 0] * self.num_hexes).long().clamp(0, self.num_hexes - 1)

            return {
                "hex_features": hex_features,
                "vehicle_features": vehicle_features,
                "vehicle_hex_ids": vehicle_hex_ids,
                "context_features": context_features,
                "adjacency": self.adjacency,
            }
        else:
            # Batched state [batch, state_dim]
            batch_size = flat_state.shape[0]
            vehicle_feature_dim = 16
            hex_feature_dim = 5
            context_dim = 9

            vehicle_size = self.num_vehicles * vehicle_feature_dim
            hex_size = self.num_hexes * hex_feature_dim

            vehicle_features = flat_state[:, :vehicle_size].view(batch_size, self.num_vehicles, vehicle_feature_dim)
            hex_features = flat_state[:, vehicle_size:vehicle_size + hex_size].view(batch_size, self.num_hexes, hex_feature_dim)
            context_features = flat_state[:, vehicle_size + hex_size:vehicle_size + hex_size + context_dim]

            # Extract vehicle_hex_ids if not provided
            if vehicle_hex_ids is None:
                vehicle_hex_ids = (vehicle_features[:, :, 0] * self.num_hexes).long().clamp(0, self.num_hexes - 1)

            return {
                "hex_features": hex_features,
                "vehicle_features": vehicle_features,
                "vehicle_hex_ids": vehicle_hex_ids,
                "context_features": context_features,
                "adjacency": self.adjacency,
            }

    def validate_state(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        """
        Validate state dict has correct structure and dimensions.

        Args:
            state_dict: State to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        required_keys = {"hex_features", "vehicle_features", "vehicle_hex_ids", "context_features", "adjacency"}
        if not required_keys.issubset(state_dict.keys()):
            raise ValueError(f"State missing required keys: {required_keys - set(state_dict.keys())}")

        # Check dimensions
        hex_feat = state_dict["hex_features"]
        veh_feat = state_dict["vehicle_features"]
        veh_ids = state_dict["vehicle_hex_ids"]
        ctx_feat = state_dict["context_features"]
        adj = state_dict["adjacency"]

        # Unbatched checks
        if hex_feat.dim() == 2:
            assert hex_feat.shape[0] == self.num_hexes, f"Expected {self.num_hexes} hexes, got {hex_feat.shape[0]}"
            assert hex_feat.shape[1] == 5, f"Expected hex_feature_dim=5, got {hex_feat.shape[1]}"
            assert veh_feat.shape[0] == self.num_vehicles, f"Expected {self.num_vehicles} vehicles"
            assert veh_feat.shape[1] == 16, f"Expected vehicle_feature_dim=16, got {veh_feat.shape[1]}"
            assert veh_ids.shape[0] == self.num_vehicles
            assert ctx_feat.shape[0] == 9, f"Expected context_dim=9, got {ctx_feat.shape[0]}"

        # Batched checks
        elif hex_feat.dim() == 3:
            batch_size = hex_feat.shape[0]
            assert hex_feat.shape[1] == self.num_hexes
            assert hex_feat.shape[2] == 5
            assert veh_feat.shape == (batch_size, self.num_vehicles, 16)
            assert veh_ids.shape == (batch_size, self.num_vehicles)
            assert ctx_feat.shape == (batch_size, 9)
        else:
            raise ValueError(f"Unexpected hex_features dim: {hex_feat.dim()}")

        assert adj.shape == (self.num_hexes, self.num_hexes), f"Adjacency shape mismatch"

        return True
