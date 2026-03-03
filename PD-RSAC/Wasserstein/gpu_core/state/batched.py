"""Batched episode state for parallel simulation."""

import torch
from typing import List, Optional
from .fleet import TensorFleetState
from .trips import TensorTripState
from .stations import TensorStationState


class BatchedEpisodeState:
    """
    Manages multiple episode states for parallel simulation.
    
    Enables running N episodes simultaneously on GPU for
    maximum throughput during training.
    """
    
    def __init__(
        self,
        batch_size: int,
        num_vehicles: int,
        num_stations: int,
        max_trips: int = 500,
        device: str = "cuda",
        initial_soc: float = 80.0,
        max_soc: float = 100.0,
    ):
        self.batch_size = batch_size
        self.num_vehicles = num_vehicles
        self.device = torch.device(device)
        
        # Create batched tensors for all vehicles across all episodes
        # Shape: [batch_size, num_vehicles] for most tensors
        self.positions = torch.zeros(batch_size, num_vehicles, dtype=torch.long, device=self.device)
        self.socs = torch.full((batch_size, num_vehicles), initial_soc, dtype=torch.float32, device=self.device)
        self.status = torch.zeros(batch_size, num_vehicles, dtype=torch.int8, device=self.device)
        self.busy_until = torch.zeros(batch_size, num_vehicles, dtype=torch.int32, device=self.device)
        self.target_hex = torch.full((batch_size, num_vehicles), -1, dtype=torch.long, device=self.device)
        self.current_trip = torch.full((batch_size, num_vehicles), -1, dtype=torch.long, device=self.device)
        self.charging_station = torch.full((batch_size, num_vehicles), -1, dtype=torch.long, device=self.device)
        self.ongoing_revenue = torch.zeros(batch_size, num_vehicles, dtype=torch.float32, device=self.device)
        
        # Current step for each episode
        self.current_step = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        
        # Episode done flags
        self.done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Per-episode trip states (list since trip counts vary)
        self.trip_states: List[TensorTripState] = [
            TensorTripState(max_trips=max_trips, device=device)
            for _ in range(batch_size)
        ]
        
        # Shared station state (stations are same across episodes)
        self.station_state = TensorStationState(
            num_stations=num_stations,
            device=device,
        )
        
        # Per-episode station occupancy [batch_size, num_stations]
        self.station_occupied = torch.zeros(batch_size, num_stations, dtype=torch.int32, device=self.device)
        
        self.max_soc = max_soc
    
    def reset_episode(self, episode_idx: int, initial_positions: Optional[torch.Tensor] = None) -> None:
        """Reset a single episode to initial state."""
        self.socs[episode_idx] = 80.0
        self.status[episode_idx] = 0
        self.busy_until[episode_idx] = 0
        self.target_hex[episode_idx] = -1
        self.current_trip[episode_idx] = -1
        self.charging_station[episode_idx] = -1
        self.ongoing_revenue[episode_idx] = 0.0
        self.current_step[episode_idx] = 0
        self.done[episode_idx] = False
        self.station_occupied[episode_idx] = 0
        
        if initial_positions is not None:
            self.positions[episode_idx] = initial_positions
        else:
            self.positions[episode_idx] = 0
        
        self.trip_states[episode_idx].reset()
    
    def reset_all(self, initial_positions: Optional[torch.Tensor] = None) -> None:
        """Reset all episodes to initial state."""
        for i in range(self.batch_size):
            pos = initial_positions[i] if initial_positions is not None else None
            self.reset_episode(i, pos)
    
    def get_episode_state(self, episode_idx: int) -> TensorFleetState:
        """Extract single episode state as TensorFleetState."""
        state = TensorFleetState.__new__(TensorFleetState)
        state.num_vehicles = self.num_vehicles
        state.device = self.device
        state.max_soc = self.max_soc
        state.positions = self.positions[episode_idx]
        state.socs = self.socs[episode_idx]
        state.status = self.status[episode_idx]
        state.busy_until = self.busy_until[episode_idx]
        state.target_hex = self.target_hex[episode_idx]
        state.current_trip = self.current_trip[episode_idx]
        state.charging_station = self.charging_station[episode_idx]
        state.ongoing_revenue = self.ongoing_revenue[episode_idx]
        state.charge_power = torch.zeros(self.num_vehicles, dtype=torch.float32, device=self.device)
        return state
    
    def set_episode_state(self, episode_idx: int, state: TensorFleetState) -> None:
        """Set single episode state from TensorFleetState."""
        self.positions[episode_idx] = state.positions
        self.socs[episode_idx] = state.socs
        self.status[episode_idx] = state.status
        self.busy_until[episode_idx] = state.busy_until
        self.target_hex[episode_idx] = state.target_hex
        self.current_trip[episode_idx] = state.current_trip
        self.charging_station[episode_idx] = state.charging_station
        self.ongoing_revenue[episode_idx] = state.ongoing_revenue
    
    def get_available_mask(self) -> torch.Tensor:
        """Get available vehicle mask for all episodes [batch_size, num_vehicles]."""
        step_expanded = self.current_step.unsqueeze(1)  # [batch_size, 1]
        return (self.busy_until <= step_expanded) & (self.status == 0)
    
    def get_active_episodes(self) -> torch.Tensor:
        """Get indices of episodes that are not done."""
        return torch.where(~self.done)[0]
    
    def mark_done(self, episode_idx: int) -> None:
        """Mark episode as done."""
        self.done[episode_idx] = True
    
    def increment_step(self) -> None:
        """Increment step counter for all active episodes."""
        active = ~self.done
        self.current_step[active] += 1
    
    def get_batch_socs(self) -> torch.Tensor:
        """Get mean SoC for each episode [batch_size]."""
        return self.socs.mean(dim=1)
    
    def get_batch_status_counts(self) -> torch.Tensor:
        """Get status counts for each episode [batch_size, 5]."""
        counts = torch.zeros(self.batch_size, 5, dtype=torch.int32, device=self.device)
        for status_val in range(5):
            counts[:, status_val] = (self.status == status_val).sum(dim=1)
        return counts
    
    def to_feature_tensor(self) -> torch.Tensor:
        """Convert all episodes to feature tensor [batch_size, num_vehicles, feature_dim]."""
        features = torch.stack([
            self.positions.float() / 1000.0,
            self.socs / self.max_soc,
            self.status.float() / 4.0,
            (self.busy_until > 0).float(),
            (self.charging_station >= 0).float(),
            self.ongoing_revenue / 10.0,
        ], dim=2)
        return features
