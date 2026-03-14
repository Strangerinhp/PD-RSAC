"""GPU-resident replay buffer for RL training."""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, NamedTuple


class Transition(NamedTuple):
    """Single transition tuple."""
    state: Dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: Dict[str, torch.Tensor]
    done: torch.Tensor


@dataclass
class ReplayBatch:
    """Batch of transitions for training."""
    states: Dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: Dict[str, torch.Tensor]
    dones: torch.Tensor
    weights: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None
    # Assignment info for auxiliary loss
    serve_vehicle_idx: Optional[torch.Tensor] = None  # [batch, max_serve]
    serve_trip_idx: Optional[torch.Tensor] = None     # [batch, max_serve]
    num_served: Optional[torch.Tensor] = None         # [batch]
    charge_vehicle_idx: Optional[torch.Tensor] = None # [batch, max_charge]
    charge_station_idx: Optional[torch.Tensor] = None # [batch, max_charge]
    num_charged: Optional[torch.Tensor] = None        # [batch]
    # Scenario info for WDRO
    scenario_demand: Optional[torch.Tensor] = None    # [batch, num_hexes]
    scenario_context: Optional[torch.Tensor] = None   # [batch, context_dim]


class GPUReplayBuffer:
    """
    GPU-resident replay buffer for efficient training.
    
    All data is stored on GPU to minimize CPU-GPU transfers.
    Supports prioritized experience replay.
    """
    
    def __init__(
        self,
        capacity: int,
        num_vehicles: int,
        vehicle_feature_dim: int,
        num_hexes: int,
        hex_feature_dim: int,
        context_dim: int,
        action_dim: int = 2,  # (action_type, target)
        prioritized: bool = True,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_annealing_steps: int = 100000,
        device: str = "cuda",
        # Assignment storage for auxiliary loss
        max_serve_per_step: int = 200,  # Max vehicles that can serve per step
        max_charge_per_step: int = 100,  # Max vehicles that can charge per step
        num_stations: int = 150,
    ):
        self.capacity = capacity
        self.device = torch.device(device)
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_annealing_steps = beta_annealing_steps
        
        self._current_beta = beta_start
        self._step = 0
        
        # Circular buffer position
        self._pos = 0
        self._size = 0
        
        # Pre-allocate GPU tensors
        # States
        self.vehicle_states = torch.zeros(
            capacity, num_vehicles, vehicle_feature_dim,
            dtype=torch.float32, device=self.device
        )
        self.hex_states = torch.zeros(
            capacity, num_hexes, hex_feature_dim,
            dtype=torch.float32, device=self.device
        )
        self.context_states = torch.zeros(
            capacity, context_dim,
            dtype=torch.float32, device=self.device
        )
        
        # Next states
        self.next_vehicle_states = torch.zeros(
            capacity, num_vehicles, vehicle_feature_dim,
            dtype=torch.float32, device=self.device
        )
        self.next_hex_states = torch.zeros(
            capacity, num_hexes, hex_feature_dim,
            dtype=torch.float32, device=self.device
        )
        self.next_context_states = torch.zeros(
            capacity, context_dim,
            dtype=torch.float32, device=self.device
        )
        
        # Actions, rewards, dones
        self.actions = torch.zeros(
            capacity, num_vehicles, action_dim,
            dtype=torch.long, device=self.device
        )
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        
        # === ASSIGNMENT STORAGE for auxiliary loss ===
        # Serve assignments: which vehicles got which trips
        self.max_serve = max_serve_per_step
        self.serve_vehicle_idx = torch.full(
            (capacity, max_serve_per_step), -1,
            dtype=torch.long, device=self.device
        )
        self.serve_trip_idx = torch.full(
            (capacity, max_serve_per_step), -1,
            dtype=torch.long, device=self.device
        )
        self.num_served = torch.zeros(capacity, dtype=torch.long, device=self.device)
        
        # Charge assignments: which vehicles got which stations
        self.max_charge = max_charge_per_step
        self.num_stations = num_stations
        self.charge_vehicle_idx = torch.full(
            (capacity, max_charge_per_step), -1,
            dtype=torch.long, device=self.device
        )
        self.charge_station_idx = torch.full(
            (capacity, max_charge_per_step), -1,
            dtype=torch.long, device=self.device
        )
        self.num_charged = torch.zeros(capacity, dtype=torch.long, device=self.device)
        
        # Priorities for PER
        if prioritized:
            self.priorities = torch.zeros(capacity, dtype=torch.float32, device=self.device)
            self.max_priority = 1.0
        
        # === SCENARIO STORAGE for WDRO ===
        # Store scenario features ξ = (demand, travel_time) for robust optimization
        # Simplified: Store hex-level demand and context features that represent scenario
        self.scenario_demand = torch.zeros(
            capacity, num_hexes,  # Demand per hex
            dtype=torch.float32, device=self.device
        )
        self.scenario_context = torch.zeros(
            capacity, context_dim,  # Time, weather, etc.
            dtype=torch.float32, device=self.device
        )
    
    def push(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool,
        serve_assignments: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        charge_assignments: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        scenario_demand: Optional[torch.Tensor] = None,
        scenario_context: Optional[torch.Tensor] = None,
    ) -> None:
        """Add transition to buffer.
        
        Args:
            state: Current state dict
            action: Actions taken
            reward: Reward received
            next_state: Next state dict
            done: Episode done flag
            serve_assignments: (vehicle_indices, trip_indices) for SERVE actions
            charge_assignments: (vehicle_indices, station_indices) for CHARGE actions
        """
        idx = self._pos
        
        # Store state
        self.vehicle_states[idx] = state["vehicle"]
        self.hex_states[idx] = state["hex"]
        self.context_states[idx] = state["context"]
        
        # Store next state
        self.next_vehicle_states[idx] = next_state["vehicle"]
        self.next_hex_states[idx] = next_state["hex"]
        self.next_context_states[idx] = next_state["context"]
        
        # Store action, reward, done
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        
        # Store scenario features for WDRO
        if scenario_demand is not None:
            self.scenario_demand[idx] = scenario_demand
        else:
            # Extract from hex state (demand features typically in hex state)
            # Assuming first feature is demand-related
            self.scenario_demand[idx] = state["hex"][:, 0] if "hex" in state else 0.0
        
        if scenario_context is not None:
            self.scenario_context[idx] = scenario_context
        else:
            # Use context state as scenario context
            self.scenario_context[idx] = state.get("context", torch.zeros(self.context_states.shape[1], device=self.device))
        
        # Store serve assignments
        if serve_assignments is not None:
            veh_idx, trip_idx = serve_assignments
            n = min(len(veh_idx), self.max_serve)
            self.serve_vehicle_idx[idx, :n] = veh_idx[:n]
            self.serve_trip_idx[idx, :n] = trip_idx[:n]
            self.num_served[idx] = n
            # Clear remaining
            if n < self.max_serve:
                self.serve_vehicle_idx[idx, n:] = -1
                self.serve_trip_idx[idx, n:] = -1
        else:
            self.serve_vehicle_idx[idx] = -1
            self.serve_trip_idx[idx] = -1
            self.num_served[idx] = 0
        
        # Store charge assignments
        if charge_assignments is not None:
            veh_idx, station_idx = charge_assignments
            n = min(len(veh_idx), self.max_charge)
            self.charge_vehicle_idx[idx, :n] = veh_idx[:n]
            self.charge_station_idx[idx, :n] = station_idx[:n]
            self.num_charged[idx] = n
            # Clear remaining
            if n < self.max_charge:
                self.charge_vehicle_idx[idx, n:] = -1
                self.charge_station_idx[idx, n:] = -1
        else:
            self.charge_vehicle_idx[idx] = -1
            self.charge_station_idx[idx] = -1
            self.num_charged[idx] = 0
        
        # Update priority
        if self.prioritized:
            self.priorities[idx] = self.max_priority
        
        # Update position
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
    
    def push_batch(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: Dict[str, torch.Tensor],
        dones: torch.Tensor,
    ) -> None:
        """Add batch of transitions to buffer (vectorized)."""
        batch_size = len(rewards)

        # Calculate indices (handle wrap-around)
        indices = torch.arange(batch_size, device=self.device) + self._pos
        indices = indices % self.capacity

        # Vectorized copy
        self.vehicle_states[indices] = states["vehicle"]
        self.hex_states[indices] = states["hex"]
        self.context_states[indices] = states["context"]
        self.next_vehicle_states[indices] = next_states["vehicle"]
        self.next_hex_states[indices] = next_states["hex"]
        self.next_context_states[indices] = next_states["context"]
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.dones[indices] = dones.float()

        # Update position and size
        self._pos = (self._pos + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)
    
    def sample(
        self,
        batch_size: int,
        adjacency: Optional[torch.Tensor] = None,
    ) -> ReplayBatch:
        """Sample batch of transitions."""
        if self.prioritized:
            return self._sample_prioritized(batch_size, adjacency)
        else:
            return self._sample_uniform(batch_size, adjacency)
    
    def _sample_uniform(
        self,
        batch_size: int,
        adjacency: Optional[torch.Tensor],
    ) -> ReplayBatch:
        """Sample uniformly from buffer."""
        indices = torch.randint(0, self._size, (batch_size,), device=self.device)
        
        states = {
            "vehicle": self.vehicle_states[indices],
            "hex": self.hex_states[indices],
            "context": self.context_states[indices],
        }
        if adjacency is not None:
            states["adjacency"] = adjacency
        
        next_states = {
            "vehicle": self.next_vehicle_states[indices],
            "hex": self.next_hex_states[indices],
            "context": self.next_context_states[indices],
        }
        if adjacency is not None:
            next_states["adjacency"] = adjacency
        
        return ReplayBatch(
            states=states,
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=next_states,
            dones=self.dones[indices],
            weights=None,
            indices=indices,
            # Assignment info for auxiliary loss
            serve_vehicle_idx=self.serve_vehicle_idx[indices],
            serve_trip_idx=self.serve_trip_idx[indices],
            num_served=self.num_served[indices],
            charge_vehicle_idx=self.charge_vehicle_idx[indices],
            charge_station_idx=self.charge_station_idx[indices],
            num_charged=self.num_charged[indices],
            # Scenario info for WDRO
            scenario_demand=self.scenario_demand[indices],
            scenario_context=self.scenario_context[indices],
        )
    
    def _sample_prioritized(
        self,
        batch_size: int,
        adjacency: Optional[torch.Tensor],
    ) -> ReplayBatch:
        """Sample based on priorities."""
        # Compute sampling probabilities
        priorities = self.priorities[:self._size]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # Compute importance sampling weights
        weights = (self._size * probs[indices]) ** (-self._current_beta)
        weights = weights / weights.max()  # Normalize
        
        states = {
            "vehicle": self.vehicle_states[indices],
            "hex": self.hex_states[indices],
            "context": self.context_states[indices],
        }
        if adjacency is not None:
            states["adjacency"] = adjacency
        
        next_states = {
            "vehicle": self.next_vehicle_states[indices],
            "hex": self.next_hex_states[indices],
            "context": self.next_context_states[indices],
        }
        if adjacency is not None:
            next_states["adjacency"] = adjacency
        
        return ReplayBatch(
            states=states,
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=next_states,
            dones=self.dones[indices],
            weights=weights,
            indices=indices,
            # Assignment info for auxiliary loss
            serve_vehicle_idx=self.serve_vehicle_idx[indices],
            serve_trip_idx=self.serve_trip_idx[indices],
            num_served=self.num_served[indices],
            charge_vehicle_idx=self.charge_vehicle_idx[indices],
            charge_station_idx=self.charge_station_idx[indices],
            num_charged=self.num_charged[indices],
            # Scenario info for WDRO
            scenario_demand=self.scenario_demand[indices],
            scenario_context=self.scenario_context[indices],
        )
    
    def update_priorities(
        self,
        indices: torch.Tensor,
        td_errors: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> None:
        """Update priorities based on TD errors."""
        if not self.prioritized:
            return
        
        priorities = torch.abs(td_errors) + epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())
    
    def step_beta(self) -> None:
        """Anneal beta for importance sampling."""
        self._step += 1
        progress = min(self._step / self.beta_annealing_steps, 1.0)
        self._current_beta = self.beta_start + progress * (self.beta_end - self.beta_start)
    
    def __len__(self) -> int:
        return self._size
    
    def clear(self) -> None:
        """Clear buffer."""
        self._pos = 0
        self._size = 0
        self._step = 0
        self._current_beta = self.beta_start
        if self.prioritized:
            self.priorities.fill_(0)
            self.max_priority = 1.0
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        stats = {
            "size": self._size,
            "capacity": self.capacity,
            "fill_ratio": self._size / self.capacity,
        }
        if self.prioritized:
            valid_priorities = self.priorities[:self._size]
            stats["mean_priority"] = valid_priorities.mean().item()
            stats["max_priority"] = self.max_priority
            stats["beta"] = self._current_beta
        return stats
