"""Simulator orchestrator - main simulation loop."""

import torch
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass

from ..config import Config
from ..state import TensorFleetState, TensorTripState, TensorStationState, VehicleStatus
from ..spatial import HexGrid
from .actions import ActionExecutor, ActionType
from .transitions import TransitionEngine
from .dynamics import EnergyDynamics, TimeDynamics


@dataclass
class StepResult:
    reward: float
    done: bool
    info: Dict


class SimulatorOrchestrator:
    """
    Main simulator orchestrator.
    
    Coordinates all simulation components for efficient GPU execution.
    """
    
    def __init__(
        self,
        config: Config,
        hex_grid: HexGrid,
        device: str = "cuda",
    ):
        self.config = config
        self.device = torch.device(device)
        self.hex_grid = hex_grid
        
        # Initialize dynamics
        self.energy_dynamics = EnergyDynamics(
            energy_per_km=config.physics.energy_per_km,
            charge_power_kw=config.station.max_power,
            max_soc=config.physics.max_soc,
            min_soc_reserve=config.physics.min_soc_reserve,
            device=device,
        )
        
        self.time_dynamics = TimeDynamics(
            step_duration_minutes=config.episode.step_duration_minutes,
            avg_speed_kmh=config.physics.avg_speed_kmh,
            episode_duration_hours=config.episode.duration_hours,
            device=device,
        )
        
        # Get distance matrix from hex grid
        distance_matrix = hex_grid.distance_matrix.distances
        
        # Initialize action executor
        self.action_executor = ActionExecutor(
            distance_matrix=distance_matrix,
            avg_speed_kmh=config.physics.avg_speed_kmh,
            step_duration_minutes=config.episode.step_duration_minutes,
            energy_per_km=config.physics.energy_per_km,
            device=device,
        )
        
        # Initialize transition engine
        self.transition_engine = TransitionEngine(
            distance_matrix=distance_matrix,
            avg_speed_kmh=config.physics.avg_speed_kmh,
            step_duration_minutes=config.episode.step_duration_minutes,
            energy_per_km=config.physics.energy_per_km,
            charge_power_kw=config.station.max_power,
            device=device,
        )
        
        # Episode state
        self.fleet: Optional[TensorFleetState] = None
        self.trips: Optional[TensorTripState] = None
        self.stations: Optional[TensorStationState] = None
        self.current_step = 0
        
        # Statistics
        self.episode_stats = {}
    
    def reset(
        self,
        initial_positions: Optional[torch.Tensor] = None,
    ) -> TensorFleetState:
        """
        Reset simulator for new episode.
        
        Returns:
            Initial fleet state
        """
        # Create fleet state
        self.fleet = TensorFleetState(
            num_vehicles=self.config.environment.num_vehicles,
            device=self.device,
            initial_soc=self.config.vehicle.initial_soc,
            max_soc=self.config.physics.max_soc,
        )
        
        # Set initial positions
        if initial_positions is not None:
            self.fleet.positions = initial_positions.to(self.device)
        else:
            # Distribute uniformly
            self.fleet.positions = self.hex_grid.distribute_vehicles(
                self.config.environment.num_vehicles,
                method="uniform",
            )
        
        # Create trip state
        self.trips = TensorTripState(
            max_trips=500,  # Will be loaded each step
            device=self.device,
        )
        
        # Create station state
        self.stations = TensorStationState(
            num_stations=self.config.environment.num_stations,
            device=self.device,
            num_ports=self.config.station.num_ports,
            max_power=self.config.station.max_power,
            electricity_price=self.config.station.electricity_price,
        )
        
        self.current_step = 0
        self.episode_stats = {
            "total_revenue": 0.0,
            "total_trips_served": 0,
            "total_trips_dropped": 0,
            "total_energy_consumed": 0.0,
            "total_energy_charged": 0.0,
        }
        
        return self.fleet
    
    def load_trips(
        self,
        trip_ids: torch.Tensor,
        pickup_hexes: torch.Tensor,
        dropoff_hexes: torch.Tensor,
        fares: torch.Tensor,
        distances: torch.Tensor,
    ) -> None:
        """Load new trips for current step."""
        self.trips.load_trips(trip_ids, pickup_hexes, dropoff_hexes, fares, distances)
    
    def step(
        self,
        action_types: torch.Tensor,
        action_targets: torch.Tensor,
    ) -> StepResult:
        """
        Execute one simulation step.
        
        Args:
            action_types: [num_vehicles] ActionType values
            action_targets: [num_vehicles] target indices
            
        Returns:
            StepResult with reward, done flag, and info dict
        """
        # 1. Execute actions
        action_results = self.action_executor.execute_batch(
            self.fleet,
            self.trips,
            self.stations,
            action_types,
            action_targets,
            self.current_step,
        )
        
        # 2. Advance simulation
        transition_stats = self.transition_engine.step(
            self.fleet,
            self.trips,
            self.stations,
            self.current_step,
        )
        
        # 3. Drop expired trips
        dropped = self.trips.drop_expired(self.config.reward.max_wait_steps)
        transition_stats["dropped_trips"] = dropped
        
        # 4. Compute reward
        reward = self._compute_reward(action_results, transition_stats)
        
        # 5. Update episode stats
        self._update_stats(action_results, transition_stats, reward)
        
        # 6. Advance step
        self.current_step += 1
        done = self.current_step >= self.time_dynamics.steps_per_episode
        
        # 7. Build info dict
        info = {
            "step": self.current_step,
            "served": action_results["served_count"],
            "repositioned": action_results["repositioned_count"],
            "charged": action_results["charged_count"],
            "dropped": dropped,
            "revenue": transition_stats["revenue_earned"],
            "mean_soc": self.fleet.mean_soc,
            "available_vehicles": self.fleet.num_available,
        }
        
        return StepResult(reward=reward, done=done, info=info)
    
    def _compute_reward(
        self,
        action_results: Dict,
        transition_stats: Dict,
    ) -> float:
        """Compute step reward."""
        cfg = self.config.reward
        
        # Revenue from serving
        revenue = transition_stats["revenue_earned"]
        
        # Driving cost
        energy_consumed = transition_stats["energy_consumed"]
        driving_cost = energy_consumed * cfg.driving_cost_per_km  # Approximate
        
        # Charging cost
        energy_charged = transition_stats["energy_charged"]
        charging_cost = energy_charged * cfg.electricity_cost_per_kwh
        
        # Wait penalty for unserved trips
        unserved = self.trips.get_unassigned_mask().sum().item()
        wait_penalty = unserved * cfg.wait_penalty_per_step
        
        # Drop penalty
        dropped = transition_stats.get("dropped_trips", 0)
        drop_penalty = dropped * cfg.drop_penalty_per_order
        
        # Total reward
        reward = revenue - driving_cost - charging_cost - wait_penalty - drop_penalty
        
        # Scale
        reward = reward / cfg.scale_factor
        
        return reward
    
    def _update_stats(
        self,
        action_results: Dict,
        transition_stats: Dict,
        reward: float,
    ) -> None:
        """Update episode statistics."""
        self.episode_stats["total_revenue"] += transition_stats["revenue_earned"]
        self.episode_stats["total_trips_served"] += action_results["served_count"]
        self.episode_stats["total_trips_dropped"] += transition_stats.get("dropped_trips", 0)
        self.episode_stats["total_energy_consumed"] += transition_stats["energy_consumed"]
        self.episode_stats["total_energy_charged"] += transition_stats["energy_charged"]
    
    def get_state_features(self) -> torch.Tensor:
        """Get feature tensor for current state."""
        return self.fleet.to_feature_tensor()
    
    def get_available_actions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get available actions for each vehicle.
        
        Returns:
            (action_mask, action_targets) 
            - action_mask: [num_vehicles, num_action_types] bool mask
            - action_targets: Dict with target options per action type
        """
        available = self.fleet.get_available_mask(self.current_step)
        
        # Action mask: which actions are available
        num_vehicles = self.fleet.num_vehicles
        action_mask = torch.zeros(num_vehicles, 4, dtype=torch.bool, device=self.device)
        
        # WAIT is always available for available vehicles
        action_mask[:, ActionType.WAIT] = available
        
        # SERVE is available if there are unassigned trips
        has_trips = self.trips.num_active > 0
        action_mask[:, ActionType.SERVE] = available & has_trips
        
        # REPOSITION is always available (to any hex)
        action_mask[:, ActionType.REPOSITION] = available
        
        # CHARGE is available at stations with ports
        station_available = self.stations.get_available_mask()
        action_mask[:, ActionType.CHARGE] = available & station_available.any()
        
        return action_mask
    
    def get_episode_summary(self) -> Dict:
        """Get episode summary statistics."""
        return {
            **self.episode_stats,
            "total_steps": self.current_step,
            "final_mean_soc": self.fleet.mean_soc,
            "final_status_counts": self.fleet.get_status_counts(),
        }
