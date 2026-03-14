"""
GPU-native RL Environment wrapper for EV Fleet simulation - V2.

Refactored version using modular components:
- TripManager: Trip loading and generation
- ActionProcessor: SERVE/CHARGE/REPOSITION execution  
- RewardComputer: Reward and penalty calculation

This version is cleaner and more maintainable than environment.py.
"""

import torch
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..config import Config
from ..state import TensorFleetState, TensorTripState, TensorStationState, VehicleStatus
from ..spatial import HexGrid
from ..spatial.assignment import TripAssigner, StationAssigner
from .dynamics import EnergyDynamics, TimeDynamics
from ..features.builder import FeatureBuilder
from ..data.real_trip_loader import RealTripLoader

# Modular components
from .trip_manager import TripManager
from .action_processor import ActionProcessor
from .reward import RewardComputer


@dataclass
class EnvInfo:
    """Information returned with each step."""
    trips_served: int = 0
    trips_dropped: int = 0
    trips_loaded: int = 0
    revenue: float = 0.0
    driving_cost: float = 0.0
    energy_cost: float = 0.0
    reposition_bonus: float = 0.0  # Demand-based reposition reward
    vehicles_charging: int = 0
    vehicles_serving: int = 0
    vehicles_idle: int = 0
    avg_soc: float = 0.0
    step: int = 0
    episode_done: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # Assignment info for auxiliary loss training
    serve_assignments: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    charge_assignments: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    # Stats for debugging
    num_serve_attempted: int = 0
    num_serve_success: int = 0
    num_charge_attempted: int = 0
    num_charge_success: int = 0


class GPUEnvironmentV2:
    """
    GPU-native environment for EV Fleet RL training - V2.
    
    Uses modular components for cleaner code:
    - TripManager: Handles trip lifecycle
    - ActionProcessor: Executes vehicle actions
    - RewardComputer: Calculates rewards/penalties
    
    Features:
    - GPU-accelerated state management
    - TripAssigner for optimal vehicle-trip matching
    - StationAssigner for vehicle-station matching
    - Support for EnhancedActor with preference-based assignment
    """
    
    def __init__(
        self,
        config: Config,
        hex_grid: HexGrid,
        trip_loader: Optional[RealTripLoader] = None,
        device: str = "cuda",
        use_hungarian: bool = False,
        max_pickup_distance: float = 5.0
    ):
        self.config = config
        self.hex_grid = hex_grid
        self.trip_loader = trip_loader
        self.device = torch.device(device)
        
        # Environment dimensions
        self.num_vehicles = config.environment.num_vehicles
        self.num_hexes = config.environment.num_hexes
        self.num_stations = config.environment.num_stations
        self.episode_steps = config.episode.steps_per_episode
        
        # Dynamics
        self.energy_dynamics = EnergyDynamics(
            energy_per_km=config.physics.energy_per_km,
            charge_power_kw=config.physics.charge_power_kw,
            max_soc=config.physics.max_soc,
            min_soc_reserve=config.physics.min_soc_reserve,
            device=device
        )
        self.time_dynamics = TimeDynamics(
            step_duration_minutes=config.episode.step_duration_minutes,
            avg_speed_kmh=config.physics.avg_speed_kmh,
            episode_duration_hours=config.episode.duration_hours,
            device=device
        )
        
        # Feature builder
        self.feature_builder = FeatureBuilder(
            hex_grid=hex_grid,
            num_vehicles=self.num_vehicles,
            max_soc=config.physics.max_soc,
            device=device
        )
        
        # State tensors (initialized in reset)
        self.fleet_state: Optional[TensorFleetState] = None
        self.trip_state: Optional[TensorTripState] = None
        self.station_state: Optional[TensorStationState] = None
        
        # Reset tracking
        self.current_step = 0
        self.episode_reward = torch.tensor(0.0, device=self.device)
        self.episode_info = EnvInfo()
        
        # State dimensions (must match FeatureBuilder output)
        # FeatureBuilder.build_vehicle_features returns [N, 16]
        #   - 1 position, 1 SOC, 5 status, 1 time_until_free, 1 is_charging
        #   - 1 low_soc, 1 high_soc, 1 norm_step, 1 busy_ratio
        #   - 3 spatial: demand_nearby, nearest_trip_dist, nearest_station_dist
        # FeatureBuilder.build_hex_features returns [H, 5]
        # FeatureBuilder.build_context_features returns [9]
        self._action_dim = 3  # SERVE=0, CHARGE=1, REPOSITION=2 (IDLE removed)
        self._vehicle_feature_dim = 16  # FIX: Matches GCNActor expectation
        self._hex_feature_dim = 5
        self._context_dim = 9
        self._state_dim = (
            self.num_vehicles * self._vehicle_feature_dim +
            self.num_hexes * self._hex_feature_dim +
            self._context_dim
        )
        
        # Assignment solvers
        self.max_pickup_distance = max_pickup_distance
        self.use_hungarian = use_hungarian
        self._trip_assigner: Optional[TripAssigner] = None
        self._station_assigner: Optional[StationAssigner] = None
        self._init_assigners()
        
        # Adjacency matrix for GCN (computed in _init_assigners)
        self._adjacency_matrix: Optional[torch.Tensor] = None
        self._compute_adjacency_matrix()
        
        # Modular components (initialized in reset)
        self._trip_manager: Optional[TripManager] = None
        self._action_processor: Optional[ActionProcessor] = None
        self._reward_computer = RewardComputer(
            config=config, device=self.device, adjacency_matrix=self._adjacency_matrix
        )
    
    def _init_assigners(self):
        """Initialize assignment solvers."""
        if hasattr(self.hex_grid, 'distance_matrix') and self.hex_grid.distance_matrix is not None:
            distance_matrix = self.hex_grid.distance_matrix._distances
            
            self._trip_assigner = TripAssigner(
                device=self.device,
                distance_matrix=distance_matrix,
                use_hungarian=self.use_hungarian,
                max_pickup_distance=self.max_pickup_distance
            )
            
            if self.num_stations > 0:
                # Initialize stations at hexes with highest pickup demand
                if hasattr(self, 'trip_loader') and self.trip_loader is not None and getattr(self.trip_loader, 'is_loaded', False) and self.trip_loader._df is not None:
                    # Get pickup counts per hex from real data
                    pickup_counts = self.trip_loader._df['pickup_hex_idx'].value_counts()
                    top_hexes = pickup_counts.nlargest(self.num_stations).index.values
                    station_hexes = torch.tensor(top_hexes, dtype=torch.long, device=self.device)
                    
                    # Pad with random hexes if not enough unique hexes (unlikely but safe)
                    if len(station_hexes) < self.num_stations:
                        missing = self.num_stations - len(station_hexes)
                        rand_hexes = torch.randint(0, self.num_hexes, (missing,), device=self.device)
                        station_hexes = torch.cat([station_hexes, rand_hexes])
                else:
                    station_hexes = torch.randint(0, self.num_hexes, (self.num_stations,), device=self.device)
                    
                ports_per_station = self.config.station.num_ports
                station_capacities = torch.full((self.num_stations,), ports_per_station, device=self.device)
                
                self._station_assigner = StationAssigner(
                    device=self.device,
                    distance_matrix=distance_matrix,
                    station_hexes=station_hexes,
                    station_capacities=station_capacities,
                    use_hungarian=self.use_hungarian
                )
    
    def _compute_adjacency_matrix(self):
        """Compute adjacency matrix for GCN based on hex distances.
        
        Two hexes are adjacent if their distance is below a threshold.
        Uses symmetric normalization for GCN: D^{-1/2} A D^{-1/2}
        """
        if not hasattr(self.hex_grid, 'distance_matrix') or self.hex_grid.distance_matrix is None:
            return
        
        distance_matrix = self.hex_grid.distance_matrix._distances
        if distance_matrix is None:
            return
        
        # Adjacent if distance < threshold (3 km - roughly 2 hex rings in H3 res 9)
        adjacency_threshold = 3.0  # km
        adj = (distance_matrix < adjacency_threshold).float()
        
        # Add self-loops
        adj = adj + torch.eye(self.num_hexes, device=self.device)
        adj = torch.clamp(adj, 0, 1)  # Ensure binary
        
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        self._adjacency_matrix = D_inv_sqrt @ adj @ D_inv_sqrt
        print(f"[GCN] Adjacency matrix computed: {self.num_hexes}x{self.num_hexes}, "
              f"avg degree: {deg.mean().item():.1f}")
    
    @property
    def adjacency_matrix(self) -> Optional[torch.Tensor]:
        """Get the adjacency matrix for GCN."""
        return self._adjacency_matrix
    
    @property
    def state_dim(self) -> int:
        return self._state_dim
    
    @property
    def action_dim(self) -> int:
        return self._action_dim
    
    def reset(
        self,
        seed: Optional[int] = None,
        episode_idx: int = 0,
    ) -> torch.Tensor:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            episode_idx: Episode index for variety in data loading
            
        Returns:
            Initial state tensor
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Initialize fleet state
        initial_soc = self.config.vehicle.initial_soc
        self.fleet_state = TensorFleetState(
            num_vehicles=self.num_vehicles,
            device=str(self.device),
            initial_soc=initial_soc
        )
        
        # Distribute vehicles uniformly across hexes
        initial_positions = self.hex_grid.distribute_vehicles(self.num_vehicles, method="uniform")
        self.fleet_state.positions.copy_(initial_positions.to(self.device))
        
        # Initialize trip state
        max_trips = self._calculate_max_trips()
        self.trip_state = TensorTripState(
            max_trips=max_trips,
            device=str(self.device)
        )
        
        # Initialize station state
        self.station_state = TensorStationState(
            num_stations=self.num_stations,
            device=str(self.device),
            num_ports=self.config.station.num_ports,
            max_power=self.config.station.max_power,
            electricity_price=self.config.station.electricity_price
        )
        
        # Sync the initialized locations with the station state
        if self._station_assigner is not None:
            self.station_state.set_locations(self._station_assigner.station_hexes)
            
        # Initialize TripManager
        self._trip_manager = TripManager(
            config=self.config,
            hex_grid=self.hex_grid,
            trip_state=self.trip_state,
            trip_loader=self.trip_loader,
            device=self.device,
            num_hexes=self.num_hexes,
            num_vehicles=self.num_vehicles,
            episode_steps=self.episode_steps,
        )
        self._trip_manager.reset(episode_idx)
        self._trip_manager.load_initial_trips()
        
        # Initialize ActionProcessor
        self._action_processor = ActionProcessor(
            config=self.config,
            hex_grid=self.hex_grid,
            fleet_state=self.fleet_state,
            trip_state=self.trip_state,
            station_state=self.station_state,
            energy_dynamics=self.energy_dynamics,
            device=self.device,
            trip_assigner=self._trip_assigner,
            station_assigner=self._station_assigner,
            max_pickup_distance=self.max_pickup_distance,
        )
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_info = EnvInfo()
        self.episode_info.trips_loaded = self._trip_manager.trips_loaded
        
        return self._get_state()
    
    def _calculate_max_trips(self) -> int:
        """Calculate max_trips buffer size."""
        if self.trip_loader is not None and self.trip_loader.is_loaded:
            total_trips = self.trip_loader.total_trips
            total_steps_in_data = 31 * 24 * 12
            avg_trips_per_step = total_trips / total_steps_in_data
            peak_multiplier = 1.6
            backlog_factor = 1.5
            max_trips = int(avg_trips_per_step * self.episode_steps * peak_multiplier * backlog_factor)
        else:
            trips_per_step = max(10, self.num_vehicles // 10)
            max_trips = int(trips_per_step * self.episode_steps * 2)
        
        return max(1000, min(max_trips, 500000))
    
    def step(
        self,
        action_type: torch.Tensor,
        reposition_target: Optional[torch.Tensor] = None,
        selected_trip: Optional[torch.Tensor] = None,  # NEW: GCN trip selections
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, EnvInfo]:
        """
        Execute one environment step.
        
        Args:
            action_type: [num_vehicles] action indices (0=SERVE, 1=CHARGE, 2=REPOSITION)
            reposition_target: [num_vehicles] target hex for REPOSITION actions
            
        Returns:
            next_state, reward, done, info
        """
        if action_type.dim() == 0:
            action_type = action_type.unsqueeze(0)
        if reposition_target is not None and reposition_target.dim() == 0:
            reposition_target = reposition_target.unsqueeze(0)
        
        # Load new trips for this step
        self._trip_manager.load_step_trips(self.current_step)
        
        # Get available vehicles
        available_mask = self.fleet_state.get_available_mask(self.current_step)
        
        # Create action masks  - SERVE=0, CHARGE=1, REPOSITION=2 (IDLE removed)
        serve_mask = (action_type == 0) & available_mask
        charge_mask = (action_type == 1) & available_mask
        reposition_mask = (action_type == 2) & available_mask
        
        # Initialize reward
        reward = torch.zeros(1, device=self.device)
        
        # === PROCESS ACTIONS ===
        
        # 1. SERVE actions
        trips_served, trip_revenue, serve_driving_cost, num_serve_failed, serve_veh_idx, serve_trip_idx = \
            self._action_processor.process_serve_actions(
                serve_mask, 
                self.current_step,
                selected_trip=selected_trip  # NEW: pass GCN trip selections
            )
        reward += trip_revenue
        reward -= serve_driving_cost
        
        # 2. CHARGE actions
        charge_travel_cost, num_charge_failed, charge_veh_idx, charge_station_idx = \
            self._action_processor.process_charge_actions(charge_mask)
        reward -= charge_travel_cost
        
        # 3. REPOSITION actions
        reposition_cost = torch.tensor(0.0, device=self.device)
        if reposition_target is not None:
            reposition_cost = self._action_processor.process_reposition_actions(
                reposition_mask, reposition_target, self.current_step
            )
            reward -= reposition_cost
            
            # Reposition penalty
            repos_penalty = reposition_mask.sum() * self._reward_computer.reposition_penalty
            reward -= repos_penalty
            # Demand-proportional dispatch bonus (immediate signal for dispatching to undersupplied hexes)
            repos_dispatch_bonus = self._reward_computer.compute_reposition_dispatch_bonus(
                reposition_mask, reposition_target, self.trip_state
            )
            reward += repos_dispatch_bonus
        
        # 4. Complete finished actions (CRITICAL: reset trip.assigned for completed vehicles)
        # Must be done BEFORE fleet_state.complete_actions resets current_trip
        serving_mask = (self.fleet_state.busy_until <= self.current_step) & \
                      (self.fleet_state.busy_until > 0) & \
                      (self.fleet_state.status == 1)  # VehicleStatus.SERVING
        
        if serving_mask.any():
            completed_trip_ids = self.fleet_state.current_trip[serving_mask]
            valid_trips = completed_trip_ids >= 0
            if valid_trips.any():
                trip_indices = completed_trip_ids[valid_trips]
                # Mark trips as completed (no longer assigned, removed from pool)
                self.trip_state.assigned[trip_indices] = False
                self.trip_state.assigned_vehicle[trip_indices] = -1
                self.trip_state.valid_mask[trip_indices] = False  # Trip completed
        
        # Track repositioning vehicles BEFORE complete_actions() resets status
        from ..state.fleet import VehicleStatus
        was_repositioning = self.fleet_state.status == VehicleStatus.REPOSITIONING
        will_complete = (self.fleet_state.busy_until <= self.current_step) & (self.fleet_state.busy_until > 0)
        completed_reposition_mask = was_repositioning & will_complete
        
        # Now complete vehicle actions (resets current_trip, status, etc.)
        completed_vehicles = self.fleet_state.complete_actions(self.current_step)
        
        # Compute reposition success bonus
        reposition_bonus = self._reward_computer.compute_reposition_bonus(
            completed_reposition_mask,
            self.fleet_state,
            self.trip_state
        )
        reward += reposition_bonus
        
        # 5. Update ongoing actions (charging progress, etc.)
        ongoing_charge_cost = self._action_processor.update_ongoing_actions(self.current_step)
        reward -= ongoing_charge_cost
        
        # === COMPUTE PENALTIES ===
        
        # Increment wait time
        self.trip_state.increment_wait()
        
        # Wait penalty
        wait_penalty = self._reward_computer.compute_wait_penalty(self.trip_state)
        reward -= wait_penalty
        
        # Drop penalty
        drop_penalty, trips_dropped = self._reward_computer.compute_drop_penalty(self.trip_state)
        reward -= drop_penalty
        
        # Low SOC penalty
        low_soc_penalty = self._reward_computer.compute_low_soc_penalty(self.fleet_state)
        reward -= low_soc_penalty
        
        # SERVE FAIL penalty
        serve_fail_penalty = num_serve_failed * self._reward_computer.serve_fail_penalty
        reward -= serve_fail_penalty
        
        # CHARGE FAIL penalty (use a small fixed rate since idle_penalty is removed)
        charge_fail_penalty = num_charge_failed * 0.5
        reward -= charge_fail_penalty
        
        # High SOC charge penalty (SOC > 60%)
        if self._reward_computer.high_soc_charge_penalty > 0:
            high_soc_charge = charge_mask & (self.fleet_state.socs > 60.0)
            reward -= high_soc_charge.sum() * self._reward_computer.high_soc_charge_penalty

        # Very high SOC charge penalty (SOC > 80%) — applied on top of above
        if self._reward_computer.very_high_soc_charge_penalty > 0:
            very_high_soc_charge = charge_mask & (self.fleet_state.socs > 80.0)
            reward -= very_high_soc_charge.sum() * self._reward_computer.very_high_soc_charge_penalty
        
        # Debug logging
        if self.current_step % 20 == 0 and self.current_step > 0:
            # Need to process serve_fail_penalty if it's a tensor
            sf_pen_item = serve_fail_penalty.item() if isinstance(serve_fail_penalty, torch.Tensor) else serve_fail_penalty
            print(f"  [Step {self.current_step}] Revenue: {trip_revenue.item():.1f}, "
                  f"ServeDrive: {serve_driving_cost.item():.1f}, "
                  f"ChargeTravel: {charge_travel_cost.item():.1f}, "
                  f"ChargeEnergy: {ongoing_charge_cost.item():.1f}, "
                  f"ReposBonus: {reposition_bonus.item():.2f}, ReposDispatch: {repos_dispatch_bonus.item():.2f}, ReposAct: {reposition_mask.sum().item():.0f}, "
                  f"ServeFail: {num_serve_failed} (Pen: {sf_pen_item:.1f}), ChargeFail: {num_charge_failed}, "
                  f"Wait: {wait_penalty.item():.1f}, "
                  f"Drop: {drop_penalty.item():.1f}, "
                  f"LowSOC: {low_soc_penalty.item():.1f}")
        
        # Scale reward
        reward = reward / self.config.reward.scale_factor
        
        # Update step
        self.current_step += 1
        done = self.current_step >= self.episode_steps
        
        # Update episode info (accumulate reward as tensor, convert at end)
        self.episode_reward += reward  # Keep as tensor
        total_charge_cost = charge_travel_cost + ongoing_charge_cost
        self._update_info(
            trips_served,
            trip_revenue.item(),
            serve_driving_cost.item(),
            total_charge_cost.item()
        )
        self.episode_info.trips_dropped += trips_dropped
        self.episode_info.trips_loaded = self._trip_manager.trips_loaded
        self.episode_info.reposition_bonus += reposition_bonus.item()
        
        # Store assignment info for auxiliary loss
        self.episode_info.serve_assignments = (serve_veh_idx, serve_trip_idx)
        self.episode_info.charge_assignments = (charge_veh_idx, charge_station_idx)
        
        # Store stats
        num_serve_attempted = serve_mask.sum().item()
        num_charge_attempted = charge_mask.sum().item()
        self.episode_info.num_serve_attempted = num_serve_attempted
        self.episode_info.num_serve_success = trips_served
        self.episode_info.num_charge_attempted = num_charge_attempted
        self.episode_info.num_charge_success = num_charge_attempted - num_charge_failed
        
        next_state = self._get_state()
        
        return next_state, reward, torch.tensor([done], device=self.device), self.episode_info
    
    def step_with_preferences(
        self,
        action_type: torch.Tensor,
        reposition_target: Optional[torch.Tensor] = None,
        serve_scores: Optional[torch.Tensor] = None,
        charge_scores: Optional[torch.Tensor] = None,
        preference_weight: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, EnvInfo]:
        """
        Execute step with actor preferences for assignment.
        
        Uses serve_scores and charge_scores from EnhancedActor for
        preference-weighted vehicle-trip and vehicle-station matching.
        """
        if action_type.dim() == 0:
            action_type = action_type.unsqueeze(0)
        if reposition_target is not None and reposition_target.dim() == 0:
            reposition_target = reposition_target.unsqueeze(0)
        
        # Load new trips
        self._trip_manager.load_step_trips(self.current_step)
        
        # Get available vehicles
        available_mask = self.fleet_state.get_available_mask(self.current_step)
        
        # Create action masks  - SERVE=0, CHARGE=1, REPOSITION=2 (IDLE removed)
        serve_mask = (action_type == 0) & available_mask
        charge_mask = (action_type == 1) & available_mask
        reposition_mask = (action_type == 2) & available_mask
        
        reward = torch.zeros(1, device=self.device)
        
        # === PROCESS ACTIONS WITH PREFERENCES ===
        
        # 1. SERVE with preferences
        if serve_scores is not None and self._trip_assigner is not None:
            trips_served, trip_revenue, serve_driving_cost, num_serve_failed, serve_veh_idx, serve_trip_idx = \
                self._action_processor.process_serve_actions_with_preferences(
                    serve_mask, self.current_step, serve_scores, preference_weight
                )
        else:
            trips_served, trip_revenue, serve_driving_cost, num_serve_failed, serve_veh_idx, serve_trip_idx = \
                self._action_processor.process_serve_actions(serve_mask, self.current_step)
        reward += trip_revenue
        reward -= serve_driving_cost
        
        # 2. CHARGE with preferences
        if charge_scores is not None and self._station_assigner is not None:
            charge_travel_cost, num_charge_failed, charge_veh_idx, charge_station_idx = \
                self._action_processor.process_charge_actions_with_preferences(
                    charge_mask, charge_scores, preference_weight
                )
        else:
            charge_travel_cost, num_charge_failed, charge_veh_idx, charge_station_idx = \
                self._action_processor.process_charge_actions(charge_mask)
        reward -= charge_travel_cost
        
        # 3. REPOSITION
        if reposition_target is not None:
            reposition_cost = self._action_processor.process_reposition_actions(
                reposition_mask, reposition_target, self.current_step
            )
            reward -= reposition_cost
            
            num_repositioning = reposition_mask.sum().item()
            repos_penalty = num_repositioning * self._reward_computer.reposition_penalty
            reward -= repos_penalty
            # Demand-proportional dispatch bonus (immediate signal for dispatching to undersupplied hexes)
            repos_dispatch_bonus = self._reward_computer.compute_reposition_dispatch_bonus(
                reposition_mask, reposition_target, self.trip_state
            )
            reward += repos_dispatch_bonus
        
        # 4. Update ongoing actions
        ongoing_charge_cost = self._action_processor.update_ongoing_actions(self.current_step)
        reward -= ongoing_charge_cost
        
        # === COMPUTE PENALTIES ===
        self.trip_state.increment_wait()
        
        wait_penalty = self._reward_computer.compute_wait_penalty(self.trip_state)
        reward -= wait_penalty
        
        drop_penalty, trips_dropped = self._reward_computer.compute_drop_penalty(self.trip_state)
        reward -= drop_penalty
        
        low_soc_penalty = self._reward_computer.compute_low_soc_penalty(self.fleet_state)
        reward -= low_soc_penalty
        
        # SERVE FAIL penalty
        serve_fail_penalty = num_serve_failed * self._reward_computer.serve_fail_penalty
        reward -= serve_fail_penalty
        
        # CHARGE FAIL penalty (fixed small rate; idle_penalty removed)
        charge_fail_penalty = num_charge_failed * 0.5
        reward -= charge_fail_penalty
        
        # High SOC charge penalty (SOC > 60%)
        if self._reward_computer.high_soc_charge_penalty > 0:
            high_soc_charge = charge_mask & (self.fleet_state.socs > 60.0)
            reward -= high_soc_charge.sum() * self._reward_computer.high_soc_charge_penalty

        # Very high SOC charge penalty (SOC > 80%) — applied on top of above
        if self._reward_computer.very_high_soc_charge_penalty > 0:
            very_high_soc_charge = charge_mask & (self.fleet_state.socs > 80.0)
            reward -= very_high_soc_charge.sum() * self._reward_computer.very_high_soc_charge_penalty
        
        # Debug logging
        if self.current_step % 20 == 0 and self.current_step > 0:
            sf_pen_item = serve_fail_penalty.item() if isinstance(serve_fail_penalty, torch.Tensor) else serve_fail_penalty
            print(f"  [Step {self.current_step}] Revenue: {trip_revenue.item():.1f}, "
                  f"ServeDrive: {serve_driving_cost.item():.1f}, "
                  f"ChargeTravel: {charge_travel_cost.item():.1f}, "
                  f"ChargeEnergy: {ongoing_charge_cost.item():.1f}, "
                  f"ServeFail: {num_serve_failed} (Pen: {sf_pen_item:.1f}), ChargeFail: {num_charge_failed}, "
                  f"Wait: {wait_penalty.item():.1f}, "
                  f"Drop: {drop_penalty.item():.1f}, "
                  f"LowSOC: {low_soc_penalty.item():.1f}")
        
        # Scale reward
        reward = reward / self.config.reward.scale_factor
        
        self.current_step += 1
        done = self.current_step >= self.episode_steps
        
        self.episode_reward += reward  # Keep as tensor
        total_charge_cost = charge_travel_cost + ongoing_charge_cost
        self._update_info(
            trips_served,
            trip_revenue.item(),
            serve_driving_cost.item(),
            total_charge_cost.item()
        )
        self.episode_info.trips_dropped += trips_dropped
        self.episode_info.trips_loaded = self._trip_manager.trips_loaded
        
        self.episode_info.serve_assignments = (serve_veh_idx, serve_trip_idx)
        self.episode_info.charge_assignments = (charge_veh_idx, charge_station_idx)
        
        # Store stats
        num_serve_attempted = serve_mask.sum().item()
        num_charge_attempted = charge_mask.sum().item()
        self.episode_info.num_serve_attempted = num_serve_attempted
        self.episode_info.num_serve_success = trips_served
        self.episode_info.num_charge_attempted = num_charge_attempted
        self.episode_info.num_charge_success = num_charge_attempted - num_charge_failed
        
        next_state = self._get_state()
        
        return next_state, reward, torch.tensor([done], device=self.device), self.episode_info
    
    def _get_state(self) -> torch.Tensor:
        """Build current state observation."""
        # Build state components using feature builder
        vehicle_features = self.feature_builder.build_vehicle_features(
            self.fleet_state, self.current_step, self.episode_steps,
            trips=self.trip_state,
            stations=self.station_state,
            max_pickup_distance=self.max_pickup_distance,
        )
        # Build hex features φ_h (phi_h) for spatial graph reasoning (paper Eq. 2)
        hex_features = self.feature_builder.build_hex_features(
            self.fleet_state, self.trip_state, self.station_state, self.current_step
        )
        context_features = self.feature_builder.build_context_features(
            self.current_step, self.episode_steps, self.fleet_state, self.trip_state
        )
        
        # Flatten and concatenate
        state = torch.cat([
            vehicle_features.view(-1),
            hex_features.view(-1),
            context_features.view(-1)
        ])
        
        return state
    
    def _update_info(self, trips_served: int, revenue: float, driving_cost: float, energy_cost: float):
        """Update episode info."""
        self.episode_info.trips_served += trips_served
        self.episode_info.revenue += revenue
        self.episode_info.driving_cost += driving_cost
        self.episode_info.energy_cost += energy_cost
        self.episode_info.step = self.current_step
        self.episode_info.avg_soc = self.fleet_state.socs.mean().item()
        
        # Count vehicle states
        status = self.fleet_state.status
        self.episode_info.vehicles_idle = (status == VehicleStatus.IDLE.value).sum().item()
        self.episode_info.vehicles_serving = (status == VehicleStatus.SERVING.value).sum().item()
        self.episode_info.vehicles_charging = (status == VehicleStatus.CHARGING.value).sum().item()
    
    def get_action_distribution(self, action_type: torch.Tensor) -> Dict[str, float]:
        """Get distribution of actions for logging."""
        total = len(action_type)
        if total == 0:
            return {"serve": 0, "charge": 0, "reposition": 0}
        
        return {
            "serve": (action_type == 0).sum().item() / total * 100,
            "charge": (action_type == 1).sum().item() / total * 100,
            "reposition": (action_type == 2).sum().item() / total * 100,
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current episode metrics."""
        return {
            "episode_reward": self.episode_reward.item() if isinstance(self.episode_reward, torch.Tensor) else self.episode_reward,
            "trips_served": self.episode_info.trips_served,
            "trips_dropped": self.episode_info.trips_dropped,
            "trips_loaded": self.episode_info.trips_loaded,
            "avg_soc": self.episode_info.avg_soc,
            "revenue": self.episode_info.revenue,
            "driving_cost": self.episode_info.driving_cost,
            "energy_cost": self.episode_info.energy_cost,
        }

    def _build_state_dict(self, flat_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert flat state tensor to dict format for replay buffer.
        
        Args:
            flat_state: Flattened state tensor from get_state()
            
        Returns:
            Dictionary with 'vehicle', 'hex', 'context' tensors
        """
        # Full state: vehicle + hex + context
        vehicle_size = self.num_vehicles * self._vehicle_feature_dim
        hex_size = self.num_hexes * self._hex_feature_dim
        
        vehicle_features = flat_state[:vehicle_size].view(self.num_vehicles, self._vehicle_feature_dim)
        hex_features = flat_state[vehicle_size:vehicle_size + hex_size].view(self.num_hexes, self._hex_feature_dim)
        context_features = flat_state[vehicle_size + hex_size:]
        
        return {
            'vehicle': vehicle_features,
            'hex': hex_features,
            'context': context_features
        }

    def get_available_actions(self) -> torch.Tensor:
        """Get mask of available actions per vehicle.
        
        Returns:
            [num_vehicles, action_dim] boolean tensor
        """
        mask = torch.ones(self.num_vehicles, self._action_dim, dtype=torch.bool, device=self.device)
        
        # Low SOC vehicles can't SERVE (0) or REPOSITION (2)
        low_soc_mask = self.fleet_state.socs < self.config.vehicle.soc_low_threshold
        mask[low_soc_mask, 0] = False  # SERVE=0
        mask[low_soc_mask, 2] = False  # REPOSITION=2
        
        # High SOC vehicles shouldn't CHARGE (1)
        high_soc_mask = self.fleet_state.socs > 90
        mask[high_soc_mask, 1] = False  # CHARGE=1
        
        # Busy vehicles can't do anything
        busy_mask = ~self.fleet_state.get_available_mask(self.current_step)
        mask[busy_mask] = False
        # No idle column to mask — IDLE action removed entirely

        return mask

    def render(self, mode: str = 'human') -> Optional[Dict]:
        """Render environment state for debugging."""
        if mode == 'human':
            print(f"\n=== Step {self.current_step}/{self.episode_steps} ===")
            print(f"Vehicles: Idle={self.episode_info.vehicles_idle}, "
                  f"Serving={self.episode_info.vehicles_serving}, "
                  f"Charging={self.episode_info.vehicles_charging}")
            print(f"Avg SOC: {self.episode_info.avg_soc:.1f}%")
            print(f"Trips Served: {self.episode_info.trips_served}")
            print(f"Trips Dropped: {self.episode_info.trips_dropped}")
            print(f"Episode Reward: {self.episode_reward.item() if isinstance(self.episode_reward, torch.Tensor) else self.episode_reward:.2f}")
            print(f"Max Pickup Distance: {self.max_pickup_distance:.1f} km")
            return None
        elif mode == 'dict':
            return {
                'step': self.current_step,
                'vehicles_idle': self.episode_info.vehicles_idle,
                'vehicles_serving': self.episode_info.vehicles_serving,
                'vehicles_charging': self.episode_info.vehicles_charging,
                'avg_soc': self.episode_info.avg_soc,
                'trips_served': self.episode_info.trips_served,
                'trips_dropped': self.episode_info.trips_dropped,
                'reward': self.episode_reward.item() if isinstance(self.episode_reward, torch.Tensor) else self.episode_reward,
                'positions': self.fleet_state.positions.cpu().numpy(),
                'soc': self.fleet_state.socs.cpu().numpy(),
                'max_pickup_distance': self.max_pickup_distance,
            }
        return None

    def set_pickup_distance(self, distance: float):
        """Set max pickup distance for curriculum learning.
        
        Args:
            distance: Maximum pickup distance in km
        """
        self.max_pickup_distance = distance
        
        # Also update TripAssigner
        if self._trip_assigner is not None:
            self._trip_assigner.max_pickup_distance = distance
        
        # Update ActionProcessor's reference  
        if self._action_processor is not None:
            self._action_processor.max_pickup_distance = distance


# Aliases for backward compatibility
GPUEnvironment = GPUEnvironmentV2
BatchedGPUEnvironment = GPUEnvironmentV2  # legacy alias; single-env V2 used everywhere
