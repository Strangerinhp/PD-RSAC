"""
Action Processing for EV Fleet Environment.

Handles SERVE, CHARGE, REPOSITION action execution.
Separated from main environment for modularity.
"""

import torch
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from ..state import TensorFleetState, TensorTripState, TensorStationState
    from ..spatial import HexGrid
    from ..spatial.assignment import TripAssigner, StationAssigner
    from .dynamics import EnergyDynamics


class ActionProcessor:
    """
    Processes vehicle actions: SERVE, CHARGE, REPOSITION.
    
    Features:
    - Vectorized batch processing for GPU efficiency
    - Optional TripAssigner/StationAssigner for optimal matching
    - Energy consumption tracking
    """
    
    def __init__(
        self,
        config: 'Config',
        hex_grid: 'HexGrid',
        fleet_state: 'TensorFleetState',
        trip_state: 'TensorTripState',
        station_state: 'TensorStationState',
        energy_dynamics: 'EnergyDynamics',
        device: torch.device,
        trip_assigner: Optional['TripAssigner'] = None,
        station_assigner: Optional['StationAssigner'] = None,
        max_pickup_distance: float = 10.0,
    ):
        self.config = config
        self.hex_grid = hex_grid
        self.fleet_state = fleet_state
        self.trip_state = trip_state
        self.station_state = station_state
        self.energy_dynamics = energy_dynamics
        self.device = device
        self.trip_assigner = trip_assigner
        self.station_assigner = station_assigner
        
        self.num_stations = config.environment.num_stations
        self.max_soc = config.physics.max_soc
        self.max_pickup_distance = max_pickup_distance  # For curriculum learning
    
    def process_serve_actions(
        self,
        serve_mask: torch.Tensor,
        current_step: int,
        selected_trip: Optional[torch.Tensor] = None,  # NEW: GCN trip selections [num_vehicles]
    ) -> Tuple[int, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Process serve actions using GCN trip selections with Hungarian conflict resolution.
        
        Args:
            serve_mask: Boolean mask of vehicles choosing SERVE
            current_step: Current simulation step
            selected_trip: [num_vehicles] GCN's trip selection per vehicle (trip indices)
            
        Returns:
            (trips_served, total_revenue, driving_cost, num_serve_failed, 
             matched_vehicles, matched_trips)
        """
        serve_indices = serve_mask.nonzero(as_tuple=True)[0]
        num_attempted = len(serve_indices)
        
        empty_tensor = torch.tensor([], dtype=torch.long, device=self.device)
        
        if num_attempted == 0:
            return 0, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), 0, empty_tensor, empty_tensor
        
        # Get unassigned trips
        unassigned_mask = self.trip_state.get_unassigned_mask()
        if not unassigned_mask.any():
            return 0, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), num_attempted, empty_tensor, empty_tensor
        
        trip_indices = unassigned_mask.nonzero(as_tuple=True)[0]
        
        # Use GCN selections if available, otherwise fallback to greedy distance
        if selected_trip is not None:
            matched_vehicles, matched_trips = self._match_with_gcn_and_hungarian(
                serve_indices, selected_trip, trip_indices
            )
        else:
            # Fallback to greedy distance matching (for backwards compatibility)
            matched_vehicles, matched_trips = self._greedy_distance_matching(
                serve_indices, trip_indices
            )
        
        trips_served = len(matched_trips)
        if trips_served == 0:
            return 0, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), num_attempted, empty_tensor, empty_tensor
        
        # Get trip info
        fares = self.trip_state.fare[matched_trips]
        trip_distances = self.trip_state.distance_km[matched_trips]
        dropoff_hexes = self.trip_state.dropoff_hex[matched_trips]
        pickup_hexes = self.trip_state.pickup_hex[matched_trips]
        
        # Calculate pickup distances (vehicle current position → pickup point)
        vehicle_positions = self.fleet_state.positions[matched_vehicles]
        pickup_distances = self.hex_grid.distance_matrix._distances[
            vehicle_positions, pickup_hexes
        ]
        
        # Calculate total distance and duration
        total_distances = pickup_distances + trip_distances
        avg_speed_kmh = self.config.physics.avg_speed_kmh
        step_minutes = self.config.episode.step_duration_minutes
        
        duration_hours = total_distances / avg_speed_kmh
        duration_minutes = duration_hours * 60.0
        durations = (duration_minutes / step_minutes).clamp(min=1).long()
        
        # Assign trips
        self.trip_state.assigned[matched_trips] = True
        self.trip_state.assigned_vehicle[matched_trips] = matched_vehicles
        
        # Set serving status
        busy_until = (current_step + durations).int()
        revenue_per_step = fares / durations.float()
        
        self.fleet_state.set_serving(
            vehicle_indices=matched_vehicles,
            trip_ids=matched_trips,
            target_hexes=dropoff_hexes,
            busy_until=busy_until,
            revenue_per_step=revenue_per_step
        )
        
        # Compute energy consumption
        energy_costs = self.energy_dynamics.compute_consumption(total_distances)
        self.fleet_state.socs[matched_vehicles] -= energy_costs
        self.fleet_state.socs = torch.clamp(self.fleet_state.socs, 0.0, self.max_soc)
        
        # Compute costs
        driving_cost = (total_distances * self.config.reward.driving_cost_per_km).sum()
        revenues = fares - energy_costs * self.config.reward.electricity_cost_per_kwh
        total_revenue = revenues.sum()
        
        num_failed = num_attempted - trips_served
        return trips_served, total_revenue, driving_cost, num_failed, matched_vehicles, matched_trips
    
    def process_serve_actions_with_preferences(
        self,
        serve_mask: torch.Tensor,
        current_step: int,
        serve_scores: Optional[torch.Tensor] = None,
        preference_weight: float = 0.5
    ) -> Tuple[int, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Process serve actions using TripAssigner with actor preferences.
        """
        serve_indices = serve_mask.nonzero(as_tuple=True)[0]
        num_attempted = len(serve_indices)
        
        empty_tensor = torch.tensor([], dtype=torch.long, device=self.device)
        
        if num_attempted == 0:
            return 0, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), 0, empty_tensor, empty_tensor
        
        unassigned_mask = self.trip_state.get_unassigned_mask()
        if not unassigned_mask.any():
            return 0, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), num_attempted, empty_tensor, empty_tensor
        
        trip_indices = unassigned_mask.nonzero(as_tuple=True)[0]
        
        # Use TripAssigner if available
        if self.trip_assigner is not None:
            positions = self.fleet_state.positions[serve_indices]
            trip_hexes = self.trip_state.pickup_hex[trip_indices]
            
            if serve_scores is not None:
                vehicle_prefs = serve_scores[serve_indices][:, :len(trip_indices)]
            else:
                vehicle_prefs = None
            
            result = self.trip_assigner.assign(
                vehicle_indices=serve_indices,
                vehicle_positions=positions,
                vehicle_preferences=vehicle_prefs,
                trip_indices=trip_indices,
                trip_pickup_hexes=trip_hexes,
                preference_weight=preference_weight
            )
            
            matched_vehicles = result.vehicle_indices
            matched_trips = result.target_indices
        else:
            return self.process_serve_actions(serve_mask, current_step)
        
        if len(matched_vehicles) == 0:
            return 0, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), num_attempted, empty_tensor, empty_tensor
        
        trips_served = len(matched_trips)
        
        # Get trip info
        fares = self.trip_state.fare[matched_trips]
        trip_distances = self.trip_state.distance_km[matched_trips]  # Pickup → Dropoff distance from dataset
        dropoff_hexes = self.trip_state.dropoff_hex[matched_trips]
        pickup_hexes = self.trip_state.pickup_hex[matched_trips]
        
        # Calculate pickup distances (vehicle current position → pickup point)
        vehicle_positions = self.fleet_state.positions[matched_vehicles]
        pickup_distances = self.hex_grid.distance_matrix._distances[
            vehicle_positions, pickup_hexes
        ]  # km
        
        # Calculate total distance and duration using config values
        total_distances = pickup_distances + trip_distances  # Total km traveled
        
        # Duration calculation:
        # total_distance (km) / avg_speed (km/h) = hours
        # hours * 60 = minutes
        # minutes / step_duration = steps
        avg_speed_kmh = self.config.physics.avg_speed_kmh  # e.g., 25 km/h
        step_minutes = self.config.episode.step_duration_minutes  # e.g., 5 minutes
        
        duration_hours = total_distances / avg_speed_kmh
        duration_minutes = duration_hours * 60.0
        durations = (duration_minutes / step_minutes).clamp(min=1).long()
        
        # Assign trips
        self.trip_state.assigned[matched_trips] = True
        self.trip_state.assigned_vehicle[matched_trips] = matched_vehicles
        
        # Set serving status
        busy_until = (current_step + durations).int()
        revenue_per_step = fares / durations.float()
        
        self.fleet_state.set_serving(
            vehicle_indices=matched_vehicles,
            trip_ids=matched_trips,
            target_hexes=dropoff_hexes,
            busy_until=busy_until,
            revenue_per_step=revenue_per_step
        )
        
        # Energy and costs (for total distance traveled)
        energy_costs = self.energy_dynamics.compute_consumption(total_distances)
        self.fleet_state.socs[matched_vehicles] -= energy_costs
        self.fleet_state.socs = torch.clamp(self.fleet_state.socs, 0.0, self.max_soc)
        
        driving_cost = (total_distances * self.config.reward.driving_cost_per_km).sum()
        revenues = fares - energy_costs * self.config.reward.electricity_cost_per_kwh
        total_revenue = revenues.sum()
        
        num_failed = num_attempted - trips_served
        return trips_served, total_revenue, driving_cost, num_failed, matched_vehicles, matched_trips
    
    def process_charge_actions(
        self,
        charge_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Process charging actions - VECTORIZED with distance-based station selection.
        
        Returns:
            Tuple of (total_cost, num_charge_failed, matched_vehicles, matched_stations)
        """
        charge_indices = charge_mask.nonzero(as_tuple=True)[0]
        num_attempted = len(charge_indices)
        
        empty_tensor = torch.tensor([], dtype=torch.long, device=self.device)
        
        if num_attempted == 0:
            return torch.tensor(0.0, device=self.device), 0, empty_tensor, empty_tensor
        
        if self.num_stations == 0:
            return torch.tensor(0.0, device=self.device), num_attempted, empty_tensor, empty_tensor
        
        # Use StationAssigner if available
        if self.station_assigner is not None:
            positions = self.fleet_state.positions[charge_indices]
            available_ports = self.station_state.get_available_ports()
            
            result = self.station_assigner.assign(
                vehicle_indices=charge_indices,
                vehicle_positions=positions,
                vehicle_preferences=None,
                available_ports=available_ports,
                preference_weight=0.0
            )
            
            if len(result.vehicle_indices) == 0:
                return torch.tensor(0.0, device=self.device), num_attempted, empty_tensor, empty_tensor
            
            valid_charge_indices = result.vehicle_indices
            valid_station_indices = result.target_indices
            num_success = len(valid_charge_indices)
            
            # OPTIMIZED: batch occupy ports
            self.station_state.occupy_ports_batch(valid_station_indices)
        else:
            # Fallback: simple modulo assignment
            positions = self.fleet_state.positions[charge_indices]
            station_indices = positions % self.num_stations
            
            success_mask = self.station_state.batch_occupy(station_indices)
            
            if not success_mask.any():
                return torch.tensor(0.0, device=self.device), num_attempted, empty_tensor, empty_tensor
            
            valid_charge_indices = charge_indices[success_mask]
            valid_station_indices = station_indices[success_mask]
            num_success = len(valid_charge_indices)
        
        # Set charging status
        current_socs = self.fleet_state.socs[valid_charge_indices]
        charge_power_kw = self.config.physics.charge_power_kw
        charge_power = torch.full((len(valid_charge_indices),), charge_power_kw, device=self.device)
        
        self.fleet_state.set_charging(
            vehicle_indices=valid_charge_indices,
            station_ids=valid_station_indices,
            charge_power=charge_power
        )
        
        # Compute costs
        energy_needed = self.max_soc - current_socs
        costs = energy_needed * self.config.station.price_per_kwh
        total_cost = costs.sum()
        
        num_failed = num_attempted - num_success
        return total_cost, num_failed, valid_charge_indices, valid_station_indices
    
    def process_charge_actions_with_preferences(
        self,
        charge_mask: torch.Tensor,
        charge_scores: Optional[torch.Tensor] = None,
        preference_weight: float = 0.3
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Process charge actions using StationAssigner with actor preferences.
        """
        charge_indices = charge_mask.nonzero(as_tuple=True)[0]
        num_attempted = len(charge_indices)
        
        empty_tensor = torch.tensor([], dtype=torch.long, device=self.device)
        
        if num_attempted == 0:
            return torch.tensor(0.0, device=self.device), 0, empty_tensor, empty_tensor
        
        if self.num_stations == 0 or self.station_assigner is None:
            return self.process_charge_actions(charge_mask)
        
        positions = self.fleet_state.positions[charge_indices]
        available_ports = self.station_state.get_available_ports()
        
        if charge_scores is not None:
            vehicle_prefs = charge_scores[charge_indices]
        else:
            vehicle_prefs = None
        
        result = self.station_assigner.assign(
            vehicle_indices=charge_indices,
            vehicle_positions=positions,
            vehicle_preferences=vehicle_prefs,
            available_ports=available_ports,
            preference_weight=preference_weight
        )
        
        if len(result.vehicle_indices) == 0:
            return torch.tensor(0.0, device=self.device), num_attempted, empty_tensor, empty_tensor
        
        valid_charge_indices = result.vehicle_indices
        valid_station_indices = result.target_indices
        num_success = len(valid_charge_indices)
        
        # OPTIMIZED: batch occupy ports
        self.station_state.occupy_ports_batch(valid_station_indices)
        
        current_socs = self.fleet_state.socs[valid_charge_indices]
        charge_power_kw = self.config.physics.charge_power_kw
        charge_power = torch.full((len(valid_charge_indices),), charge_power_kw, device=self.device)
        
        self.fleet_state.set_charging(
            vehicle_indices=valid_charge_indices,
            station_ids=valid_station_indices,
            charge_power=charge_power
        )
        
        energy_needed = self.max_soc - current_socs
        costs = energy_needed * self.config.station.price_per_kwh
        total_cost = costs.sum()
        
        num_failed = num_attempted - num_success
        return total_cost, num_failed, valid_charge_indices, valid_station_indices
    
    def process_reposition_actions(
        self,
        reposition_mask: torch.Tensor,
        targets: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        """Process repositioning actions - VECTORIZED."""
        reposition_indices = reposition_mask.nonzero(as_tuple=True)[0]
        
        if len(reposition_indices) == 0:
            return torch.tensor(0.0, device=self.device)
        
        current_positions = self.fleet_state.positions[reposition_indices]
        target_positions = targets[reposition_indices] if targets.dim() > 0 else targets.expand(len(reposition_indices))
        
        distances = self.hex_grid.distance_matrix.get_distances_batch(current_positions, target_positions)
        
        # REPOSITION always takes 1 step
        durations = torch.ones(len(reposition_indices), dtype=torch.long, device=self.device)
        busy_until = (current_step + durations).int()
        
        self.fleet_state.set_repositioning(
            vehicle_indices=reposition_indices,
            target_hexes=target_positions.long(),
            busy_until=busy_until
        )
        
        # Energy consumption
        energy_costs = self.energy_dynamics.compute_consumption(distances)
        self.fleet_state.socs[reposition_indices] -= energy_costs
        self.fleet_state.socs = torch.clamp(self.fleet_state.socs, 0.0, self.max_soc)
        
        total_cost = (distances * self.config.reward.driving_cost_per_km).sum()
        return total_cost
    
    def update_ongoing_actions(
        self,
        current_step: int,
    ):
        """Update vehicles with ongoing actions."""
        completed = self.fleet_state.complete_actions(current_step)
        
        charging_mask = self.fleet_state.get_charging_mask()
        if charging_mask.any():
            step_duration_hours = self.config.episode.step_duration_minutes / 60.0
            charge_power_per_vehicle = self.fleet_state.charge_power[charging_mask]
            energy_added = charge_power_per_vehicle * step_duration_hours
            
            self.fleet_state.socs[charging_mask] += energy_added
            self.fleet_state.socs = torch.clamp(self.fleet_state.socs, 0.0, self.max_soc)
            
            # Release fully charged vehicles
            fully_charged = self.fleet_state.socs[charging_mask] >= self.max_soc * 0.99
            if fully_charged.any():
                charging_indices = charging_mask.nonzero(as_tuple=True)[0]
                finished = charging_indices[fully_charged]
                
                # OPTIMIZED: batch release ports
                station_indices = self.fleet_state.charging_station[finished]
                valid_stations = station_indices[station_indices >= 0]
                if len(valid_stations) > 0:
                    self.station_state.release_ports_batch(valid_stations)
                
                self.fleet_state.set_idle(finished)

    def _match_with_gcn_and_hungarian(
        self,
        serve_indices: torch.Tensor,
        selected_trip: torch.Tensor,
        trip_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match vehicles to trips using GCN selections with Hungarian conflict resolution.
        
        Returns:
            matched_vehicles, matched_trips
        """
        device = serve_indices.device
        N_serve = len(serve_indices)
        
        if N_serve == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        
        # Get GCN's trip choices for serving vehicles
        gcn_choices = selected_trip[serve_indices]
        
        # Map to actual unassigned trip IDs
        num_available = len(trip_indices)
        
        if num_available == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        
        # CRITICAL FIX: Modulo wrap GCN choices into valid range
        # GCN outputs [0, max_trips) but only num_available trips exist
        # Use modulo to map: e.g., choice 450 with 300 trips → 450 % 300 = 150
        valid_gcn_choices = gcn_choices % num_available
        
        # All vehicles are now valid (after modulo wrap)
        valid_serve_indices = serve_indices
        
        # Map to actual trip IDs
        chosen_trips = trip_indices[valid_gcn_choices]
        
        # Check pickup distance constraints
        vehicle_positions = self.fleet_state.positions[valid_serve_indices]
        trip_pickup_hexes = self.trip_state.pickup_hex[chosen_trips]
        
        distances = self.hex_grid.distance_matrix._distances[vehicle_positions, trip_pickup_hexes]
        within_limit = distances <= self.max_pickup_distance
        
        if not within_limit.any():
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        
        # Filter by distance
        valid_serve_indices = valid_serve_indices[within_limit]
        chosen_trips = chosen_trips[within_limit]
        
        # Detect conflicts
        unique_trips, inverse_indices = chosen_trips.unique(return_inverse=True)
        
        if len(unique_trips) == len(chosen_trips):
            return valid_serve_indices, chosen_trips
        
        # OPTIMIZED: Fully vectorized conflict resolution
        # For each unique trip, find the nearest vehicle among competitors
        
        # Build distance tensor: [num_valid_vehicles, num_unique_trips]
        veh_positions = self.fleet_state.positions[valid_serve_indices]
        trip_hexes = self.trip_state.pickup_hex[unique_trips]
        distances = self.hex_grid.distance_matrix._distances[veh_positions[:, None], trip_hexes[None, :]]
        
        # Mask out non-competing vehicles (distance = inf where vehicle didn't choose this trip)
        competition_mask = inverse_indices[None, :] == torch.arange(len(unique_trips), device=device)[:, None]
        masked_distances = distances.t().clone()  # [num_unique_trips, num_valid_vehicles]
        masked_distances[~competition_mask] = float('inf')
        
        # Find winner (nearest vehicle) per trip
        winner_indices = masked_distances.argmin(dim=1)  # [num_unique_trips]
        
        # Filter out trips with no valid winner (all inf)
        valid_winners = masked_distances[torch.arange(len(unique_trips), device=device), winner_indices] < float('inf')
        
        if not valid_winners.any():
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        
        matched_vehicles = valid_serve_indices[winner_indices[valid_winners]]
        matched_trips = unique_trips[valid_winners]
        
        return matched_vehicles, matched_trips

    def _greedy_distance_matching(
        self,
        serve_indices: torch.Tensor,
        trip_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fallback greedy distance-based matching.
        """
        device = serve_indices.device
        num_vehicles = len(serve_indices)
        num_trips = len(trip_indices)
        
        if num_vehicles == 0 or num_trips == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        
        positions = self.fleet_state.positions[serve_indices]
        trip_hexes = self.trip_state.pickup_hex[trip_indices]
        
        veh_positions_expanded = positions.unsqueeze(1).expand(num_vehicles, num_trips)
        trip_hexes_expanded = trip_hexes.unsqueeze(0).expand(num_vehicles, num_trips)
        
        distances = self.hex_grid.distance_matrix._distances[
            veh_positions_expanded.reshape(-1), 
            trip_hexes_expanded.reshape(-1)
        ].reshape(num_vehicles, num_trips)
        
        valid_mask = distances <= self.max_pickup_distance
        
        if not valid_mask.any():
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        
        masked_distances = distances.clone()
        masked_distances[~valid_mask] = float('inf')
        
        nearest_trip_local = masked_distances.argmin(dim=1)
        min_distances = masked_distances[torch.arange(num_vehicles, device=device), nearest_trip_local]
        
        valid_vehicles = min_distances < float('inf')
        
        if not valid_vehicles.any():
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        
        valid_veh_local = valid_vehicles.nonzero(as_tuple=True)[0]
        matched_vehicles = serve_indices[valid_veh_local]
        matched_trip_local = nearest_trip_local[valid_veh_local]
        
        sorted_order = matched_trip_local.argsort()
        sorted_trips = matched_trip_local[sorted_order]
        sorted_vehicles = matched_vehicles[sorted_order]
        
        unique_trips_local, inverse, counts = sorted_trips.unique_consecutive(return_inverse=True, return_counts=True)
        
        first_indices = torch.zeros(len(unique_trips_local), dtype=torch.long, device=device)
        first_indices[1:] = counts[:-1].cumsum(0)
        
        matched_vehicles = sorted_vehicles[first_indices]
        matched_trips = trip_indices[unique_trips_local]
        
        return matched_vehicles, matched_trips
