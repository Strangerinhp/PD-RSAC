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
        max_pickup_distance: float = 5.0,
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
        
        self._release_charging_ports(matched_vehicles)
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
        
        self._release_charging_ports(matched_vehicles)
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
        
        # Travel to assigned station before charging
        station_hex_positions = self.station_state.hex_ids[valid_station_indices].long()
        vehicle_positions = self.fleet_state.positions[valid_charge_indices].long()
        distances = self.hex_grid.distance_matrix.get_distances_batch(
            vehicle_positions,
            station_hex_positions
        )
        travel_energy_costs = self.energy_dynamics.compute_consumption(distances)
        self.fleet_state.socs[valid_charge_indices] -= travel_energy_costs
        self.fleet_state.socs = torch.clamp(self.fleet_state.socs, 0.0, self.max_soc)

        # Set charging status
        charge_power_kw = self.config.physics.charge_power_kw
        charge_power = torch.full((len(valid_charge_indices),), charge_power_kw, device=self.device)
        
        self._release_charging_ports(valid_charge_indices)
        self.fleet_state.set_charging(
            vehicle_indices=valid_charge_indices,
            station_ids=valid_station_indices,
            charge_power=charge_power
        )
        # Vehicles are now physically at the station hex.
        self.fleet_state.positions[valid_charge_indices] = station_hex_positions
        
        # Charging action cost now only includes travel-to-station driving cost.
        # Electricity is charged incrementally in update_ongoing_actions().
        total_cost = (distances * self.config.reward.driving_cost_per_km).sum()
        
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
        
        # Travel to assigned station before charging
        station_hex_positions = self.station_state.hex_ids[valid_station_indices].long()
        vehicle_positions = self.fleet_state.positions[valid_charge_indices].long()
        distances = self.hex_grid.distance_matrix.get_distances_batch(
            vehicle_positions,
            station_hex_positions
        )
        travel_energy_costs = self.energy_dynamics.compute_consumption(distances)
        self.fleet_state.socs[valid_charge_indices] -= travel_energy_costs
        self.fleet_state.socs = torch.clamp(self.fleet_state.socs, 0.0, self.max_soc)

        charge_power_kw = self.config.physics.charge_power_kw
        charge_power = torch.full((len(valid_charge_indices),), charge_power_kw, device=self.device)
        
        self._release_charging_ports(valid_charge_indices)
        self.fleet_state.set_charging(
            vehicle_indices=valid_charge_indices,
            station_ids=valid_station_indices,
            charge_power=charge_power
        )
        # Vehicles are now physically at the station hex.
        self.fleet_state.positions[valid_charge_indices] = station_hex_positions
        
        # Charging action cost now only includes travel-to-station driving cost.
        # Electricity is charged incrementally in update_ongoing_actions().
        total_cost = (distances * self.config.reward.driving_cost_per_km).sum()
        
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
        
        # Release charging ports if interrupted
        self._release_charging_ports(reposition_indices)
        
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
    ) -> torch.Tensor:
        """Update vehicles with ongoing actions."""
        completed = self.fleet_state.complete_actions(current_step)
        _ = completed

        step_charge_cost = torch.tensor(0.0, device=self.device)
        
        charging_mask = self.fleet_state.get_charging_mask()
        if charging_mask.any():
            step_duration_hours = self.config.episode.step_duration_minutes / 60.0
            charge_power_per_vehicle = self.fleet_state.charge_power[charging_mask]
            requested_energy_added = charge_power_per_vehicle * step_duration_hours
            prev_socs = self.fleet_state.socs[charging_mask]
            next_socs = torch.clamp(prev_socs + requested_energy_added, 0.0, self.max_soc)
            actual_energy_added = torch.clamp(next_socs - prev_socs, min=0.0)
            
            self.fleet_state.socs[charging_mask] = next_socs
            step_charge_cost = (actual_energy_added * self.config.station.price_per_kwh).sum()
            
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
        
        return step_charge_cost

    def _release_charging_ports(self, vehicle_indices: torch.Tensor):
        if getattr(self, 'station_state', None) is None:
            return
        was_charging = self.fleet_state.status[vehicle_indices] == 3  # VehicleStatus.CHARGING
        if was_charging.any():
            interrupting_indices = vehicle_indices[was_charging]
            station_indices = self.fleet_state.charging_station[interrupting_indices]
            valid_stations = station_indices[station_indices >= 0]
            if len(valid_stations) > 0:
                self.station_state.release_ports_batch(valid_stations)
            self.fleet_state.charging_station[interrupting_indices] = -1

    # Maximum re-matching rounds for iterative conflict resolution.
    # Keeps computation bounded while recovering most serve failures.
    MAX_REMATCH_ROUNDS = 5

    def _match_with_gcn_and_hungarian(
        self,
        serve_indices: torch.Tensor,
        selected_trip: torch.Tensor,
        trip_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match vehicles to trips using GCN selections with iterative conflict resolution.

        The upstream per-vehicle trip mask (episode_collector._build_per_vehicle_trip_mask)
        already guarantees that each vehicle's GCN choice is within max_pickup_distance.
        Steps:
          1. Modulo wrap for index safety (max_trips > num_available edge-case).
          2. Greedy distance fallback for vehicles whose mask fell back to all-trips.
          3. Conflict resolution: nearest vehicle wins each trip.
          4. **Iterative re-matching**: losing vehicles are re-matched to remaining
             unassigned trips via greedy distance matching (up to MAX_REMATCH_ROUNDS).
        """
        device = serve_indices.device
        N_serve = len(serve_indices)

        if N_serve == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

        gcn_choices  = selected_trip[serve_indices]  # [N_serve] indices into [0, max_trips)
        num_available = len(trip_indices)

        if num_available == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

        # Modulo wrap: safety guard in case mask fallback produced an out-of-range index
        valid_gcn_choices = gcn_choices % num_available
        chosen_trips      = trip_indices[valid_gcn_choices]  # map to actual trip IDs

        # Check for the "mask fallback" case: vehicles with no nearby trips were given
        # all available trips in the mask. If their chosen trip is still out of range,
        # fall back to greedy nearest-trip matching for those vehicles.
        vehicle_positions  = self.fleet_state.positions[serve_indices]
        trip_pickup_hexes  = self.trip_state.pickup_hex[chosen_trips]
        distances          = self.hex_grid.distance_matrix._distances[vehicle_positions, trip_pickup_hexes]
        within_limit       = distances <= self.max_pickup_distance

        if not within_limit.all():
            # Only a few vehicles should have out-of-range choices (mask-fallback vehicles).
            # Handle them with greedy matching; keep in-range choices as-is.
            if not within_limit.any():
                return self._greedy_distance_matching(serve_indices, trip_indices)
            # Mixed: keep within-range vehicles, greedy-match the rest
            ok_vehicles    = serve_indices[within_limit]
            ok_trips       = chosen_trips[within_limit]
            far_vehicles   = serve_indices[~within_limit]
            g_veh, g_trip  = self._greedy_distance_matching(far_vehicles, trip_indices)
            if len(g_veh) > 0:
                serve_indices = torch.cat([ok_vehicles, g_veh])
                chosen_trips  = torch.cat([ok_trips,    g_trip])
            else:
                serve_indices = ok_vehicles
                chosen_trips  = ok_trips

        # ---- Conflict resolution with iterative re-matching ----
        # Round 1: resolve GCN conflicts (nearest vehicle wins per trip).
        # Subsequent rounds: losers get re-matched via greedy distance to
        # remaining trips, then conflicts in the new matches are resolved again.

        all_matched_vehicles = []
        all_matched_trips = []

        # Track which trips from the full pool are still available
        remaining_trip_set = set(trip_indices.tolist())

        current_vehicles = serve_indices
        current_trips = chosen_trips

        for _round in range(self.MAX_REMATCH_ROUNDS):
            if len(current_vehicles) == 0 or len(remaining_trip_set) == 0:
                break

            unique_trips, inverse_indices = current_trips.unique(return_inverse=True)

            if len(unique_trips) == len(current_trips):
                # No conflicts — all vehicles matched successfully
                all_matched_vehicles.append(current_vehicles)
                all_matched_trips.append(current_trips)
                # Remove these trips from the available pool
                remaining_trip_set -= set(current_trips.tolist())
                break

            # Resolve conflicts: nearest vehicle wins each contested trip
            veh_positions = self.fleet_state.positions[current_vehicles]
            trip_hexes = self.trip_state.pickup_hex[unique_trips]
            dist_matrix = self.hex_grid.distance_matrix._distances[
                veh_positions[:, None], trip_hexes[None, :]
            ]

            competition_mask = inverse_indices[None, :] == torch.arange(
                len(unique_trips), device=device
            )[:, None]
            masked_distances = dist_matrix.t().clone()  # [unique_trips, vehicles]
            masked_distances[~competition_mask] = float('inf')

            winner_indices = masked_distances.argmin(dim=1)  # [unique_trips]
            valid_winners = masked_distances[
                torch.arange(len(unique_trips), device=device), winner_indices
            ] < float('inf')

            if not valid_winners.any():
                break

            # Collect winners
            round_vehicles = current_vehicles[winner_indices[valid_winners]]
            round_trips = unique_trips[valid_winners]
            all_matched_vehicles.append(round_vehicles)
            all_matched_trips.append(round_trips)

            # Remove won trips from available pool
            remaining_trip_set -= set(round_trips.tolist())

            # Identify losers: vehicles that didn't win
            winner_set = set(winner_indices[valid_winners].tolist())
            loser_mask = torch.ones(len(current_vehicles), dtype=torch.bool, device=device)
            for w_idx in winner_set:
                loser_mask[w_idx] = False
            loser_vehicles = current_vehicles[loser_mask]

            if len(loser_vehicles) == 0 or len(remaining_trip_set) == 0:
                break

            # Re-match losers to remaining trips via greedy distance
            remaining_trips = torch.tensor(
                sorted(remaining_trip_set), dtype=torch.long, device=device
            )
            g_veh, g_trip = self._greedy_distance_matching(loser_vehicles, remaining_trips)

            if len(g_veh) == 0:
                break

            # Next round: resolve any new conflicts from greedy matching
            current_vehicles = g_veh
            current_trips = g_trip

        # Combine all rounds
        if len(all_matched_vehicles) == 0:
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        return torch.cat(all_matched_vehicles), torch.cat(all_matched_trips)

    def _greedy_distance_matching(
        self,
        serve_indices: torch.Tensor,
        trip_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy distance-based matching with iterative re-matching.

        Losers (vehicles whose nearest trip was claimed by another vehicle)
        are re-matched to remaining unclaimed trips, up to MAX_REMATCH_ROUNDS.
        """
        device = serve_indices.device

        if len(serve_indices) == 0 or len(trip_indices) == 0:
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        all_matched_vehicles = []
        all_matched_trips = []

        remaining_vehicles = serve_indices
        remaining_trips = trip_indices

        for _round in range(self.MAX_REMATCH_ROUNDS):
            if len(remaining_vehicles) == 0 or len(remaining_trips) == 0:
                break

            winners, won_trips, losers = self._greedy_distance_single_pass(
                remaining_vehicles, remaining_trips
            )

            if len(winners) == 0:
                break

            all_matched_vehicles.append(winners)
            all_matched_trips.append(won_trips)

            # Remove won trips from pool for next round
            won_set = set(won_trips.tolist())
            keep_mask = torch.tensor(
                [t.item() not in won_set for t in remaining_trips],
                dtype=torch.bool, device=device
            )
            remaining_trips = remaining_trips[keep_mask]
            remaining_vehicles = losers

        if len(all_matched_vehicles) == 0:
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        return torch.cat(all_matched_vehicles), torch.cat(all_matched_trips)

    def _greedy_distance_single_pass(
        self,
        serve_indices: torch.Tensor,
        trip_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-pass greedy nearest-trip matching (no re-matching).

        Returns:
            (matched_vehicles, matched_trips, unmatched_vehicles)
        """
        device = serve_indices.device
        num_vehicles = len(serve_indices)
        num_trips = len(trip_indices)

        empty = torch.empty(0, dtype=torch.long, device=device)

        if num_vehicles == 0 or num_trips == 0:
            return empty, empty, serve_indices

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
            return empty, empty, serve_indices

        masked_distances = distances.clone()
        masked_distances[~valid_mask] = float('inf')

        nearest_trip_local = masked_distances.argmin(dim=1)
        min_distances = masked_distances[torch.arange(num_vehicles, device=device), nearest_trip_local]

        valid_vehicles = min_distances < float('inf')

        if not valid_vehicles.any():
            return empty, empty, serve_indices

        valid_veh_local = valid_vehicles.nonzero(as_tuple=True)[0]
        matched_vehicles = serve_indices[valid_veh_local]
        matched_trip_local = nearest_trip_local[valid_veh_local]

        # For each trip picked by multiple vehicles, keep the nearest one
        sorted_order = matched_trip_local.argsort()
        sorted_trips = matched_trip_local[sorted_order]
        sorted_vehicles = matched_vehicles[sorted_order]
        sorted_distances = min_distances[valid_veh_local][sorted_order]

        unique_trips_local, inverse, counts = sorted_trips.unique_consecutive(
            return_inverse=True, return_counts=True
        )

        # For each unique trip, pick the vehicle with minimum distance
        first_indices = torch.zeros(len(unique_trips_local), dtype=torch.long, device=device)
        first_indices[1:] = counts[:-1].cumsum(0)

        # Within each group, find the argmin distance vehicle
        best_indices = []
        for i, (start, count) in enumerate(zip(first_indices, counts)):
            group_distances = sorted_distances[start:start + count]
            best_in_group = start + group_distances.argmin()
            best_indices.append(best_in_group.item())

        best_indices_t = torch.tensor(best_indices, dtype=torch.long, device=device)
        winners = sorted_vehicles[best_indices_t]
        won_trips = trip_indices[unique_trips_local]

        # Identify losers: vehicles that were valid but didn't win
        winner_set = set(winners.tolist())
        # Include vehicles that had no valid trip at all
        no_valid = serve_indices[~valid_vehicles] if (~valid_vehicles).any() else empty
        lost_valid = matched_vehicles[
            torch.tensor([v.item() not in winner_set for v in matched_vehicles],
                        dtype=torch.bool, device=device)
        ] if len(matched_vehicles) > 0 else empty

        losers = torch.cat([lost_valid, no_valid]) if (len(lost_valid) + len(no_valid)) > 0 else empty

        return winners, won_trips, losers
