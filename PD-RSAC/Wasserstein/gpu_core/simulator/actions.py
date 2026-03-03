"""Action execution for vehicles."""

import torch
from enum import IntEnum
from typing import Tuple, Optional
from ..state import TensorFleetState, TensorTripState, TensorStationState, VehicleStatus


class ActionType(IntEnum):
    WAIT = 0
    SERVE = 1
    REPOSITION = 2
    CHARGE = 3


class ActionExecutor:
    """Executes actions for vehicles in a vectorized manner."""
    
    def __init__(
        self,
        distance_matrix: torch.Tensor,
        avg_speed_kmh: float = 30.0,
        step_duration_minutes: float = 5.0,
        energy_per_km: float = 0.2,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.distance_matrix = distance_matrix.to(self.device)
        self.avg_speed_kmh = avg_speed_kmh
        self.step_duration_minutes = step_duration_minutes
        self.energy_per_km = energy_per_km
        
        # Pre-compute travel time matrix (in steps)
        distance_km = self.distance_matrix
        time_hours = distance_km / avg_speed_kmh
        time_minutes = time_hours * 60
        self.travel_time_steps = torch.ceil(time_minutes / step_duration_minutes).to(torch.int32)
    
    def execute_serve(
        self,
        fleet: TensorFleetState,
        trips: TensorTripState,
        vehicle_indices: torch.Tensor,
        trip_indices: torch.Tensor,
        current_step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute serve actions for vehicles.
        
        Returns:
            (served_mask, total_steps) - mask of successfully served, steps until completion
        """
        if len(vehicle_indices) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device), torch.zeros(0, dtype=torch.int32, device=self.device)
        
        # Get trip info
        pickup_hexes = trips.pickup_hex[trip_indices]
        dropoff_hexes = trips.dropoff_hex[trip_indices]
        fares = trips.fare[trip_indices]
        
        # Get vehicle positions
        vehicle_positions = fleet.positions[vehicle_indices]
        
        # Calculate travel times
        pickup_time = self.travel_time_steps[vehicle_positions, pickup_hexes]
        serve_time = self.travel_time_steps[pickup_hexes, dropoff_hexes]
        total_time = pickup_time + serve_time
        
        # Calculate energy requirements
        pickup_distance = self.distance_matrix[vehicle_positions, pickup_hexes]
        serve_distance = self.distance_matrix[pickup_hexes, dropoff_hexes]
        total_distance = pickup_distance + serve_distance
        energy_needed = total_distance * self.energy_per_km
        
        # Check if vehicles have enough energy
        can_serve = fleet.socs[vehicle_indices] >= energy_needed
        
        # Execute for vehicles that can serve
        if can_serve.any():
            serving_vehicles = vehicle_indices[can_serve]
            serving_trips = trip_indices[can_serve]
            
            busy_until = current_step + total_time[can_serve]
            revenue_per_step = fares[can_serve] / total_time[can_serve].float()
            
            fleet.set_serving(
                serving_vehicles,
                trips.trip_ids[serving_trips],
                dropoff_hexes[can_serve],
                busy_until,
                revenue_per_step,
            )
            
            # Mark trips as assigned - OPTIMIZED: vectorized call
            trips.assign_trips_batch(serving_trips, serving_vehicles)
        
        return can_serve, total_time
    
    def execute_reposition(
        self,
        fleet: TensorFleetState,
        vehicle_indices: torch.Tensor,
        target_hexes: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        """
        Execute reposition actions.
        
        Returns:
            travel_times - steps to reach target
        """
        if len(vehicle_indices) == 0:
            return torch.zeros(0, dtype=torch.int32, device=self.device)
        
        vehicle_positions = fleet.positions[vehicle_indices]
        travel_times = self.travel_time_steps[vehicle_positions, target_hexes]
        
        # Calculate energy needed
        distances = self.distance_matrix[vehicle_positions, target_hexes]
        energy_needed = distances * self.energy_per_km
        
        # Check if can reposition
        can_reposition = fleet.socs[vehicle_indices] >= energy_needed
        
        if can_reposition.any():
            repositioning = vehicle_indices[can_reposition]
            busy_until = current_step + travel_times[can_reposition]
            
            fleet.set_repositioning(
                repositioning,
                target_hexes[can_reposition],
                busy_until,
            )
        
        return travel_times
    
    def execute_charge(
        self,
        fleet: TensorFleetState,
        stations: TensorStationState,
        vehicle_indices: torch.Tensor,
        station_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute charge actions.
        
        Returns:
            success_mask - which vehicles successfully started charging
        """
        if len(vehicle_indices) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        
        # Try to occupy stations
        success = stations.batch_occupy(station_indices)
        
        if success.any():
            charging_vehicles = vehicle_indices[success]
            charging_stations = station_indices[success]
            
            # Get charge power
            charge_power = stations.power_capacity[charging_stations]
            
            fleet.set_charging(
                charging_vehicles,
                charging_stations,
                charge_power,
            )
        
        return success
    
    def execute_wait(
        self,
        fleet: TensorFleetState,
        vehicle_indices: torch.Tensor,
    ) -> None:
        """Execute wait actions - vehicles stay idle."""
        if len(vehicle_indices) == 0:
            return
        
        # Ensure vehicles are in IDLE state
        fleet.set_idle(vehicle_indices)
    
    def execute_batch(
        self,
        fleet: TensorFleetState,
        trips: TensorTripState,
        stations: TensorStationState,
        action_types: torch.Tensor,
        action_targets: torch.Tensor,
        current_step: int,
    ) -> dict:
        """
        Execute batch of actions for all vehicles.
        
        Args:
            action_types: [num_vehicles] ActionType values
            action_targets: [num_vehicles] target index (trip/hex/station)
            
        Returns:
            Dict with execution results
        """
        results = {
            "served_count": 0,
            "repositioned_count": 0,
            "charged_count": 0,
            "waited_count": 0,
            "failed_serve": 0,
            "failed_charge": 0,
        }
        
        # SERVE actions
        serve_mask = action_types == ActionType.SERVE
        if serve_mask.any():
            vehicle_indices = torch.where(serve_mask)[0]
            trip_indices = action_targets[serve_mask]
            
            success, _ = self.execute_serve(
                fleet, trips, vehicle_indices, trip_indices, current_step
            )
            results["served_count"] = success.sum().item()
            results["failed_serve"] = (~success).sum().item()
        
        # REPOSITION actions
        reposition_mask = action_types == ActionType.REPOSITION
        if reposition_mask.any():
            vehicle_indices = torch.where(reposition_mask)[0]
            target_hexes = action_targets[reposition_mask]
            
            self.execute_reposition(fleet, vehicle_indices, target_hexes, current_step)
            results["repositioned_count"] = reposition_mask.sum().item()
        
        # CHARGE actions
        charge_mask = action_types == ActionType.CHARGE
        if charge_mask.any():
            vehicle_indices = torch.where(charge_mask)[0]
            station_indices = action_targets[charge_mask]
            
            success = self.execute_charge(fleet, stations, vehicle_indices, station_indices)
            results["charged_count"] = success.sum().item()
            results["failed_charge"] = (~success).sum().item()
        
        # WAIT actions
        wait_mask = action_types == ActionType.WAIT
        if wait_mask.any():
            vehicle_indices = torch.where(wait_mask)[0]
            self.execute_wait(fleet, vehicle_indices)
            results["waited_count"] = wait_mask.sum().item()
        
        return results
