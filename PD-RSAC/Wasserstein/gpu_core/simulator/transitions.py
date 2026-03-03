"""State transition engine."""

import torch
from typing import Tuple, Optional
from ..state import TensorFleetState, TensorTripState, TensorStationState, VehicleStatus


class TransitionEngine:
    """Handles state transitions between simulation steps."""
    
    def __init__(
        self,
        distance_matrix: torch.Tensor,
        avg_speed_kmh: float = 30.0,
        step_duration_minutes: float = 5.0,
        energy_per_km: float = 0.2,
        charge_power_kw: float = 50.0,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.distance_matrix = distance_matrix.to(self.device)
        self.avg_speed_kmh = avg_speed_kmh
        self.step_duration_minutes = step_duration_minutes
        self.step_duration_hours = step_duration_minutes / 60.0
        self.energy_per_km = energy_per_km
        self.charge_power_kw = charge_power_kw
        
        # Distance per step (km)
        self.km_per_step = avg_speed_kmh * self.step_duration_hours
    
    def step(
        self,
        fleet: TensorFleetState,
        trips: TensorTripState,
        stations: TensorStationState,
        current_step: int,
    ) -> dict:
        """
        Advance simulation by one step.
        
        Returns:
            Dict with step statistics
        """
        stats = {
            "completed_trips": 0,
            "completed_repositions": 0,
            "energy_consumed": 0.0,
            "energy_charged": 0.0,
            "revenue_earned": 0.0,
            "dropped_trips": 0,
        }
        
        # 1. Complete actions for vehicles that have finished
        completed = fleet.complete_actions(current_step)
        
        # Count completions by type (from previous status)
        stats["completed_trips"] = completed.sum().item()
        
        # 2. Update energy for moving vehicles
        stats["energy_consumed"] = self._update_moving_energy(fleet)
        
        # 3. Update charging vehicles
        stats["energy_charged"] = self._update_charging(fleet, stations)
        
        # 4. Collect revenue from serving vehicles
        stats["revenue_earned"] = self._collect_revenue(fleet)
        
        # 5. Update trip wait times and drop expired
        trips.increment_wait()
        
        return stats
    
    def _update_moving_energy(self, fleet: TensorFleetState) -> float:
        """Update energy for moving vehicles (serving/repositioning)."""
        moving = (fleet.status == VehicleStatus.SERVING) | (fleet.status == VehicleStatus.REPOSITIONING)
        
        if not moving.any():
            return 0.0
        
        # Energy consumed per step while moving
        energy_per_step = self.km_per_step * self.energy_per_km
        
        # Consume energy
        old_socs = fleet.socs[moving].clone()
        fleet.socs[moving] = torch.clamp(fleet.socs[moving] - energy_per_step, min=0.0)
        
        energy_consumed = (old_socs - fleet.socs[moving]).sum().item()
        return energy_consumed
    
    def _update_charging(
        self,
        fleet: TensorFleetState,
        stations: TensorStationState,
    ) -> float:
        """Update charging vehicles."""
        charging = fleet.status == VehicleStatus.CHARGING
        
        if not charging.any():
            return 0.0
        
        # Get charge power for each vehicle
        charge_power = fleet.charge_power[charging]
        
        # Energy added per step (kWh)
        energy_per_step = charge_power * self.step_duration_hours
        
        # Update SoC
        old_socs = fleet.socs[charging].clone()
        fleet.socs[charging] = torch.clamp(
            fleet.socs[charging] + energy_per_step,
            max=fleet.max_soc
        )
        
        energy_charged = (fleet.socs[charging] - old_socs).sum().item()
        
        # Check if fully charged
        fully_charged = fleet.socs[charging] >= fleet.max_soc * 0.99
        if fully_charged.any():
            # Release stations and set to idle
            charging_indices = torch.where(charging)[0]
            finished = charging_indices[fully_charged]
            
            for idx in finished:
                station_idx = fleet.charging_station[idx].item()
                if station_idx >= 0:
                    stations.release_port(station_idx)
            
            fleet.set_idle(finished)
        
        return energy_charged
    
    def _collect_revenue(self, fleet: TensorFleetState) -> float:
        """Collect revenue from serving vehicles."""
        serving = fleet.status == VehicleStatus.SERVING
        if not serving.any():
            return 0.0
        
        return fleet.ongoing_revenue[serving].sum().item()
    
    def complete_charging(
        self,
        fleet: TensorFleetState,
        stations: TensorStationState,
        vehicle_indices: torch.Tensor,
    ) -> None:
        """Force complete charging for specified vehicles."""
        for idx in vehicle_indices:
            station_idx = fleet.charging_station[idx].item()
            if station_idx >= 0:
                stations.release_port(station_idx)
        
        fleet.set_idle(vehicle_indices)
    
    def get_travel_time(
        self,
        from_hexes: torch.Tensor,
        to_hexes: torch.Tensor,
    ) -> torch.Tensor:
        """Get travel time in steps between hex pairs."""
        distances = self.distance_matrix[from_hexes, to_hexes]
        time_hours = distances / self.avg_speed_kmh
        time_steps = torch.ceil(time_hours / self.step_duration_hours).to(torch.int32)
        return time_steps
    
    def get_energy_cost(
        self,
        from_hexes: torch.Tensor,
        to_hexes: torch.Tensor,
    ) -> torch.Tensor:
        """Get energy cost (kWh) for travel between hex pairs."""
        distances = self.distance_matrix[from_hexes, to_hexes]
        return distances * self.energy_per_km
    
    def can_reach(
        self,
        fleet: TensorFleetState,
        vehicle_indices: torch.Tensor,
        target_hexes: torch.Tensor,
    ) -> torch.Tensor:
        """Check if vehicles can reach target hexes with current energy."""
        from_hexes = fleet.positions[vehicle_indices]
        energy_needed = self.get_energy_cost(from_hexes, target_hexes)
        return fleet.socs[vehicle_indices] >= energy_needed
