#!/usr/bin/env python3
"""
Heuristic Matching Baseline for EV Fleet Management

This script runs a heuristic matching algorithm (SOC-aware + profit-aware) to serve
as a baseline for comparing with the actor-critic model.

Heuristic Algorithm:
1. SOC-aware charging: Charge if SOC < critical_threshold (default: 20%)
2. Profit-aware serving: Serve trips with highest profit (fare - costs)
3. Idle: If no profitable actions available

Usage:
    python3 heuristic_matching.py \
        --config gpu_core/scripts/config.yaml \
        --real-data data/nyc_full/trips_processed.parquet \
        --start-date 2009-01-15 \
        --end-date 2009-01-15 \
        --trip-sample 0.2 \
        --num-vehicles 1000 \
        --num-hexes 1000 \
        --critical-soc-threshold 20.0 \
        --output results/heuristic_baseline.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import ConfigLoader, Config
from gpu_core.spatial.grid import HexGrid
from gpu_core.simulator.environment import GPUEnvironment
from gpu_core.simulator.environment import GPUEnvironmentV2
from gpu_core.data.real_trip_loader import RealTripLoader
from gpu_core.spatial.assignment import UltraFastGreedyAssignment, AssignmentResult
from dataclasses import dataclass, field


@dataclass
class SimpleEnvInfo:
    """Simple environment info for heuristic matching."""
    trips_served: int = 0
    trips_loaded: int = 0
    trips_dropped: int = 0
    revenue: float = 0.0
    energy_cost: float = 0.0
    driving_cost: float = 0.0
    avg_soc: float = 0.0


class SimpleHeuristicEnvironment:
    """
    Simple environment for heuristic matching baseline.
    
    Basic functionality:
    - Vehicle state (position, SOC, busy status)
    - Trip management (load trips from RealTripLoader)
    - Simple matching (no conflicts - 1 vehicle to 1 trip)
    - Revenue/cost tracking
    """
    
    def __init__(
        self,
        num_vehicles: int,
        hex_grid: HexGrid,
        trip_loader: RealTripLoader,
        config: Config,
        device: str = "cuda",
        episode_start_idx: Optional[int] = None,
        max_pickup_distance: float = 300.0
    ):
        self.num_vehicles = num_vehicles
        self.hex_grid = hex_grid
        self.trip_loader = trip_loader
        self.config = config
        self.device = torch.device(device)
        
        # Episode configuration
        self.episode_steps = config.episode.steps_per_episode
        self.step_duration_minutes = config.episode.step_duration_minutes
        
        # Cost parameters
        self.driving_cost_per_km = config.reward.driving_cost_per_km
        self.electricity_cost_per_kwh = config.reward.electricity_cost_per_kwh
        self.energy_per_km = config.physics.energy_per_km
        self.max_soc = config.physics.max_soc
        self.initial_soc = config.vehicle.initial_soc
        self.max_wait_steps = config.reward.max_wait_steps
        self.max_pickup_distance = max_pickup_distance
        
        # Episode start index
        if episode_start_idx is None:
            valid_starts = trip_loader.get_episode_start_indices(
                episode_duration_hours=config.episode.duration_hours
            )
            if len(valid_starts) > 0:
                self.episode_start_idx = valid_starts[0].item()
            else:
                self.episode_start_idx = 0
        else:
            self.episode_start_idx = episode_start_idx
        
        # Vehicle state [num_vehicles]
        self.vehicle_positions: Optional[torch.Tensor] = None  # Hex positions
        self.vehicle_socs: Optional[torch.Tensor] = None  # SOC in kWh
        self.vehicle_busy_until: Optional[torch.Tensor] = None  # Step when vehicle becomes available
        self.vehicle_target_hex: Optional[torch.Tensor] = None  # Destination hex (for serving)
        
        # Unassigned trips (list of dicts with trip info)
        self.unassigned_trips: list = []  # List of {trip_id, pickup_hex, dropoff_hex, fare, distance, wait_steps}
        self.next_trip_id = 0
        
        # Tracking
        self.current_step = 0
        self.total_trips_loaded = 0
        self.total_trips_served = 0
        self.total_trips_dropped = 0
        self.total_revenue = 0.0
        self.total_driving_cost = 0.0
        self.total_charging_cost = 0.0
        
        # Episode info
        self.episode_info = SimpleEnvInfo()
    
    def reset(self) -> torch.Tensor:
        """Reset environment to initial state."""
        # Initialize vehicles
        num_hexes = len(self.hex_grid._hex_ids) if hasattr(self.hex_grid, '_hex_ids') else self.config.environment.num_hexes
        
        # Random initial positions
        self.vehicle_positions = torch.randint(0, num_hexes, (self.num_vehicles,), device=self.device)
        self.vehicle_socs = torch.full((self.num_vehicles,), self.initial_soc, device=self.device)
        self.vehicle_busy_until = torch.zeros(self.num_vehicles, dtype=torch.long, device=self.device)
        self.vehicle_target_hex = torch.zeros(self.num_vehicles, dtype=torch.long, device=self.device)
        
        # Reset trip state
        self.unassigned_trips = []
        self.next_trip_id = 0
        self.current_step = 0
        
        # Reset tracking
        self.total_trips_loaded = 0
        self.total_trips_served = 0
        self.total_trips_dropped = 0
        self.total_revenue = 0.0
        self.total_driving_cost = 0.0
        self.total_charging_cost = 0.0
        
        self.episode_info = SimpleEnvInfo()
        
        # Return dummy state (not used in heuristic)
        return torch.zeros(self.num_vehicles, device=self.device)
    
    def _load_step_trips(self):
        """Load new trips for current step from RealTripLoader."""
        pickup_hexes, dropoff_hexes, fares, distances = self.trip_loader.get_trips_for_episode_step(
            episode_start_idx=self.episode_start_idx,
            step=self.current_step,
            step_duration_minutes=self.step_duration_minutes,
            episode_duration_hours=self.config.episode.duration_hours,
        )
        
        num_new_trips = len(pickup_hexes)
        if num_new_trips == 0:
            return
        
        # Get number of hexes
        num_hexes = len(self.hex_grid._hex_ids) if hasattr(self.hex_grid, '_hex_ids') else self.config.environment.num_hexes
        
        # Add to unassigned trips
        for i in range(num_new_trips):
            pickup_hex = pickup_hexes[i].item()
            dropoff_hex = dropoff_hexes[i].item()
            
            # Ensure hex indices are valid
            pickup_hex = pickup_hex % num_hexes
            dropoff_hex = dropoff_hex % num_hexes
            
            # Avoid same pickup/dropoff
            if pickup_hex == dropoff_hex:
                dropoff_hex = (dropoff_hex + 1) % num_hexes
            
            self.unassigned_trips.append({
                'trip_id': self.next_trip_id,
                'pickup_hex': pickup_hex,
                'dropoff_hex': dropoff_hex,
                'fare': fares[i].item(),
                'distance': distances[i].item(),
                'wait_steps': 0
            })
            self.next_trip_id += 1
        
        self.total_trips_loaded += num_new_trips
    
    def _drop_expired_trips(self):
        """Drop trips that waited too long."""
        dropped_count = 0
        valid_trips = []
        
        for trip in self.unassigned_trips:
            trip['wait_steps'] += 1
            if trip['wait_steps'] <= self.max_wait_steps:
                valid_trips.append(trip)
            else:
                dropped_count += 1
        
        self.unassigned_trips = valid_trips
        self.total_trips_dropped += dropped_count
        return dropped_count
    
    def get_debug_stats(self) -> Dict:
        """Get debug statistics for current state."""
        available_vehicles = self.get_available_vehicles()
        busy_vehicles = self.num_vehicles - len(available_vehicles)
        avg_busy_duration = (self.vehicle_busy_until - self.current_step).clamp(min=0).float().mean().item()
        
        return {
            'available_vehicles': len(available_vehicles),
            'busy_vehicles': busy_vehicles,
            'unassigned_trips': len(self.unassigned_trips),
            'avg_busy_duration': avg_busy_duration,
            'avg_soc': self.vehicle_socs.mean().item(),
            'min_soc': self.vehicle_socs.min().item(),
            'avg_wait_steps': sum(t['wait_steps'] for t in self.unassigned_trips) / max(len(self.unassigned_trips), 1)
        }
    
    def _update_vehicle_states(self):
        """
        Update vehicle states: complete serving trips, advance time.
        
        IMPORTANT: This must be called at the START of each step to:
        1. Free vehicles that have finished their trips (busy_until <= current_step)
        2. Update vehicle positions to dropoff locations
        """
        # Vehicles become available when busy_until <= current_step
        # Note: We check <= (not ==) to handle any edge cases
        available_mask = self.vehicle_busy_until <= self.current_step
        
        # Update positions: vehicles that JUST finished serving (busy_until == current_step)
        # move to dropoff location
        finished_mask = (self.vehicle_busy_until == self.current_step) & (self.vehicle_target_hex > 0)
        self.vehicle_positions[finished_mask] = self.vehicle_target_hex[finished_mask]
        self.vehicle_target_hex[finished_mask] = 0
        
        # Debug: verify no vehicles are stuck in busy state incorrectly
        # (This should never happen, but good to check)
        assert (self.vehicle_busy_until >= 0).all(), "Invalid busy_until values"
    
    def get_available_vehicles(self) -> torch.Tensor:
        """Get indices of available vehicles."""
        available_mask = self.vehicle_busy_until <= self.current_step
        return available_mask.nonzero(as_tuple=True)[0]
    
    def get_unassigned_trips(self) -> list:
        """Get list of unassigned trips."""
        return self.unassigned_trips
    
    def step(
        self,
        serve_assignments: Dict[int, int],  # {vehicle_idx: trip_idx_in_unassigned_list}
        charge_vehicles: Optional[torch.Tensor] = None  # Vehicle indices that charge
    ) -> Tuple[torch.Tensor, SimpleEnvInfo]:
        """
        Execute one step.
        
        IMPORTANT: This method ensures vehicles busy in previous steps are protected:
        1. _update_vehicle_states() is called FIRST to free vehicles that finished
        2. Only available vehicles (busy_until <= current_step) can be assigned
        3. Vehicles assigned to serve get new busy_until = current_step + duration_steps
        4. Vehicles busy will NOT be available in next steps until busy_until <= current_step
        
        Args:
            serve_assignments: Dict mapping vehicle_idx -> trip_idx (in unassigned_trips list)
            charge_vehicles: Optional tensor of vehicle indices to charge
        
        Returns:
            (dummy_state, info)
        """
        # STEP 1: Update vehicle states FIRST (free vehicles that finished serving)
        # This ensures vehicles busy in previous steps are properly released
        self._update_vehicle_states()
        
        # STEP 2: Load new trips for this step
        self._load_step_trips()
        
        # STEP 3: Get available vehicles AFTER state update
        # This ensures only vehicles that are truly available (not busy) can be assigned
        available_vehicles = self.get_available_vehicles()
        available_set = set(available_vehicles.cpu().numpy().tolist())
        
        # Process charging
        if charge_vehicles is not None:
            charge_set = set(charge_vehicles.cpu().numpy().tolist())
            charge_vehicles_available = [v for v in charge_set if v in available_set]
            
            # Simple charging: add SOC (20% per step = 20kWh in 5 min at 200kW)
            # Assuming 5 min step = 200kW * 5/60 hours = 16.67 kWh
            charge_per_step = 200.0 * (self.step_duration_minutes / 60.0)  # kWh
            charging_cost = 0.0
            
            for v_idx in charge_vehicles_available:
                if self.vehicle_socs[v_idx] < self.max_soc:
                    energy_to_add = min(charge_per_step, self.max_soc - self.vehicle_socs[v_idx])
                    self.vehicle_socs[v_idx] += energy_to_add
                    self.vehicle_busy_until[v_idx] = self.current_step + 1  # Charge takes 1 step
                    charging_cost += energy_to_add * self.electricity_cost_per_kwh
            
            self.total_charging_cost += charging_cost
        
        # Process serve assignments
        step_revenue = 0.0
        step_driving_cost = 0.0
        served_trip_ids = set()
        
        for vehicle_idx, trip_idx in serve_assignments.items():
            # CRITICAL VALIDATION: Ensure vehicle is available
            # This protects vehicles that are busy from previous steps
            if vehicle_idx not in available_set:
                # Vehicle is busy (busy_until > current_step) - skip this assignment
                # This should not happen if heuristic is correct, but we double-check
                continue  # Vehicle not available (busy from previous assignment)
            
            # Double-check: Vehicle should not be busy
            assert self.vehicle_busy_until[vehicle_idx] <= self.current_step, \
                f"Vehicle {vehicle_idx} is busy until step {self.vehicle_busy_until[vehicle_idx]}, but current_step is {self.current_step}"
            
            if trip_idx >= len(self.unassigned_trips):
                continue  # Invalid trip index
            
            if trip_idx in served_trip_ids:
                continue  # Trip already served
            
            vehicle_idx_tensor = torch.tensor(vehicle_idx, device=self.device)
            trip = self.unassigned_trips[trip_idx]
            
            # Compute distances
            vehicle_hex_idx = self.vehicle_positions[vehicle_idx].item()
            pickup_hex_idx = trip['pickup_hex']
            dropoff_hex_idx = trip['dropoff_hex']
            
            distance_matrix = self.hex_grid.distance_matrix._distances
            pickup_distance = distance_matrix[vehicle_hex_idx, pickup_hex_idx].item()
            trip_distance = trip['distance']
            total_distance = pickup_distance + trip_distance
            
            # Check pickup distance (should already be checked in heuristic, but double-check)
            if pickup_distance > self.max_pickup_distance:
                continue
            
            # Check energy feasibility
            energy_needed = total_distance * self.energy_per_km
            if self.vehicle_socs[vehicle_idx] < (energy_needed + 10.0):  # 10 kWh reserve
                continue  # Not enough energy
            
            # Execute trip
            # 1. Update vehicle SOC
            self.vehicle_socs[vehicle_idx] -= energy_needed
            
            # 2. Set vehicle busy (time = distance / speed, in steps)
            # Formula: duration_hours = distance / speed, duration_minutes = hours * 60, duration_steps = minutes / step_duration
            # This matches ActionProcessor calculation
            avg_speed_kmh = self.config.physics.avg_speed_kmh
            step_minutes = self.step_duration_minutes
            
            duration_hours = total_distance / avg_speed_kmh
            duration_minutes = duration_hours * 60.0
            duration_steps = max(1, int(duration_minutes / step_minutes))
            
            # CRITICAL: Set vehicle busy until future step
            # This vehicle will NOT be available in next (duration_steps - 1) steps
            # until busy_until <= current_step again
            self.vehicle_busy_until[vehicle_idx] = self.current_step + duration_steps
            self.vehicle_target_hex[vehicle_idx] = dropoff_hex_idx
            
            # Verify: Vehicle should now be busy
            assert self.vehicle_busy_until[vehicle_idx] > self.current_step, \
                f"Vehicle {vehicle_idx} busy_until should be > current_step {self.current_step}"
            
            # 3. Calculate revenue and costs
            fare = trip['fare']
            pickup_cost = pickup_distance * self.driving_cost_per_km
            trip_cost = trip_distance * self.driving_cost_per_km
            energy_cost = energy_needed * self.electricity_cost_per_kwh
            
            # Revenue = fare - energy_cost (as per environment calculation)
            revenue = fare - energy_cost
            driving_cost = pickup_cost + trip_cost
            
            step_revenue += revenue
            step_driving_cost += driving_cost
            
            # 4. Mark trip as served
            served_trip_ids.add(trip_idx)
            self.total_trips_served += 1
        
        # Remove served trips from unassigned list (in reverse order to maintain indices)
        for trip_idx in sorted(served_trip_ids, reverse=True):
            del self.unassigned_trips[trip_idx]
        
        # Drop expired trips
        self._drop_expired_trips()
        
        # Accumulate totals
        self.total_revenue += step_revenue
        self.total_driving_cost += step_driving_cost
        
        # Update episode info
        self.episode_info.trips_served = self.total_trips_served
        self.episode_info.trips_loaded = self.total_trips_loaded
        self.episode_info.trips_dropped = self.total_trips_dropped
        self.episode_info.revenue = self.total_revenue
        self.episode_info.driving_cost = self.total_driving_cost
        self.episode_info.energy_cost = self.total_charging_cost
        self.episode_info.avg_soc = self.vehicle_socs.mean().item()
        
        # STEP 4: Advance step
        # IMPORTANT: After advancing, vehicles with busy_until <= new current_step
        # will become available in the next call to _update_vehicle_states()
        self.current_step += 1
        done = self.current_step >= self.episode_steps
        
        return torch.zeros(self.num_vehicles, device=self.device), self.episode_info


class HeuristicMatcher:
    """
    Heuristic matching algorithm: SOC-aware charging + Profit-aware serving.
    """
    
    def __init__(
        self,
        env,
        critical_soc_threshold: float = 20.0,
        target_soc_threshold: float = 90.0,
        max_pickup_distance: float = 150.0,
        min_profit_threshold: float = 0.0  # Minimum profit to serve (can be negative)
    ):
        self.env = env
        self.critical_soc_threshold = critical_soc_threshold
        self.target_soc_threshold = target_soc_threshold
        self.max_pickup_distance = max_pickup_distance
        self.min_profit_threshold = min_profit_threshold
        
        # Get cost parameters from config
        self.driving_cost_per_km = env.config.reward.driving_cost_per_km
        self.electricity_cost_per_kwh = env.config.reward.electricity_cost_per_kwh
        self.energy_per_km = env.config.physics.energy_per_km
        
        # Assignment solver for profit-based matching
        self.assignment_solver = UltraFastGreedyAssignment(
            device=torch.device(env.device),
            max_cost=1e6
        )
    
    def compute_profit_matrix(
        self,
        vehicle_indices: torch.Tensor,
        vehicle_positions: torch.Tensor,
        vehicle_socs: torch.Tensor,
        trip_indices: torch.Tensor,
        trip_pickup_hexes: torch.Tensor,
        trip_fares: torch.Tensor,
        trip_distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute profit matrix [num_vehicles, num_trips].
        
        Profit = fare - pickup_cost - trip_cost - energy_cost
        
        Returns negative profit (as cost) for minimization.
        """
        num_vehicles = len(vehicle_indices)
        num_trips = len(trip_indices)
        
        if num_trips == 0:
            return torch.zeros(num_vehicles, 0, device=self.env.device)
        
        # Get distance matrix
        distance_matrix = self.env.hex_grid.distance_matrix._distances
        
        # Compute pickup distances [num_vehicles, num_trips]
        vehicle_pos_expanded = vehicle_positions.unsqueeze(1).expand(num_vehicles, num_trips)
        trip_pickup_expanded = trip_pickup_hexes.unsqueeze(0).expand(num_vehicles, num_trips)
        
        pickup_distances = distance_matrix[
            vehicle_pos_expanded.reshape(-1),
            trip_pickup_expanded.reshape(-1)
        ].reshape(num_vehicles, num_trips)
        
        # Mask out trips too far
        valid_mask = pickup_distances <= self.max_pickup_distance
        
        # Trip distances [num_trips]
        trip_distances_expanded = trip_distances.unsqueeze(0).expand(num_vehicles, num_trips)
        
        # Total distance = pickup + trip
        total_distances = pickup_distances + trip_distances_expanded
        
        # Energy needed [num_vehicles, num_trips]
        energy_needed = total_distances * self.energy_per_km
        
        # Check energy feasibility: vehicle SOC must be sufficient
        energy_available = vehicle_socs.unsqueeze(1)  # kWh
        feasible_mask = energy_available >= (energy_needed + 10.0)  # 10 kWh reserve
        valid_mask = valid_mask & feasible_mask
        
        # Costs
        pickup_cost = pickup_distances * self.driving_cost_per_km
        trip_cost = trip_distances_expanded * self.driving_cost_per_km
        energy_cost = energy_needed * self.electricity_cost_per_kwh
        
        # Profit = fare - all costs
        fares_expanded = trip_fares.unsqueeze(0).expand(num_vehicles, num_trips)
        profit = fares_expanded - pickup_cost - trip_cost - energy_cost
        
        # Mask invalid/unprofitable trips
        profit[~valid_mask] = -1e6  # Very negative for invalid
        profit[profit < self.min_profit_threshold] = -1e6  # Filter unprofitable
        
        # Return negative profit (as cost) for minimization assignment
        return -profit
    
    def select_actions_simple(
        self,
        env: 'SimpleHeuristicEnvironment'
    ) -> Tuple[Dict[int, int], torch.Tensor]:
        """
        Select actions for SimpleHeuristicEnvironment.
        
        Optimized greedy matching: Match maximum number of trips possible.
        Uses iterative greedy approach to match as many trips as possible.
        
        Returns:
            (serve_assignments, charge_vehicles)
            - serve_assignments: Dict {vehicle_idx: trip_idx_in_unassigned_list}
            - charge_vehicles: Tensor of vehicle indices to charge
        """
        device = self.env.device
        serve_assignments = {}
        
        # Get available vehicles
        available_vehicles = env.get_available_vehicles()
        if len(available_vehicles) == 0:
            return {}, torch.tensor([], dtype=torch.long, device=device)
        
        # Get vehicle states
        vehicle_positions = env.vehicle_positions[available_vehicles]
        vehicle_socs = env.vehicle_socs[available_vehicles]
        
        if getattr(self, 'is_charging', None) is None or len(self.is_charging) != self.env.num_vehicles:
            self.is_charging = torch.zeros(self.env.num_vehicles, dtype=torch.bool, device=device)
            
        # Update charging state based on SOC thresholds
        # Stop charging if SOC >= target_soc_threshold
        self.is_charging[available_vehicles[vehicle_socs >= self.target_soc_threshold]] = False
        
        # Start charging if SOC < critical_soc_threshold
        self.is_charging[available_vehicles[vehicle_socs < self.critical_soc_threshold]] = True
        
        # Active charging mask for currently available vehicles
        active_charge_mask = self.is_charging[available_vehicles]
        charge_vehicles = available_vehicles[active_charge_mask].clone()
        
        # Phase 2: Optimized greedy serving for remaining vehicles
        remaining_mask = ~active_charge_mask
        remaining_vehicles = available_vehicles[remaining_mask]
        
        if len(remaining_vehicles) > 0:
            # Get unassigned trips
            unassigned_trips = env.get_unassigned_trips()
            
            if len(unassigned_trips) > 0:
                # Prepare trip data
                num_vehicles = len(remaining_vehicles)
                num_trips = len(unassigned_trips)
                
                trip_pickup_hexes = torch.tensor([t['pickup_hex'] for t in unassigned_trips], device=device)
                trip_fares = torch.tensor([t['fare'] for t in unassigned_trips], device=device)
                trip_distances = torch.tensor([t['distance'] for t in unassigned_trips], device=device)
                
                remaining_positions = vehicle_positions[remaining_mask]
                remaining_socs = vehicle_socs[remaining_mask]
                
                # Compute profit matrix
                cost_matrix = self.compute_profit_matrix(
                    vehicle_indices=remaining_vehicles,
                    vehicle_positions=remaining_positions,
                    vehicle_socs=remaining_socs,
                    trip_indices=torch.arange(num_trips, device=device),
                    trip_pickup_hexes=trip_pickup_hexes,
                    trip_fares=trip_fares,
                    trip_distances=trip_distances
                )
                
                if cost_matrix.shape[1] > 0:
                    # Optimized greedy matching: Iterative approach to maximize matches
                    # Strategy: Each trip picks its best available vehicle
                    # This ensures we match as many trips as possible
                    
                    # Convert cost to profit (negative cost = positive profit)
                    profit_matrix = -cost_matrix  # [num_vehicles, num_trips]
                    
                    # Mark invalid assignments (where cost_matrix == max_cost)
                    invalid_mask = (cost_matrix >= 1e5)  # Invalid assignments
                    profit_matrix[invalid_mask] = -1e6  # Very negative
                    
                    # Track which vehicles and trips are already matched
                    vehicle_matched = torch.zeros(num_vehicles, dtype=torch.bool, device=device)
                    trip_matched = torch.zeros(num_trips, dtype=torch.bool, device=device)
                    
                    # Iterative greedy: Each trip picks best available vehicle
                    # Sort trips by their best profit (to prioritize valuable trips)
                    best_vehicle_profits_per_trip, _ = profit_matrix.max(dim=0)  # [num_trips]
                    
                    # Sort trips by best profit (descending) to prioritize valuable trips
                    trip_priority_order = best_vehicle_profits_per_trip.argsort(descending=True)
                    
                    for trip_local_idx in trip_priority_order:
                        if trip_matched[trip_local_idx]:
                            continue
                        
                        # Get profits for this trip from all unmatched vehicles
                        trip_profits = profit_matrix[:, trip_local_idx]  # [num_vehicles]
                        
                        # Mask out matched vehicles and invalid assignments
                        available_vehicle_mask = ~vehicle_matched & (trip_profits > -1e5)
                        
                        if available_vehicle_mask.any():
                            # Pick best available vehicle for this trip
                            best_vehicle_local_idx = (trip_profits * available_vehicle_mask.float() - (1 - available_vehicle_mask.float()) * 1e6).argmax()
                            
                            if trip_profits[best_vehicle_local_idx] > -1e5:
                                # Valid match found
                                vehicle_global_idx = remaining_vehicles[best_vehicle_local_idx].item()
                                serve_assignments[vehicle_global_idx] = trip_local_idx.item()
                                
                                # Mark as matched
                                vehicle_matched[best_vehicle_local_idx] = True
                                trip_matched[trip_local_idx] = True
        
        return serve_assignments, charge_vehicles
    
    def select_actions(
        self,
        state
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Select actions using heuristic algorithm.
        
        Returns:
            (action_type, reposition_target)
            - action_type: [num_vehicles] (0=IDLE, 1=SERVE, 2=CHARGE, 3=REPOSITION)
            - reposition_target: None (we don't use reposition)
        """
        device = self.env.device
        num_vehicles = self.env.num_vehicles
        
        # Initialize all to IDLE
        action_type = torch.zeros(num_vehicles, dtype=torch.long, device=device)
        
        # Get available vehicles
        available_mask = self.env.fleet_state.get_available_mask(self.env.current_step)
        available_indices = available_mask.nonzero(as_tuple=True)[0]
        
        if len(available_indices) == 0:
            return action_type, None
        
        # Get vehicle states
        vehicle_positions = self.env.fleet_state.positions[available_indices]
        vehicle_socs = self.env.fleet_state.socs[available_indices]
        
        if getattr(self, 'is_charging', None) is None or len(self.is_charging) != self.env.num_vehicles:
            self.is_charging = torch.zeros(self.env.num_vehicles, dtype=torch.bool, device=device)
            
        # Update charging state based on SOC thresholds
        self.is_charging[available_indices[vehicle_socs >= self.target_soc_threshold]] = False
        self.is_charging[available_indices[vehicle_socs < self.critical_soc_threshold]] = True
        
        active_charge_mask = self.is_charging[available_indices]
        critical_indices = available_indices[active_charge_mask]
        
        if len(critical_indices) > 0:
            # Check if stations are available
            if hasattr(self.env, 'station_state') and self.env.station_state is not None:
                available_ports = self.env.station_state.get_available_ports()
                has_available_station = available_ports.sum() > 0
                
                if has_available_station:
                    # Set all critical vehicles to CHARGE
                    # Environment will handle station assignment automatically
                    action_type[critical_indices] = 2  # CHARGE
        
        # Phase 2: Profit-aware serving for remaining available vehicles
        remaining_mask = ~active_charge_mask
        remaining_indices = available_indices[remaining_mask]
        
        if len(remaining_indices) > 0:
            # Get unassigned trips
            unassigned_mask = self.env.trip_state.get_unassigned_mask()
            
            if unassigned_mask.any():
                trip_indices = unassigned_mask.nonzero(as_tuple=True)[0]
                trip_pickup_hexes = self.env.trip_state.pickup_hex[trip_indices]
                trip_fares = self.env.trip_state.fare[trip_indices]
                trip_distances = self.env.trip_state.distance_km[trip_indices]
                
                remaining_positions = vehicle_positions[remaining_mask]
                remaining_socs = vehicle_socs[remaining_mask]
                
                # Compute profit matrix (negative profit as cost)
                cost_matrix = self.compute_profit_matrix(
                    vehicle_indices=remaining_indices,
                    vehicle_positions=remaining_positions,
                    vehicle_socs=remaining_socs,
                    trip_indices=trip_indices,
                    trip_pickup_hexes=trip_pickup_hexes,
                    trip_fares=trip_fares,
                    trip_distances=trip_distances
                )
                
                if cost_matrix.shape[1] > 0:  # If there are trips
                    # Solve assignment (minimize cost = maximize profit)
                    result: AssignmentResult = self.assignment_solver.solve(
                        cost_matrix=cost_matrix,
                        maximize=False
                    )
                    
                    # Set SERVE action for matched vehicles
                    if len(result.vehicle_indices) > 0:
                        matched_vehicles = remaining_indices[result.vehicle_indices]
                        action_type[matched_vehicles] = 1  # SERVE
        
        # Remaining vehicles stay IDLE (action_type already 0)
        return action_type, None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run heuristic matching baseline simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    
    # Environment
    parser.add_argument('--num-vehicles', type=int, default=None,
                        help='Number of vehicles in fleet')
    parser.add_argument('--num-hexes', type=int, default=None,
                        help='Number of hexagons in grid')
    parser.add_argument('--episode-duration-hours', type=float, default=None,
                        help='Episode duration in hours (e.g., 24.0 for full day)')
    parser.add_argument('--env-v2', action='store_true', default=False,
                        help='Use GPUEnvironmentV2')
    
    # Data
    parser.add_argument('--real-data', type=str, default=None,
                        help='Path to real trip data parquet file')
    parser.add_argument('--trip-sample', type=float, default=1.0,
                        help='Sample ratio for trip data (0.0-1.0). Default: 1.0 (all trips)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Filter trips from this date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Filter trips until this date (YYYY-MM-DD)')
    parser.add_argument('--target-h3-resolution', type=int, default=None,
                        help='Target H3 resolution')
    parser.add_argument('--max-hex-count', type=int, default=None,
                        help='Maximum number of hexes')
    
    # Heuristic parameters
    parser.add_argument('--critical-soc-threshold', type=float, default=20.0,
                        help='SOC threshold for forced charging (percentage)')
    parser.add_argument('--max-pickup-distance', type=float, default=5.0,
                        help='Maximum pickup distance in km (default: 5.0)')
    parser.add_argument('--min-profit-threshold', type=float, default=0.0,
                        help='Minimum profit to serve a trip')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def create_config(args) -> Config:
    """Create config from YAML and CLI arguments."""
    if args.config:
        config = ConfigLoader.from_yaml(args.config)
    else:
        config = Config()
    
    # Override with CLI args
    if args.num_vehicles is not None:
        config.environment.num_vehicles = args.num_vehicles
    if args.num_hexes is not None:
        config.environment.num_hexes = args.num_hexes
    if args.episode_duration_hours is not None:
        config.episode.duration_hours = args.episode_duration_hours
        print(f"[Episode] Duration set to {args.episode_duration_hours} hours "
              f"({config.episode.steps_per_episode} steps @ {config.episode.step_duration_minutes} min/step)")
    
    return config


def create_environment(config: Config, device: torch.device, args, trip_loader: Optional[RealTripLoader] = None):
    """Create GPU environment."""
    num_hexes = config.environment.num_hexes
    device_str = str(device)
    hex_grid = HexGrid(device=device_str)
    
    if trip_loader and trip_loader.is_loaded:
        hex_ids = trip_loader.hex_ids
        lats, lons = trip_loader.get_hex_coordinates()
        num_hexes = len(hex_ids)
        config.environment.num_hexes = num_hexes
        hex_grid._hex_ids = hex_ids
        hex_grid._hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
        hex_grid._latitudes = lats.to(device)
        hex_grid._longitudes = lons.to(device)
        hex_grid._initialized = True
        hex_grid.distance_matrix.compute(hex_grid._latitudes, hex_grid._longitudes, hex_ids=hex_ids)
    else:
        # Fallback: synthetic grid
        grid_size = int(num_hexes ** 0.5) + 1
        fake_hex_ids = [f"hex_{i}" for i in range(num_hexes)]
        hex_grid._hex_ids = fake_hex_ids
        hex_grid._hex_to_idx = {h: i for i, h in enumerate(fake_hex_ids)}
        base_lat, base_lon = 40.7128, -74.0060
        lat_per_km, lon_per_km = 0.009, 0.012
        lats = torch.zeros(num_hexes, device=device)
        lons = torch.zeros(num_hexes, device=device)
        for i in range(num_hexes):
            row = i // grid_size
            col = i % grid_size
            lats[i] = base_lat + row * 0.5 * lat_per_km
            lons[i] = base_lon + col * 0.5 * lon_per_km
        hex_grid._latitudes = lats
        hex_grid._longitudes = lons
        hex_grid._initialized = True
        lat_diff = lats.unsqueeze(1) - lats.unsqueeze(0)
        lon_diff = lons.unsqueeze(1) - lons.unsqueeze(0)
        lat_km = lat_diff / lat_per_km
        lon_km = lon_diff / lon_per_km
        distances = torch.sqrt(lat_km**2 + lon_km**2)
        distances.fill_diagonal_(0)
        hex_grid.distance_matrix._distances = distances
        hex_grid.distance_matrix._num_hexes = num_hexes
    
    if args.env_v2:
        env = GPUEnvironmentV2(
            config=config,
            hex_grid=hex_grid,
            trip_loader=trip_loader,
            device=device_str
        )
    else:
        env = GPUEnvironment(
            config=config,
            hex_grid=hex_grid,
            trip_loader=trip_loader,
            device=device_str
        )
    
    return env


def run_heuristic_simulation_simple(
    env: SimpleHeuristicEnvironment,
    heuristic: HeuristicMatcher,
) -> Dict:
    """
    Run heuristic simulation with SimpleHeuristicEnvironment.
    
    Returns:
        Dictionary with simulation results, including hourly revenue and net profit
    """
    # Track action counts
    action_counts = defaultdict(int)
    
    # Track daily metrics if duration is long
    daily_metrics = defaultdict(lambda: {'revenue': 0.0, 'driving_cost': 0.0, 'energy_cost': 0.0})
    
    # Track hourly metrics
    hourly_metrics = defaultdict(lambda: {'revenue': 0.0, 'driving_cost': 0.0, 'energy_cost': 0.0})
    
    # Reset environment
    state = env.reset()
    max_steps = env.episode_steps
    step_duration_minutes = env.step_duration_minutes
    steps_per_hour = int(60.0 / step_duration_minutes)  # e.g., 60/5 = 12 steps per hour
    
    print(f"\nRunning heuristic simulation for {max_steps} steps...")
    print(f"  Step duration: {step_duration_minutes} minutes ({steps_per_hour} steps/hour)")
    start_time = time.time()
    
    # Track previous cumulative values to compute deltas
    prev_revenue = 0.0
    prev_driving_cost = 0.0
    prev_energy_cost = 0.0
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Select actions using heuristic
        serve_assignments, charge_vehicles = heuristic.select_actions_simple(env)
        
        # Count actions
        # Note: We count actions for ALL vehicles, not just available ones
        # - SERVE: vehicles that are assigned to serve trips
        # - CHARGE: vehicles that are charging
        # - IDLE: vehicles that are available but not serving/charging
        # - BUSY: vehicles that are currently serving (not counted as action, but tracked separately)
        
        num_serve = len(serve_assignments)
        num_charge = len(charge_vehicles) if charge_vehicles is not None else 0
        available_vehicles = env.get_available_vehicles()
        num_idle = len(available_vehicles) - num_serve - num_charge
        
        # Count busy vehicles (vehicles that are not available because they're serving)
        total_vehicles = env.num_vehicles
        num_busy = total_vehicles - len(available_vehicles)
        
        action_counts['SERVE'] += num_serve
        action_counts['CHARGE'] += num_charge
        action_counts['IDLE'] += num_idle
        action_counts['BUSY'] += num_busy  # Track busy vehicles separately
        
        # Step environment
        next_state, info = env.step(serve_assignments, charge_vehicles)
        
        # Calculate incremental revenue and costs for this step
        delta_revenue = info.revenue - prev_revenue
        delta_driving_cost = info.driving_cost - prev_driving_cost
        delta_energy_cost = info.energy_cost - prev_energy_cost
        
        # Determine current hour (0-23)
        current_hour = int((step * step_duration_minutes) // 60) % 24
        
        # Accumulate daily metrics
        current_day = int((step * step_duration_minutes) // (60 * 24))
        daily_metrics[current_day]['revenue'] += delta_revenue
        daily_metrics[current_day]['driving_cost'] += delta_driving_cost
        daily_metrics[current_day]['energy_cost'] += delta_energy_cost
        
        # Accumulate hourly metrics
        hourly_metrics[current_hour]['revenue'] += delta_revenue
        hourly_metrics[current_hour]['driving_cost'] += delta_driving_cost
        hourly_metrics[current_hour]['energy_cost'] += delta_energy_cost
        
        # Update previous values for next iteration
        prev_revenue = info.revenue
        prev_driving_cost = info.driving_cost
        prev_energy_cost = info.energy_cost
        
        state = next_state
        step += 1
        done = step >= max_steps
        
        # Progress update with debug stats
        if step % 20 == 0:
            debug_stats = env.get_debug_stats()
            current_service_rate = info.trips_served / max(info.trips_loaded, 1) * 100
            num_serve_attempted = len(serve_assignments)
            num_charge_attempted = len(charge_vehicles) if charge_vehicles is not None else 0
            print(f"  Step {step}/{max_steps}: Revenue=${info.revenue:.2f}, "
                  f"Trips={info.trips_served}/{info.trips_loaded} (ServiceRate={current_service_rate:.1f}%), "
                  f"Available={debug_stats['available_vehicles']}/{env.num_vehicles}, "
                  f"Unassigned={debug_stats['unassigned_trips']}, "
                  f"ServeAttempted={num_serve_attempted}, ChargeAttempted={num_charge_attempted}, "
                  f"AvgWait={debug_stats['avg_wait_steps']:.1f} steps, AvgSOC={debug_stats['avg_soc']:.1f}")
    
    elapsed_time = time.time() - start_time
    
    # Calculate net profit
    net_profit = info.revenue - info.driving_cost - info.energy_cost
    
    # Service rate
    service_rate = info.trips_served / max(info.trips_loaded, 1)
    
    # Format hourly metrics: calculate net profit for each hour
    hourly_results = {}
    for hour in sorted(hourly_metrics.keys()):
        hour_data = hourly_metrics[hour]
        hourly_net_profit = hour_data['revenue'] - hour_data['driving_cost'] - hour_data['energy_cost']
        hourly_results[int(hour)] = {
            'revenue': float(hour_data['revenue']),
            'driving_cost': float(hour_data['driving_cost']),
            'energy_cost': float(hour_data['energy_cost']),
            'net_profit': float(hourly_net_profit)
        }
    
    # Format daily metrics
    daily_results = {}
    for day in sorted(daily_metrics.keys()):
        day_data = daily_metrics[day]
        daily_net_profit = day_data['revenue'] - day_data['driving_cost'] - day_data['energy_cost']
        daily_results[int(day)] = {
            'revenue': float(day_data['revenue']),
            'driving_cost': float(day_data['driving_cost']),
            'energy_cost': float(day_data['energy_cost']),
            'net_profit': float(daily_net_profit)
        }
    
    results = {
        'total_trips_loaded': int(info.trips_loaded),
        'total_trips_served': int(info.trips_served),
        'total_trips_dropped': int(info.trips_dropped),
        'service_rate': float(service_rate),
        'total_revenue': float(info.revenue),
        'total_driving_cost': float(info.driving_cost),
        'total_charging_cost': float(info.energy_cost),
        'net_profit': float(net_profit),
        'final_avg_soc': float(info.avg_soc),
        'action_counts': dict(action_counts),
        'simulation_time_seconds': float(elapsed_time),
        'steps_completed': int(step),
        'hourly_metrics': hourly_results,
        'daily_metrics': daily_results
    }
    
    return results


def run_heuristic_simulation(
    env,
    heuristic: HeuristicMatcher,
    max_steps: Optional[int] = None
) -> Dict:
    """
    Run heuristic simulation and collect metrics.
    
    Returns:
        Dictionary with simulation results
    """
    # Initialize metrics
    total_revenue = 0.0
    total_charging_cost = 0.0
    total_trips_served = 0
    total_trips_loaded = 0
    total_trips_dropped = 0
    
    # Track action counts
    action_counts = defaultdict(int)
    
    # Reset environment
    state = env.reset()
    max_steps = max_steps or env.episode_steps
    
    print(f"\nRunning heuristic simulation for {max_steps} steps...")
    start_time = time.time()
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Select actions using heuristic
        action_type, reposition_target = heuristic.select_actions(state)
        
        # Count actions (only for available vehicles)
        available_mask = env.fleet_state.get_available_mask(env.current_step)
        available_actions = action_type[available_mask]
        for action_id in available_actions.cpu().numpy():
            if 0 <= action_id < 4:
                action_names = ['IDLE', 'SERVE', 'CHARGE', 'REPOSITION']
                action_counts[action_names[action_id]] += 1
        
        # Step environment
        next_state, reward, done_tensor, info = env.step(action_type, reposition_target)
        done = done_tensor.item() if isinstance(done_tensor, torch.Tensor) else done_tensor
        
        # Note: revenue, energy_cost, trips_served, trips_loaded, trips_dropped 
        # in info are already CUMULATIVE (accumulated by environment),
        # so we just take the final values, don't accumulate again
        total_revenue = info.revenue
        total_charging_cost = info.energy_cost
        total_trips_served = info.trips_served
        total_trips_loaded = info.trips_loaded
        total_trips_dropped = info.trips_dropped
        
        state = next_state
        step += 1
        
        # Progress update
        if step % 20 == 0:
            print(f"  Step {step}/{max_steps}: Revenue=${total_revenue:.2f}, "
                  f"Trips={total_trips_served}/{total_trips_loaded}, "
                  f"ServiceRate={total_trips_served/max(total_trips_loaded,1)*100:.1f}%")
    
    elapsed_time = time.time() - start_time
    
    # Note: Driving cost is handled internally by environment in reward calculation
    # Net profit = revenue - charging_cost (driving cost already factored into revenue)
    # For more accurate calculation, we'd need to track driving cost separately
    net_profit = total_revenue - total_charging_cost
    
    # Service rate
    service_rate = total_trips_served / max(total_trips_loaded, 1)
    
    # Get final state info
    final_avg_soc = info.avg_soc if hasattr(info, 'avg_soc') else 0.0
    
    results = {
        'total_trips_loaded': int(total_trips_loaded),
        'total_trips_served': int(total_trips_served),
        'total_trips_dropped': int(total_trips_dropped),
        'service_rate': float(service_rate),
        'total_revenue': float(total_revenue),
        'total_charging_cost': float(total_charging_cost),
        'net_profit': float(net_profit),
        'final_avg_soc': float(final_avg_soc),
        'action_counts': dict(action_counts),
        'simulation_time_seconds': float(elapsed_time),
        'steps_completed': int(step)
    }
    
    return results


def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load config
    config = create_config(args)
    
    # Load trip data
    trip_loader = None
    if args.real_data:
        data_path = Path(args.real_data)
        if data_path.exists():
            print(f"\nLoading trip data from: {data_path}")
            if args.trip_sample is not None and args.trip_sample < 1.0:
                print(f"  Using {args.trip_sample*100:.1f}% sample of trips")
            try:
                trip_loader = RealTripLoader(
                    parquet_path=str(data_path),
                    device=str(device),
                    sample_ratio=args.trip_sample if args.trip_sample is not None else 1.0,
                    target_h3_resolution=args.target_h3_resolution,
                    max_hex_count=args.max_hex_count,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
                trip_loader.load()
                config.environment.num_hexes = trip_loader.num_hexes
                print(f"  Loaded {trip_loader.total_trips:,} trips after sampling")
            except Exception as e:
                print(f"Failed to load trip data: {e}")
                sys.exit(1)
        else:
            print(f"Trip data file not found: {data_path}")
            sys.exit(1)
    
    # Create simple environment
    print(f"\nCreating simple heuristic environment...")
    print(f"  Vehicles: {config.environment.num_vehicles}")
    print(f"  Hexes: {config.environment.num_hexes}")
    
    if trip_loader is None:
        print("Error: --real-data is required for simple environment")
        sys.exit(1)
    
    # Create hex grid
    hex_grid = HexGrid(device=str(device))
    hex_ids = trip_loader.hex_ids
    lats, lons = trip_loader.get_hex_coordinates()
    num_hexes = len(hex_ids)
    config.environment.num_hexes = num_hexes
    hex_grid._hex_ids = hex_ids
    hex_grid._hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
    hex_grid._latitudes = lats.to(device)
    hex_grid._longitudes = lons.to(device)
    hex_grid._initialized = True
    hex_grid.distance_matrix.compute(hex_grid._latitudes, hex_grid._longitudes, hex_ids=hex_ids)
    
    print(f"  Hex grid initialized with {num_hexes} hexes")
    
    # Create simple environment
    env = SimpleHeuristicEnvironment(
        num_vehicles=config.environment.num_vehicles,
        hex_grid=hex_grid,
        trip_loader=trip_loader,
        config=config,
        device=str(device),
        max_pickup_distance=args.max_pickup_distance
    )
    
    # Create heuristic matcher (need to use a dummy env for initialization, will override methods)
    dummy_env = type('DummyEnv', (), {'device': device, 'config': config, 'hex_grid': hex_grid})()
    heuristic = HeuristicMatcher(
        env=dummy_env,
        critical_soc_threshold=args.critical_soc_threshold,
        max_pickup_distance=args.max_pickup_distance,
        min_profit_threshold=args.min_profit_threshold
    )
    heuristic.env = env  # Replace with simple env
    
    print(f"\nInitializing heuristic matcher...")
    print(f"  Critical SOC threshold: {args.critical_soc_threshold}%")
    print(f"  Max pickup distance: {args.max_pickup_distance} km")
    print(f"  Min profit threshold: ${args.min_profit_threshold:.2f}")
    print(f"  Episode duration: {config.episode.duration_hours} hours ({config.episode.steps_per_episode} steps)")
    
    # Run simulation
    results = run_heuristic_simulation_simple(env, heuristic)
    
    # Add metadata
    results['config'] = {
        'num_vehicles': config.environment.num_vehicles,
        'num_hexes': config.environment.num_hexes,
        'episode_duration_hours': config.episode.duration_hours,
        'steps_per_episode': config.episode.steps_per_episode,
        'trip_sample_ratio': args.trip_sample if args.trip_sample is not None else 1.0,
        'critical_soc_threshold': args.critical_soc_threshold,
        'max_pickup_distance': args.max_pickup_distance,
        'min_profit_threshold': args.min_profit_threshold,
    }
    
    # Add driving cost to results if available
    if 'total_driving_cost' in results:
        results['config']['driving_cost_tracked'] = True
    else:
        results['total_driving_cost'] = 0.0
    
    if args.start_date:
        results['date'] = args.start_date
    
    # Print summary
    print("\n" + "="*60)
    print("HEURISTIC MATCHING RESULTS")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Vehicles: {results['config']['num_vehicles']} (số xe trong fleet)")
    print(f"  Steps: {results['steps_completed']} (episode duration: {results['steps_completed']*5/60:.1f} hours @ 5 min/step)")
    print(f"  Trip sample: {results['config']['trip_sample_ratio']*100:.1f}%")
    print(f"\nTrips:")
    print(f"  Loaded: {results['total_trips_loaded']:,} (tổng trips xuất hiện trong episode)")
    print(f"  Served: {results['total_trips_served']:,} (số trips được serve)")
    print(f"  Dropped: {results['total_trips_dropped']:,} (số trips bị drop do quá thời gian chờ)")
    print(f"  Service rate: {results['service_rate']*100:.2f}% (served/loaded)")
    print(f"\nFinancial:")
    print(f"  Revenue: ${results['total_revenue']:,.2f}")
    print(f"  Driving cost: ${results.get('total_driving_cost', 0.0):,.2f}")
    print(f"  Charging cost: ${results['total_charging_cost']:,.2f}")
    print(f"  Net profit: ${results['net_profit']:,.2f} (Revenue - Driving Cost - Charging Cost)")
    
    # Print hourly or daily metrics depending on duration
    # If duration > 48 hours, print daily metrics to avoid flooding the console
    if 'daily_metrics' in results and results['daily_metrics'] and results['config']['episode_duration_hours'] > 48.0:
        print(f"\nDaily metrics:")
        for day in sorted(results['daily_metrics'].keys()):
            day_data = results['daily_metrics'][day]
            print(f"  Day {day+1:2d}: Revenue=${day_data['revenue']:>10,.2f}, "
                  f"Net Profit=${day_data['net_profit']:>10,.2f}")
    elif 'hourly_metrics' in results and results['hourly_metrics']:
        print(f"\nHourly metrics:")
        for hour in sorted(results['hourly_metrics'].keys()):
            hour_data = results['hourly_metrics'][hour]
            print(f"  Hour {hour:2d}: Revenue=${hour_data['revenue']:>10,.2f}, "
                  f"Net Profit=${hour_data['net_profit']:>10,.2f}")
    
    print(f"\nOther metrics:")
    print(f"  Final avg SOC: {results['final_avg_soc']:.1f}% (battery level trung bình)")
    print(f"  Action distribution: {results['action_counts']}")
    print(f"  Simulation time: {results['simulation_time_seconds']:.2f}s")
    print("="*60)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    else:
        print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

