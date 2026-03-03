"""Feature builder for neural network inputs."""

import torch
from typing import Dict, Optional, Tuple

from ..state import TensorFleetState, TensorTripState, TensorStationState, VehicleStatus
from ..spatial import HexGrid


class FeatureBuilder:
    """
    Builds feature tensors for neural network inputs.
    
    Combines vehicle, trip, station, and spatial features
    into efficient GPU tensors.
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
        self.max_soc = max_soc
        
        # Pre-compute normalized adjacency for GCN
        self._norm_adj: Optional[torch.Tensor] = None
    
    @property
    def norm_adjacency(self) -> torch.Tensor:
        if self._norm_adj is None:
            self._norm_adj = self.hex_grid.get_normalized_adjacency()
        return self._norm_adj
    
    def build_vehicle_features(
        self,
        fleet: TensorFleetState,
        current_step: int,
        max_steps: int,
        trips: Optional[TensorTripState] = None,
        stations: Optional[TensorStationState] = None,
        max_pickup_distance: float = 10.0,
    ) -> torch.Tensor:
        """
        Build per-vehicle feature tensor with spatial awareness.
        
        Returns:
            [num_vehicles, vehicle_feature_dim] tensor
        """
        # Normalize features
        norm_positions = fleet.positions.float() / max(self.num_hexes - 1, 1)
        norm_soc = fleet.socs / self.max_soc
        norm_step = current_step / max(max_steps, 1)
        
        # One-hot status
        status_onehot = torch.zeros(
            self.num_vehicles, len(VehicleStatus),
            dtype=torch.float32, device=self.device
        )
        status_onehot.scatter_(1, fleet.status.unsqueeze(1).long(), 1.0)
        
        # Time until available
        time_until_free = torch.clamp(
            (fleet.busy_until - current_step).float() / 10.0,
            min=0.0, max=1.0
        )
        
        # Is charging
        is_charging = (fleet.status == VehicleStatus.CHARGING).float()
        
        # Low SOC indicator
        low_soc = (fleet.socs < 20.0).float()

        # High SOC indicator (helps avoid unnecessary charging)
        high_soc = (fleet.socs > 60.0).float()

        # Fleet coordination: ratio of busy vehicles (understand fleet utilization)
        busy_mask = fleet.busy_until > current_step
        busy_ratio = busy_mask.sum().float() / max(self.num_vehicles, 1)

        # === NEW: Spatial awareness features ===
        # These help model learn WHEN to SERVE vs REPOSITION

        # Default values if no trips/stations provided
        demand_nearby = torch.zeros(self.num_vehicles, device=self.device)
        nearest_trip_dist = torch.ones(self.num_vehicles, device=self.device)  # 1.0 = far
        nearest_station_dist = torch.ones(self.num_vehicles, device=self.device)
        
        if trips is not None and hasattr(self.hex_grid, 'distance_matrix'):
            unassigned_mask = trips.get_unassigned_mask()
            if unassigned_mask.any():
                trip_hexes = trips.pickup_hex[unassigned_mask]
                
                # Only process if we have actual trips after masking
                if len(trip_hexes) > 0:
                    # Get distance matrix
                    dist_matrix = self.hex_grid.distance_matrix._distances  # [num_hexes, num_hexes]
                    
                    # For each vehicle, count trips within pickup_distance
                    vehicle_positions = fleet.positions  # [num_vehicles]
                    
                    # Distances from each vehicle to each trip [num_vehicles, num_trips]
                    veh_to_trip_dist = dist_matrix[vehicle_positions][:, trip_hexes]
                    
                    # Count trips within range
                    trips_in_range = (veh_to_trip_dist <= max_pickup_distance).sum(dim=1).float()
                    demand_nearby = torch.clamp(trips_in_range / 10.0, max=1.0)  # Normalize
                    
                    # Distance to nearest trip (normalized by max_pickup_distance)
                    # CRITICAL: Protect against division by zero AND empty tensor min()
                    if veh_to_trip_dist.numel() > 0:  # Safety check
                        min_dist = veh_to_trip_dist.min(dim=1).values
                        # Avoid division by very small values
                        safe_divisor = max(max_pickup_distance, 0.1)  # At least 100m
                        nearest_trip_dist = torch.clamp(min_dist / safe_divisor, max=1.0)
                    # else: nearest_trip_dist stays at default (1.0)
        
        if stations is not None and hasattr(self.hex_grid, 'distance_matrix'):
            station_hexes = stations.hex_ids
            
            # Only compute if we have stations
            if len(station_hexes) > 0:
                dist_matrix = self.hex_grid.distance_matrix._distances
                vehicle_positions = fleet.positions
                
                # Distance to nearest station
                # CRITICAL: Protect against empty tensor min()
                veh_to_station_dist = dist_matrix[vehicle_positions][:, station_hexes]
                if veh_to_station_dist.numel() > 0:  # Safety check
                    min_station_dist = veh_to_station_dist.min(dim=1).values
                    nearest_station_dist = torch.clamp(min_station_dist / 20.0, max=1.0)  # 20km max
                # else: nearest_station_dist stays at default (1.0)
        
        features = torch.cat([
            norm_positions.unsqueeze(1),  # [N, 1] - normalized hex position
            norm_soc.unsqueeze(1),  # [N, 1] - normalized SOC
            status_onehot,  # [N, 5] - vehicle status (IDLE/SERVING/CHARGING/etc)
            time_until_free.unsqueeze(1),  # [N, 1] - normalized time until available
            is_charging.unsqueeze(1),  # [N, 1] - binary charging indicator
            low_soc.unsqueeze(1),  # [N, 1] - binary low SOC warning
            high_soc.unsqueeze(1),  # [N, 1] - binary high SOC indicator
            torch.full((self.num_vehicles, 1), norm_step, device=self.device),  # [N, 1] - episode progress
            torch.full((self.num_vehicles, 1), busy_ratio, device=self.device),  # [N, 1] - fleet utilization
            demand_nearby.unsqueeze(1),  # [N, 1] - trips nearby count
            nearest_trip_dist.unsqueeze(1),  # [N, 1] - distance to nearest trip
            nearest_station_dist.unsqueeze(1),  # [N, 1] - distance to nearest station
        ], dim=1)

        return features  # [N, 16]
    
    def build_hex_features(
        self,
        fleet: TensorFleetState,
        trips: TensorTripState,
        stations: TensorStationState,
        current_step: int,
    ) -> torch.Tensor:
        """
        Build per-hex feature tensor φ_h for GCN.
        
        Paper notation: φ_h (phi_h) - hex features for spatial graph reasoning.
        
        Returns:
            [num_hexes, hex_feature_dim] tensor
        
        Alias: phi_h = hex_features
        """
        # Vehicle counts per hex
        vehicle_counts = torch.zeros(self.num_hexes, dtype=torch.float32, device=self.device)
        vehicle_counts.scatter_add_(
            0,
            fleet.positions,
            torch.ones(self.num_vehicles, device=self.device)
        )
        norm_vehicle_counts = vehicle_counts / max(self.num_vehicles, 1)
        
        # Available vehicle counts per hex
        available_mask = fleet.get_available_mask(current_step)
        available_counts = torch.zeros(self.num_hexes, dtype=torch.float32, device=self.device)
        available_positions = fleet.positions[available_mask]
        if len(available_positions) > 0:
            available_counts.scatter_add_(
                0,
                available_positions,
                torch.ones(len(available_positions), device=self.device)
            )
        norm_available = available_counts / max(self.num_vehicles, 1)
        
        # Trip demand per hex (unassigned trips)
        trip_demand = torch.zeros(self.num_hexes, dtype=torch.float32, device=self.device)
        unassigned_mask = trips.get_unassigned_mask()
        if unassigned_mask.any():
            pickup_hexes = trips.pickup_hex[unassigned_mask]
            trip_demand.scatter_add_(
                0,
                pickup_hexes,
                torch.ones(len(pickup_hexes), device=self.device)
            )
        norm_demand = torch.clamp(trip_demand / 10.0, max=1.0)
        
        # Station presence and availability
        station_presence = torch.zeros(self.num_hexes, dtype=torch.float32, device=self.device)
        station_availability = torch.zeros(self.num_hexes, dtype=torch.float32, device=self.device)
        
        station_hexes = stations.hex_ids
        station_presence[station_hexes] = 1.0
        
        available_ports = stations.get_available_ports().float()
        max_ports = stations.ports.float()
        availability_ratio = available_ports / torch.clamp(max_ports, min=1.0)
        station_availability[station_hexes] = availability_ratio
        
        features = torch.stack([
            norm_vehicle_counts,
            norm_available,
            norm_demand,
            station_presence,
            station_availability,
        ], dim=1)
        
        return features  # [num_hexes, 5]
    
    def build_context_features(
        self,
        current_step: int,
        max_steps: int,
        fleet: TensorFleetState,
        trips: TensorTripState,
    ) -> torch.Tensor:
        """
        Build global context features.
        
        Returns:
            [context_dim] tensor
        """
        # Time features
        progress = current_step / max(max_steps, 1)
        remaining = 1.0 - progress
        
        # Fleet stats
        mean_soc = fleet.socs.mean() / self.max_soc
        min_soc = fleet.socs.min() / self.max_soc
        
        status_counts = fleet.get_status_counts()
        serving_ratio = status_counts.get("SERVING", 0) / self.num_vehicles
        charging_ratio = status_counts.get("CHARGING", 0) / self.num_vehicles
        available_ratio = status_counts.get("IDLE", 0) / self.num_vehicles
        
        # Trip stats
        active_trips = trips.num_active
        total_fare = trips.total_fare
        norm_trips = min(active_trips / 100.0, 1.0)
        norm_fare = min(total_fare / 1000.0, 1.0)
        
        context = torch.tensor([
            progress,
            remaining,
            mean_soc,
            min_soc,
            serving_ratio,
            charging_ratio,
            available_ratio,
            norm_trips,
            norm_fare,
        ], dtype=torch.float32, device=self.device)
        
        return context  # [9]
    
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
        Build complete state dictionary for policy network.
        
        Returns:
            Dict with 'vehicle', 'hex', 'context', 'adjacency' tensors
        """
        return {
            "vehicle": self.build_vehicle_features(
                fleet, current_step, max_steps, 
                trips=trips, stations=stations, 
                max_pickup_distance=max_pickup_distance
            ),
            "hex": self.build_hex_features(fleet, trips, stations, current_step),
            "context": self.build_context_features(current_step, max_steps, fleet, trips),
            "adjacency": self.norm_adjacency,
        }
    
    def build_batch_state(
        self,
        states: list,
    ) -> Dict[str, torch.Tensor]:
        """
        Stack multiple states into batch.
        
        Args:
            states: List of state dicts from build_state()
            
        Returns:
            Batched state dict with [batch_size, ...] tensors
        """
        return {
            "vehicle": torch.stack([s["vehicle"] for s in states]),
            "hex": torch.stack([s["hex"] for s in states]),
            "context": torch.stack([s["context"] for s in states]),
            "adjacency": states[0]["adjacency"],  # Same for all
        }
    
    def build_trip_features(
        self,
        trips: TensorTripState,
        fleet: TensorFleetState,
        max_trips: int = 500
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build per-trip features for EnhancedActor's serve head.
        
        Features per trip (8 dims):
        - Normalized pickup hex
        - Normalized dropoff hex  
        - Normalized fare
        - Normalized distance
        - Normalized wait time
        - Demand density at pickup hex
        - Estimated pickup time (based on nearest vehicle)
        - Priority score
        
        Args:
            trips: Current trip state
            fleet: Current fleet state
            max_trips: Maximum number of trips to output features for
            
        Returns:
            (trip_features, trip_mask): 
                trip_features: [max_trips, 8] tensor
                trip_mask: [max_trips] boolean mask of valid trips
        """
        unassigned_mask = trips.get_unassigned_mask()
        unassigned_indices = unassigned_mask.nonzero(as_tuple=True)[0]
        num_trips = min(len(unassigned_indices), max_trips)
        
        # Initialize output tensors
        trip_features = torch.zeros(max_trips, 8, dtype=torch.float32, device=self.device)
        trip_mask = torch.zeros(max_trips, dtype=torch.bool, device=self.device)
        
        if num_trips == 0:
            return trip_features, trip_mask
        
        # Get trip data for first max_trips unassigned trips
        indices = unassigned_indices[:num_trips]
        
        pickup_hexes = trips.pickup_hex[indices]
        dropoff_hexes = trips.dropoff_hex[indices]
        fares = trips.fare[indices]
        distances = trips.distance_km[indices]
        wait_times = trips.wait_steps[indices]
        
        # Normalize features
        trip_features[:num_trips, 0] = pickup_hexes.float() / max(self.num_hexes - 1, 1)
        trip_features[:num_trips, 1] = dropoff_hexes.float() / max(self.num_hexes - 1, 1)
        trip_features[:num_trips, 2] = torch.clamp(fares / 50.0, max=1.0)  # Normalize fare
        trip_features[:num_trips, 3] = torch.clamp(distances / 20.0, max=1.0)  # Normalize distance
        trip_features[:num_trips, 4] = torch.clamp(wait_times.float() / 10.0, max=1.0)  # Wait time
        
        # Compute demand density at pickup hexes
        trip_demand = torch.zeros(self.num_hexes, dtype=torch.float32, device=self.device)
        all_pickup = trips.pickup_hex[unassigned_mask]
        trip_demand.scatter_add_(0, all_pickup, torch.ones(len(all_pickup), device=self.device))
        demand_at_pickup = trip_demand[pickup_hexes]
        trip_features[:num_trips, 5] = torch.clamp(demand_at_pickup / 10.0, max=1.0)
        
        # Compute fare per km ratio (profitability)
        fare_per_km = fares / torch.clamp(distances, min=0.1)
        trip_features[:num_trips, 6] = torch.clamp(fare_per_km / 5.0, max=1.0)
        
        # Priority score based on wait time (longer wait = higher priority)
        priority = 1.0 - torch.exp(-wait_times.float() / 5.0)
        trip_features[:num_trips, 7] = priority
        
        # Set valid mask
        trip_mask[:num_trips] = True
        
        return trip_features, trip_mask
    
    def build_station_features(
        self,
        stations: TensorStationState,
        fleet: TensorFleetState,
        current_step: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build per-station features for EnhancedActor's charge head.
        
        Features per station (6 dims):
        - Normalized station hex
        - Available ports ratio
        - Queue length (vehicles waiting)
        - Price per kWh (normalized)
        - Distance to fleet centroid
        - Utilization rate
        
        Args:
            stations: Current station state
            fleet: Current fleet state
            current_step: Current time step
            
        Returns:
            (station_features, station_mask):
                station_features: [num_stations, 6] tensor
                station_mask: [num_stations] boolean mask of available stations
        """
        num_stations = len(stations.hex_ids)
        
        # Initialize output tensors
        station_features = torch.zeros(num_stations, 6, dtype=torch.float32, device=self.device)
        station_mask = torch.zeros(num_stations, dtype=torch.bool, device=self.device)
        
        if num_stations == 0:
            return station_features, station_mask
        
        # Station hex positions
        station_hexes = stations.hex_ids
        station_features[:, 0] = station_hexes.float() / max(self.num_hexes - 1, 1)
        
        # Available ports ratio
        available = stations.get_available_ports().float()
        total_ports = stations.ports.float()
        availability_ratio = available / torch.clamp(total_ports, min=1.0)
        station_features[:, 1] = availability_ratio
        
        # Queue length (number of vehicles charging at station)
        charging_mask = fleet.status == VehicleStatus.CHARGING.value
        vehicles_per_station = torch.zeros(num_stations, device=self.device)
        if charging_mask.any():
            charging_stations = fleet.charging_station[charging_mask]
            valid_stations = (charging_stations >= 0) & (charging_stations < num_stations)
            if valid_stations.any():
                vehicles_per_station.scatter_add_(
                    0,
                    charging_stations[valid_stations].long(),
                    torch.ones(valid_stations.sum(), device=self.device)
                )
        station_features[:, 2] = torch.clamp(vehicles_per_station / 10.0, max=1.0)
        
        # Price per kWh (assume uniform for now, can be extended)
        station_features[:, 3] = 0.5  # Normalized price
        
        # Distance to fleet centroid
        fleet_positions = fleet.positions.float()
        centroid = fleet_positions.mean()
        station_distances = torch.abs(station_hexes.float() - centroid)
        station_features[:, 4] = torch.clamp(station_distances / self.num_hexes, max=1.0)
        
        # Utilization rate
        utilization = 1.0 - availability_ratio
        station_features[:, 5] = utilization
        
        # Stations with available ports are valid
        station_mask = available > 0
        
        return station_features, station_mask
