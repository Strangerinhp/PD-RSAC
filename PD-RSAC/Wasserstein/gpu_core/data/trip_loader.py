"""Trip data loader from parquet files."""

import torch
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta


class TripLoader:
    """
    Loads and preprocesses trip data from parquet files.
    
    Supports streaming/batched loading for memory efficiency.
    """
    
    def __init__(
        self,
        parquet_path: str,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
    ):
        self.parquet_path = Path(parquet_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = torch.device(device)
        
        self._df: Optional[pd.DataFrame] = None
        self._loaded = False
        
        # Column mappings (adjust based on your data format)
        self.pickup_time_col = "pickup_datetime"
        self.dropoff_time_col = "dropoff_datetime"
        self.pickup_lat_col = "pickup_latitude"
        self.pickup_lon_col = "pickup_longitude"
        self.dropoff_lat_col = "dropoff_latitude"
        self.dropoff_lon_col = "dropoff_longitude"
        self.fare_col = "fare_amount"
        self.distance_col = "trip_distance"
    
    def load(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load parquet data into memory."""
        if self._loaded and self._df is not None:
            return self._df
        
        if columns is None:
            columns = [
                self.pickup_time_col,
                self.pickup_lat_col,
                self.pickup_lon_col,
                self.dropoff_lat_col,
                self.dropoff_lon_col,
                self.fare_col,
                self.distance_col,
            ]
        
        # Load from parquet
        self._df = pd.read_parquet(self.parquet_path, columns=columns)
        
        # Parse datetime if needed
        if self.pickup_time_col in self._df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self._df[self.pickup_time_col]):
                self._df[self.pickup_time_col] = pd.to_datetime(self._df[self.pickup_time_col])
        
        self._loaded = True
        return self._df
    
    def get_trips_for_date(
        self,
        date: str,
        percentage: float = 1.0,
    ) -> pd.DataFrame:
        """Get trips for a specific date."""
        if not self._loaded:
            self.load()
        
        # Parse date
        target_date = pd.to_datetime(date).date()
        
        # Filter by date
        mask = self._df[self.pickup_time_col].dt.date == target_date
        trips = self._df[mask].copy()
        
        # Sample if percentage < 1
        if percentage < 1.0:
            n = int(len(trips) * percentage)
            trips = trips.sample(n=n, random_state=42)
        
        return trips
    
    def get_trips_for_step(
        self,
        trips_df: pd.DataFrame,
        step: int,
        step_duration_minutes: float,
        start_hour: int = 0,
    ) -> pd.DataFrame:
        """Get trips starting in a specific step."""
        # Calculate time window for this step
        start_minutes = start_hour * 60 + step * step_duration_minutes
        end_minutes = start_minutes + step_duration_minutes
        
        # Filter by pickup time
        pickup_minutes = (
            trips_df[self.pickup_time_col].dt.hour * 60 +
            trips_df[self.pickup_time_col].dt.minute
        )
        
        mask = (pickup_minutes >= start_minutes) & (pickup_minutes < end_minutes)
        return trips_df[mask]
    
    def preprocess_episode(
        self,
        trips_df: pd.DataFrame,
        step_duration_minutes: float,
        episode_duration_hours: float,
        start_hour: int = 0,
    ) -> Dict[int, pd.DataFrame]:
        """
        Preprocess trips into per-step dictionaries.
        
        Returns:
            Dict mapping step number to trips DataFrame
        """
        steps_per_episode = int(episode_duration_hours * 60 / step_duration_minutes)
        
        step_trips = {}
        for step in range(steps_per_episode):
            step_trips[step] = self.get_trips_for_step(
                trips_df, step, step_duration_minutes, start_hour
            )
        
        return step_trips
    
    def to_tensors(
        self,
        trips_df: pd.DataFrame,
        hex_mapper: "HexMapper",
    ) -> Tuple[torch.Tensor, ...]:
        """
        Convert trips DataFrame to GPU tensors.
        
        Returns:
            (trip_ids, pickup_hexes, dropoff_hexes, fares, distances)
        """
        n = len(trips_df)
        
        if n == 0:
            return (
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
            )
        
        # Generate trip IDs
        trip_ids = torch.arange(n, dtype=torch.long, device=self.device)
        
        # Map coordinates to hex indices
        pickup_hexes = hex_mapper.coords_to_hex_indices(
            trips_df[self.pickup_lat_col].values,
            trips_df[self.pickup_lon_col].values,
        )
        
        dropoff_hexes = hex_mapper.coords_to_hex_indices(
            trips_df[self.dropoff_lat_col].values,
            trips_df[self.dropoff_lon_col].values,
        )
        
        # Extract fares and distances
        fares = torch.tensor(
            trips_df[self.fare_col].values,
            dtype=torch.float32,
            device=self.device,
        )
        
        distances = torch.tensor(
            trips_df[self.distance_col].values,
            dtype=torch.float32,
            device=self.device,
        )
        
        return trip_ids, pickup_hexes, dropoff_hexes, fares, distances
    
    def get_available_dates(self) -> List[str]:
        """Get list of available dates in the data."""
        if not self._loaded:
            self.load()
        
        dates = self._df[self.pickup_time_col].dt.date.unique()
        return [str(d) for d in sorted(dates)]
    
    def get_stats(self) -> Dict:
        """Get statistics about the loaded data."""
        if not self._loaded:
            self.load()
        
        return {
            "total_trips": len(self._df),
            "date_range": (
                str(self._df[self.pickup_time_col].min()),
                str(self._df[self.pickup_time_col].max()),
            ),
            "avg_fare": self._df[self.fare_col].mean(),
            "avg_distance": self._df[self.distance_col].mean(),
        }
