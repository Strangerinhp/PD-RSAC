"""
Real NYC Taxi Data Loader
Loads 13.7M trips from nyc_full dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class RealDataLoader:
    """Load and process real NYC taxi trips"""
    
    def __init__(self, data_dir: str = "Wasserstein/data/nyc_full"):
        self.data_dir = Path(data_dir)
        self.trips_df = None
        self.hexagons = None
        self.metadata = {}
        
    def load(self) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Load all data files"""
        print(f"Loading real NYC data from {self.data_dir}...")
        
        # Load trips
        trips_path = self.data_dir / "trips_processed.parquet"
        self.trips_df = pd.read_parquet(trips_path)
        print(f"✅ Loaded {len(self.trips_df):,} trips")
        
        # Load hexagons
        hex_path = self.data_dir / "hexagons.txt"
        with open(hex_path, 'r') as f:
            # Format: hex_id,lat,lon
            self.hexagons = []
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 1:
                        self.hexagons.append(parts[0])  # Only take hex_id
        print(f"✅ Loaded {len(self.hexagons):,} hexagons")
        
        # Load metadata
        meta_path = self.data_dir / "metadata.txt"
        with open(meta_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    self.metadata[key.strip()] = value.strip()
        print(f"✅ Loaded metadata: {self.metadata.get('date_range', 'N/A')}")
        
        return self.trips_df, self.hexagons, self.metadata
    
    def get_time_window(self, start_hour: int = 0, duration_hours: int = 24) -> pd.DataFrame:
        """Get trips within a time window"""
        if self.trips_df is None:
            self.load()
        
        # Filter by hour
        mask = (self.trips_df['hour'] >= start_hour) & (self.trips_df['hour'] < start_hour + duration_hours)
        filtered = self.trips_df[mask].copy()
        print(f"Time window [{start_hour}:00 - {start_hour+duration_hours}:00]: {len(filtered):,} trips")
        return filtered
    
    def sample_trips(self, n_trips: int = 10000, random_state: int = 42) -> pd.DataFrame:
        """Sample random trips for quick testing"""
        if self.trips_df is None:
            self.load()
        
        sample = self.trips_df.sample(n=min(n_trips, len(self.trips_df)), random_state=random_state)
        print(f"✅ Sampled {len(sample):,} trips")
        return sample
    
    def get_demand_by_hex_hour(self) -> pd.DataFrame:
        """Aggregate demand by hex and hour"""
        if self.trips_df is None:
            self.load()
        
        # Pickup demand
        pickup_demand = self.trips_df.groupby(['pickup_hex', 'hour']).size().reset_index(name='pickups')
        
        # Dropoff demand (potential supply)
        dropoff_demand = self.trips_df.groupby(['dropoff_hex', 'hour']).size().reset_index(name='dropoffs')
        
        # Merge
        demand = pd.merge(
            pickup_demand, 
            dropoff_demand, 
            left_on=['pickup_hex', 'hour'],
            right_on=['dropoff_hex', 'hour'],
            how='outer'
        ).fillna(0)
        
        return demand
    
    def get_trips_for_episode(self, episode_start: datetime, episode_duration_min: int = 60) -> pd.DataFrame:
        """Get trips for a single RL episode"""
        if self.trips_df is None:
            self.load()
        
        # Convert string to datetime if needed
        self.trips_df['pickup_time'] = pd.to_datetime(self.trips_df['pickup_time'])
        
        episode_end = episode_start + timedelta(minutes=episode_duration_min)
        
        # Filter trips in time range
        mask = (self.trips_df['pickup_time'] >= episode_start) & (self.trips_df['pickup_time'] < episode_end)
        episode_trips = self.trips_df[mask].copy()
        
        return episode_trips
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        if self.trips_df is None:
            self.load()
        
        stats = {
            'total_trips': len(self.trips_df),
            'unique_hexagons': len(self.hexagons),
            'avg_distance_km': self.trips_df['distance_km'].mean(),
            'avg_duration_min': self.trips_df['duration_min'].mean(),
            'avg_fare': self.trips_df['fare'].mean(),
            'date_range': f"{self.trips_df['pickup_time'].min()} to {self.trips_df['pickup_time'].max()}"
        }
        
        return stats


if __name__ == "__main__":
    # Test loader
    loader = RealDataLoader()
    trips_df, hexagons, metadata = loader.load()
    
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    stats = loader.get_stats()
    for key, value in stats.items():
        print(f"{key:20s}: {value}")
    
    print("\n" + "="*60)
    print("Sample trips:")
    print("="*60)
    sample = loader.sample_trips(n_trips=5)
    print(sample[['trip_id', 'pickup_hex', 'dropoff_hex', 'distance_km', 'duration_min']])
    
    print("\n" + "="*60)
    print("Peak hour demand:")
    print("="*60)
    hourly_demand = trips_df.groupby('hour').size()
    print(hourly_demand.sort_values(ascending=False).head(10))
