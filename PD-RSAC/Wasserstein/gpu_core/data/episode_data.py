"""Episode data management with prefetching."""

import torch
import threading
from queue import Queue
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from .trip_loader import TripLoader
from .hex_mapper import HexMapper


@dataclass
class EpisodeTrips:
    """Pre-processed trip data for one episode."""
    date: str
    step_trips: Dict[int, Tuple[torch.Tensor, ...]]
    total_trips: int


class EpisodeDataManager:
    """
    Manages episode data loading with background prefetching.
    
    Loads and preprocesses trip data in background threads
    for seamless episode transitions.
    """
    
    def __init__(
        self,
        trip_loader: TripLoader,
        hex_mapper: HexMapper,
        step_duration_minutes: float = 5.0,
        episode_duration_hours: float = 10.0,
        trip_percentage: float = 0.3,
        prefetch_count: int = 3,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
    ):
        self.trip_loader = trip_loader
        self.hex_mapper = hex_mapper
        self.step_duration_minutes = step_duration_minutes
        self.episode_duration_hours = episode_duration_hours
        self.trip_percentage = trip_percentage
        self.prefetch_count = prefetch_count
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = torch.device(device)
        
        self.steps_per_episode = int(episode_duration_hours * 60 / step_duration_minutes)
        
        # Available dates
        self._available_dates: List[str] = []
        self._date_idx = 0
        
        # Prefetch queue and cache
        self._prefetch_queue: Queue = Queue()
        self._cache: Dict[str, EpisodeTrips] = {}
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_prefetch = threading.Event()
    
    def initialize(self) -> None:
        """Initialize data manager and start prefetching."""
        # Load data and get available dates
        self.trip_loader.load()
        self._available_dates = self.trip_loader.get_available_dates()
        
        if len(self._available_dates) == 0:
            raise ValueError("No dates available in trip data")
        
        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Start prefetch thread
        self._start_prefetch()
    
    def _start_prefetch(self) -> None:
        """Start background prefetch thread."""
        self._stop_prefetch.clear()
        self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._prefetch_thread.start()
        
        # Queue initial prefetches
        for i in range(min(self.prefetch_count, len(self._available_dates))):
            date = self._available_dates[(self._date_idx + i) % len(self._available_dates)]
            if date not in self._cache:
                self._prefetch_queue.put(date)
    
    def _prefetch_loop(self) -> None:
        """Background prefetch loop."""
        while not self._stop_prefetch.is_set():
            try:
                date = self._prefetch_queue.get(timeout=1.0)
                if date not in self._cache:
                    episode = self._load_episode(date)
                    self._cache[date] = episode
            except Exception:
                continue
    
    def _load_episode(self, date: str) -> EpisodeTrips:
        """Load and preprocess episode data for a date."""
        # Check disk cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"{date}.pt"
            if cache_path.exists():
                cached = torch.load(cache_path)
                return EpisodeTrips(
                    date=date,
                    step_trips=cached["step_trips"],
                    total_trips=cached["total_trips"],
                )
        
        # Load from parquet
        trips_df = self.trip_loader.get_trips_for_date(date, self.trip_percentage)
        
        # Preprocess per step
        step_dfs = self.trip_loader.preprocess_episode(
            trips_df,
            self.step_duration_minutes,
            self.episode_duration_hours,
        )
        
        # Convert to tensors
        step_trips = {}
        total_trips = 0
        
        for step, df in step_dfs.items():
            tensors = self.trip_loader.to_tensors(df, self.hex_mapper)
            step_trips[step] = tensors
            total_trips += len(df)
        
        episode = EpisodeTrips(
            date=date,
            step_trips=step_trips,
            total_trips=total_trips,
        )
        
        # Save to disk cache
        if self.cache_dir:
            torch.save({
                "step_trips": step_trips,
                "total_trips": total_trips,
            }, cache_path)
        
        return episode
    
    def get_episode(self, date: Optional[str] = None) -> EpisodeTrips:
        """
        Get episode data for a date.
        
        If date is None, returns next episode in sequence.
        """
        if date is None:
            date = self._available_dates[self._date_idx]
            self._date_idx = (self._date_idx + 1) % len(self._available_dates)
        
        # Check cache
        if date in self._cache:
            episode = self._cache.pop(date)
        else:
            # Load synchronously if not cached
            episode = self._load_episode(date)
        
        # Queue next prefetches
        for i in range(self.prefetch_count):
            next_idx = (self._date_idx + i) % len(self._available_dates)
            next_date = self._available_dates[next_idx]
            if next_date not in self._cache:
                self._prefetch_queue.put(next_date)
        
        return episode
    
    def get_trips_for_step(
        self,
        episode: EpisodeTrips,
        step: int,
    ) -> Tuple[torch.Tensor, ...]:
        """Get trip tensors for a specific step in episode."""
        if step not in episode.step_trips:
            return (
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
            )
        
        return episode.step_trips[step]
    
    def get_dates(self) -> List[str]:
        """Get list of available dates."""
        return self._available_dates
    
    def shutdown(self) -> None:
        """Stop prefetching and clean up."""
        self._stop_prefetch.set()
        if self._prefetch_thread:
            self._prefetch_thread.join(timeout=5.0)
        self._cache.clear()
    
    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
    
    def __len__(self) -> int:
        return len(self._available_dates)
    
    def __del__(self):
        self.shutdown()
