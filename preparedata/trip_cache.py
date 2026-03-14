import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from .cache_builder import HexCacheBuilder, HexCache

LOGGER = logging.getLogger(__name__)


@dataclass
class TripRecordSlice:
    trip_id: np.ndarray
    pickup_idx: np.ndarray
    dropoff_idx: np.ndarray
    pickup_ts: np.ndarray
    fare: np.ndarray
    distance_km: np.ndarray
    duration_min: np.ndarray


class PreparedTripDataset:
    """
    Memory-mapped dataset built from prepared .npz trip chunks.

    Supports fast time-range queries without pandas overhead.
    """

    def __init__(self, cache_dir: Path, trip_cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.trip_cache_dir = Path(trip_cache_dir)
        self.hex_cache: HexCache = HexCacheBuilder.load_cache(self.cache_dir)
        self.hex_ids = np.array(self.hex_cache.hex_ids, dtype=np.object_)
        self.trip_files = sorted(self.trip_cache_dir.glob("trips_chunk_*.npz"))
        if not self.trip_files:
            raise FileNotFoundError(f"No trip cache files found in {self.trip_cache_dir}")
        self.records = self._load_trip_arrays(self.trip_files)

        # Calculate data time range for looping
        self.min_ts = int(self.records.pickup_ts.min())
        self.max_ts = int(self.records.pickup_ts.max())
        self.data_duration_sec = self.max_ts - self.min_ts

    def _load_trip_arrays(self, files: List[Path]) -> TripRecordSlice:
        arrays = {key: [] for key in ["trip_id", "pickup_idx", "dropoff_idx", "pickup_ts", "fare", "distance_km", "duration_min"]}
        for path in files:
            LOGGER.info("Loading prepared trips chunk %s", path)
            npz = np.load(path, allow_pickle=True)
            for key in arrays.keys():
                if key not in npz:
                    raise KeyError(f"Missing '{key}' in prepared chunk {path}")
                arrays[key].append(npz[key])

        concat = {key: np.concatenate(vals) for key, vals in arrays.items()}
        # ensure sorted by pickup_ts
        order = np.argsort(concat["pickup_ts"])
        sorted_arrays = {key: val[order] for key, val in concat.items()}
        LOGGER.info("Prepared trip dataset loaded with %s rows", len(sorted_arrays["trip_id"]))
        return TripRecordSlice(**sorted_arrays)

    @property
    def num_trips(self) -> int:
        return len(self.records.trip_id)

    def _indices_for_window(self, start_ts: int, end_ts: int) -> np.ndarray:
        pickup_ts = self.records.pickup_ts
        left = np.searchsorted(pickup_ts, start_ts, side="left")
        right = np.searchsorted(pickup_ts, end_ts, side="right")
        return np.arange(left, right, dtype=np.int64)

    def query_window(self, start_dt: datetime, end_dt: datetime, percentage: float = 1.0) -> List[dict]:
        """
        Return trip dicts in [start_dt, end_dt).

        IMPORTANT: Automatically loops data when query time exceeds available data range.
        This allows infinite training without running out of trips.
        """
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())

        # LOOP DATA: Map query timestamps into available data range using modulo
        # If start_ts > max_ts, wrap it back to beginning cyclically
        if start_ts > self.max_ts:
            # Calculate how many full cycles have passed
            offset_from_start = start_ts - self.min_ts
            wrapped_offset = offset_from_start % self.data_duration_sec
            start_ts = self.min_ts + wrapped_offset
            end_ts = start_ts + (end_ts - int(start_dt.timestamp()))

            LOGGER.debug(
                f"[TRIP_LOOP] Query beyond data range, wrapped to "
                f"{datetime.utcfromtimestamp(start_ts)} - {datetime.utcfromtimestamp(end_ts)}"
            )

        indices = self._indices_for_window(start_ts, end_ts)
        if len(indices) == 0:
            LOGGER.warning(
                f"[TRIP_LOOP] No trips found for window {start_dt} - {end_dt} "
                f"(ts: {start_ts} - {end_ts}, data range: {self.min_ts} - {self.max_ts})"
            )
            return []

        original_count = len(indices)
        if percentage < 1.0:
            sample_n = max(1, int(len(indices) * percentage))
            seed = hash(start_dt.strftime("%Y-%m-%d %H:%M:%S")) % (2**32)
            rng = np.random.default_rng(seed)
            indices = rng.choice(indices, size=sample_n, replace=False)
            indices.sort()
            LOGGER.debug(
                f"[SAMPLING] Applied {percentage:.1%} sampling: {original_count:,} → {len(indices):,} trips"
            )

        rec = self.records
        pickup_hexes = self.hex_ids[rec.pickup_idx[indices]]
        dropoff_hexes = self.hex_ids[rec.dropoff_idx[indices]]

        trip_ids = rec.trip_id[indices]
        fares = rec.fare[indices]
        distances = rec.distance_km[indices]
        durations = rec.duration_min[indices]
        pickup_ts = rec.pickup_ts[indices]

        trips = []
        for i in range(len(indices)):
            trips.append(
                {
                    "trip_id": str(trip_ids[i]),
                    "pickup_hex": str(pickup_hexes[i]),
                    "dropoff_hex": str(dropoff_hexes[i]),
                    "distance_km": float(distances[i]),
                    "duration_min": float(durations[i]),
                    "fare": float(fares[i]),
                    "pickup_time": datetime.utcfromtimestamp(int(pickup_ts[i])),
                }
            )
        return trips

