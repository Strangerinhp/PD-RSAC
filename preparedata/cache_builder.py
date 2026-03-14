import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h3
except ImportError as exc:  # pragma: no cover - h3 should exist in runtime env
    raise ImportError("h3 library is required for cache building") from exc

try:
    from scipy.spatial import Delaunay
    from shapely.geometry import MultiPolygon, Polygon, Point
    from shapely.ops import unary_union
    CONCAVE_HULL_AVAILABLE = True
except ImportError:
    CONCAVE_HULL_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

# =============================================================================
# FILTERING CONSTANTS - Adjust these to control hex filtering
# =============================================================================
# Minimum number of trips a hex must have to be included in the cache
# Hexes with fewer trips than this threshold will be filtered out
# Set to 0 to disable frequency filtering
MIN_HEX_TRIP_FREQUENCY: int = 100


@dataclass
class HexCache:
    """Container for cached hex metadata."""

    hex_ids: List[str]
    hex_to_idx: Dict[str, int]
    latlng: np.ndarray  # shape [num_hexes, 2]

    @property
    def size(self) -> int:
        return len(self.hex_ids)


class HexCacheBuilder:
    """
    Build reusable caches for hexagon metadata.

    The cache removes the need to repeatedly call h3 conversions during training.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        hex_file: str = "hexagons.txt",
        resolution: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.hex_file = self.data_dir / hex_file
        self.resolution = resolution
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_hex_ids(self) -> List[str]:
        """Read the list of hex ids from file."""
        if not self.hex_file.exists():
            raise FileNotFoundError(f"Hexagon file not found: {self.hex_file}")

        hex_ids: List[str] = []
        with self.hex_file.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                # allow optional "hex_id,lat,lon"
                hex_ids.append(line.split(",")[0])

        if self.resolution is not None:
            LOGGER.info("Filtering hexagons to resolution %s", self.resolution)
            hex_ids = [hid for hid in hex_ids if h3.get_resolution(hid) == self.resolution]

        if not hex_ids:
            raise ValueError("No hex IDs loaded; check hexagons file content")

        # deduplicate while preserving order
        deduped: List[str] = []
        seen = set()
        for hid in hex_ids:
            if hid not in seen:
                deduped.append(hid)
                seen.add(hid)
        return deduped

    def build_cache(
        self,
        use_concave_hull: bool = False,  # Disabled by default (not useful for pre-filtered data)
        alpha: float = 0.05,
        buffer_distance: float = 0.005,
        min_trip_frequency: Optional[int] = None,  # Use MIN_HEX_TRIP_FREQUENCY if None
    ) -> HexCache:
        """
        Compute lat/lng for each hex and build mapping.

        Args:
            use_concave_hull: If True, filter hexes to concave hull of data
            alpha: Alpha parameter for concave hull (smaller = more concave)
            buffer_distance: Buffer around hull in degrees (~0.01 = 1km)
            min_trip_frequency: Minimum trips per hex to keep (None = use global constant)
        """
        hex_ids = self.load_hex_ids()
        latlng = np.zeros((len(hex_ids), 2), dtype=np.float32)
        for idx, hex_id in enumerate(hex_ids):
            lat, lng = h3.cell_to_latlng(hex_id)
            latlng[idx] = (lat, lng)

        LOGGER.info(f"Loaded {len(hex_ids)} hexes from {self.hex_file}")

        # Apply frequency filtering (remove low-demand hexes)
        freq_threshold = min_trip_frequency if min_trip_frequency is not None else MIN_HEX_TRIP_FREQUENCY
        if freq_threshold > 0:
            hex_ids, latlng = self.filter_by_demand_frequency(
                hex_ids, latlng, min_trips=freq_threshold
            )

        # Apply concave hull filtering
        if use_concave_hull and CONCAVE_HULL_AVAILABLE:
            hex_ids, latlng = self.filter_by_concave_hull(
                hex_ids, latlng, alpha=alpha, buffer_distance=buffer_distance
            )
        elif use_concave_hull and not CONCAVE_HULL_AVAILABLE:
            LOGGER.warning(
                "Concave hull filtering requested but scipy/shapely not available. "
                "Install with: pip install scipy shapely"
            )

        hex_to_idx = {hex_id: idx for idx, hex_id in enumerate(hex_ids)}
        LOGGER.info(f"Final hex cache: {len(hex_ids)} hexes")
        return HexCache(hex_ids=hex_ids, hex_to_idx=hex_to_idx, latlng=latlng)

    def save_cache(self, cache: HexCache, filename: str = "hex_cache.npz") -> Path:
        """Persist cache to disk."""
        output_path = self.output_dir / filename
        LOGGER.info("Saving hex cache to %s", output_path)
        np.savez_compressed(
            output_path,
            hex_ids=np.array(cache.hex_ids, dtype=np.object_),
            latlng=cache.latlng,
        )
        # json mapping for easy loading in Python
        mapping_path = output_path.with_suffix(".json")
        with mapping_path.open("w") as handle:
            json.dump(cache.hex_to_idx, handle)
        return output_path

    @staticmethod
    def _compute_alpha_shape(points: np.ndarray, alpha: float) -> Optional[Polygon]:
        """
        Compute concave hull (alpha shape) of a set of points.

        Args:
            points: np.ndarray of shape [N, 2] (lat, lon)
            alpha: Alpha parameter - smaller = more concave, larger = more convex
                   Typical values: 0.01 - 0.1 for lat/lon coordinates

        Returns:
            Shapely Polygon or MultiPolygon representing the concave hull
        """
        if not CONCAVE_HULL_AVAILABLE:
            LOGGER.warning("scipy/shapely not available, skipping concave hull")
            return None

        if len(points) < 4:
            return None

        # Compute Delaunay triangulation
        tri = Delaunay(points)

        # Filter triangles by circumradius (alpha criterion)
        triangles = []
        for simplex in tri.simplices:
            pts = points[simplex]
            # Compute circumradius of triangle
            a = np.linalg.norm(pts[0] - pts[1])
            b = np.linalg.norm(pts[1] - pts[2])
            c = np.linalg.norm(pts[2] - pts[0])
            s = (a + b + c) / 2.0
            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
            if area > 0:
                circumradius = (a * b * c) / (4.0 * area)
                # Keep triangle if circumradius < 1/alpha
                if circumradius < 1.0 / alpha:
                    triangles.append(Polygon(pts))

        if not triangles:
            LOGGER.warning("No triangles passed alpha filter, using convex hull")
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            return Polygon(points[hull.vertices])

        # Merge all triangles into one shape
        concave_hull = unary_union(triangles)
        return concave_hull

    def filter_by_demand_frequency(
        self,
        hex_ids: List[str],
        latlng: np.ndarray,
        min_trips: int = 100,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Filter hexes by demand frequency from trip data.

        Args:
            hex_ids: List of hex IDs
            latlng: Corresponding lat/lon coordinates [N, 2]
            min_trips: Minimum number of trips (pickup OR dropoff) to keep hex

        Returns:
            Filtered (hex_ids, latlng)
        """
        original_count = len(hex_ids)
        LOGGER.info(f"Filtering hexes by demand frequency (min_trips={min_trips})...")

        # Load trip data to count frequency per hex
        trips_file = self.data_dir / "trips_processed.parquet"
        if not trips_file.exists():
            LOGGER.warning(f"Trip file not found: {trips_file}, skipping frequency filter")
            return hex_ids, latlng

        try:
            import pandas as pd
            df = pd.read_parquet(trips_file)

            # Count trips per hex (both pickup and dropoff)
            pickup_col = 'origin_hex' if 'origin_hex' in df.columns else 'pickup_hex'
            dropoff_col = 'dest_hex' if 'dest_hex' in df.columns else 'dropoff_hex'

            pickup_counts = df[pickup_col].value_counts()
            dropoff_counts = df[dropoff_col].value_counts()

            # Total frequency = pickup + dropoff
            hex_frequency: Dict[str, int] = {}
            for hex_id in hex_ids:
                freq = pickup_counts.get(hex_id, 0) + dropoff_counts.get(hex_id, 0)
                hex_frequency[hex_id] = freq

            # Filter hexes
            filtered_hex_ids = []
            filtered_latlng = []

            for hex_id, coord in zip(hex_ids, latlng):
                if hex_frequency.get(hex_id, 0) >= min_trips:
                    filtered_hex_ids.append(hex_id)
                    filtered_latlng.append(coord)

            filtered_latlng = np.array(filtered_latlng, dtype=np.float32) if filtered_latlng else np.empty((0, 2), dtype=np.float32)
            filtered_count = len(filtered_hex_ids)
            removed_count = original_count - filtered_count

            # Log statistics
            freqs = list(hex_frequency.values())
            LOGGER.info(
                f"Demand frequency stats: min={min(freqs)}, max={max(freqs)}, "
                f"median={np.median(freqs):.0f}, mean={np.mean(freqs):.0f}"
            )
            LOGGER.info(
                f"Frequency filtering: {original_count} -> {filtered_count} hexes "
                f"(removed {removed_count}, {removed_count/original_count*100:.1f}%)"
            )

            return filtered_hex_ids, filtered_latlng

        except Exception as e:
            LOGGER.warning(f"Failed to filter by frequency: {e}, keeping all hexes")
            return hex_ids, latlng

    def filter_by_concave_hull(
        self,
        hex_ids: List[str],
        latlng: np.ndarray,
        alpha: float = 0.05,
        buffer_distance: float = 0.005  # ~0.5km buffer in degrees
    ) -> Tuple[List[str], np.ndarray]:
        """
        Filter hexes to keep only those within the concave hull.

        Args:
            hex_ids: List of hex IDs
            latlng: Corresponding lat/lon coordinates [N, 2]
            alpha: Alpha parameter for concave hull (smaller = more concave)
            buffer_distance: Buffer around hull in degrees (~0.01 = 1km)

        Returns:
            Filtered (hex_ids, latlng)
        """
        original_count = len(hex_ids)
        LOGGER.info(f"Computing concave hull for {original_count} hexes (alpha={alpha})...")

        concave_hull = self._compute_alpha_shape(latlng, alpha)

        if concave_hull is None:
            LOGGER.warning("Could not compute concave hull, keeping all hexes")
            return hex_ids, latlng

        # Add buffer to hull
        buffered_hull = concave_hull.buffer(buffer_distance)

        # Filter hexes within buffered hull
        filtered_hex_ids = []
        filtered_latlng = []

        for hex_id, coord in zip(hex_ids, latlng):
            point = Point(coord[0], coord[1])  # lat, lon
            if buffered_hull.contains(point) or buffered_hull.touches(point):
                filtered_hex_ids.append(hex_id)
                filtered_latlng.append(coord)

        filtered_latlng = np.array(filtered_latlng, dtype=np.float32)
        filtered_count = len(filtered_hex_ids)
        removed_count = original_count - filtered_count

        LOGGER.info(
            f"Concave hull filtering: {original_count} -> {filtered_count} hexes "
            f"(removed {removed_count}, {removed_count/original_count*100:.1f}%)"
        )

        return filtered_hex_ids, filtered_latlng

    @staticmethod
    def load_cache(cache_dir: Path, filename: str = "hex_cache.npz") -> HexCache:
        """Load cache from disk."""
        cache_path = Path(cache_dir) / filename
        mapping_path = cache_path.with_suffix(".json")
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
        npz = np.load(cache_path, allow_pickle=True)
        latlng = npz["latlng"]
        hex_ids = npz["hex_ids"].tolist()
        if mapping_path.exists():
            with mapping_path.open("r") as handle:
                hex_to_idx = json.load(handle)
        else:
            hex_to_idx = {hex_id: idx for idx, hex_id in enumerate(hex_ids)}
        return HexCache(hex_ids=hex_ids, hex_to_idx=hex_to_idx, latlng=latlng)

