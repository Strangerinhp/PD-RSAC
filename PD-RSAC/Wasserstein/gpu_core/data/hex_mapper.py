"""Hex coordinate mapping utilities."""

import torch
import numpy as np
from typing import Optional, List, Tuple, Union


class HexMapper:
    """
    Maps geographic coordinates to hex cell indices.
    
    Uses H3 library for hex operations but caches results
    for efficient GPU-based lookups.
    """
    
    def __init__(
        self,
        hex_ids: Optional[List[str]] = None,
        resolution: int = 8,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.resolution = resolution
        
        self._hex_ids = hex_ids
        self._hex_to_idx: dict = {}
        self._centers_lat: Optional[torch.Tensor] = None
        self._centers_lon: Optional[torch.Tensor] = None
        
        if hex_ids:
            self._build_index(hex_ids)
    
    def _build_index(self, hex_ids: List[str]) -> None:
        """Build hex ID to index mapping."""
        self._hex_ids = hex_ids
        self._hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
        
        # Get centers
        try:
            import h3
            lats, lons = [], []
            for h in hex_ids:
                lat, lon = h3.cell_to_latlng(h)
                lats.append(lat)
                lons.append(lon)
            
            self._centers_lat = torch.tensor(lats, dtype=torch.float32, device=self.device)
            self._centers_lon = torch.tensor(lons, dtype=torch.float32, device=self.device)
        except ImportError:
            pass
    
    @property
    def num_hexes(self) -> int:
        return len(self._hex_ids) if self._hex_ids else 0
    
    def coord_to_hex_id(self, lat: float, lon: float) -> str:
        """Convert single coordinate to hex ID."""
        try:
            import h3
            return h3.latlng_to_cell(lat, lon, self.resolution)
        except ImportError:
            raise ImportError("h3 library required for coordinate to hex conversion")
    
    def coord_to_hex_idx(self, lat: float, lon: float) -> int:
        """Convert single coordinate to hex index."""
        hex_id = self.coord_to_hex_id(lat, lon)
        return self._hex_to_idx.get(hex_id, -1)
    
    def coords_to_hex_indices(
        self,
        lats: Union[np.ndarray, List[float]],
        lons: Union[np.ndarray, List[float]],
    ) -> torch.Tensor:
        """
        Convert arrays of coordinates to hex indices.
        
        Returns tensor of hex indices (-1 for coordinates outside known hexes).
        """
        n = len(lats)
        indices = torch.full((n,), -1, dtype=torch.long, device=self.device)
        
        try:
            import h3
            for i in range(n):
                hex_id = h3.latlng_to_cell(lats[i], lons[i], self.resolution)
                idx = self._hex_to_idx.get(hex_id, -1)
                if idx >= 0:
                    indices[i] = idx
        except ImportError:
            raise ImportError("h3 library required for coordinate to hex conversion")
        
        return indices
    
    def coords_to_hex_indices_nearest(
        self,
        lats: Union[np.ndarray, List[float], torch.Tensor],
        lons: Union[np.ndarray, List[float], torch.Tensor],
    ) -> torch.Tensor:
        """
        Convert coordinates to nearest known hex index.
        
        Uses GPU for fast nearest-neighbor search.
        """
        if self._centers_lat is None:
            raise ValueError("Hex centers not initialized")
        
        # Convert to tensors
        if not isinstance(lats, torch.Tensor):
            lats = torch.tensor(lats, dtype=torch.float32, device=self.device)
        if not isinstance(lons, torch.Tensor):
            lons = torch.tensor(lons, dtype=torch.float32, device=self.device)
        
        n = len(lats)
        
        # Compute distances to all hex centers (vectorized)
        # Using simplified euclidean distance (good enough for small areas)
        lat_diff = lats.unsqueeze(1) - self._centers_lat.unsqueeze(0)  # [n, num_hexes]
        lon_diff = lons.unsqueeze(1) - self._centers_lon.unsqueeze(0)  # [n, num_hexes]
        
        distances = torch.sqrt(lat_diff ** 2 + lon_diff ** 2)
        
        # Get nearest hex
        indices = distances.argmin(dim=1)
        
        return indices
    
    def hex_idx_to_coords(self, hex_idx: int) -> Tuple[float, float]:
        """Convert hex index to center coordinates."""
        if self._centers_lat is None:
            raise ValueError("Hex centers not initialized")
        
        return (
            self._centers_lat[hex_idx].item(),
            self._centers_lon[hex_idx].item(),
        )
    
    def hex_indices_to_coords(
        self,
        hex_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert tensor of hex indices to coordinate tensors."""
        if self._centers_lat is None:
            raise ValueError("Hex centers not initialized")
        
        return (
            self._centers_lat[hex_indices],
            self._centers_lon[hex_indices],
        )
    
    def initialize_from_bounds(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        resolution: int = 8,
    ) -> List[str]:
        """
        Initialize hex grid from geographic bounds.
        
        Returns list of hex IDs covering the area.
        """
        try:
            import h3
        except ImportError:
            raise ImportError("h3 library required")
        
        self.resolution = resolution
        
        # Get hexes covering the bounding box
        polygon = [
            (min_lat, min_lon),
            (min_lat, max_lon),
            (max_lat, max_lon),
            (max_lat, min_lon),
            (min_lat, min_lon),
        ]
        
        # Convert to H3 polygon format
        hex_ids = list(h3.geo_to_cells(
            {"type": "Polygon", "coordinates": [[(lon, lat) for lat, lon in polygon]]},
            resolution,
        ))
        
        self._build_index(hex_ids)
        return hex_ids
    
    def get_hex_ids(self) -> List[str]:
        """Get all hex IDs."""
        return self._hex_ids or []
    
    def save(self, path: str) -> None:
        """Save hex mapper state."""
        data = {
            "hex_ids": self._hex_ids,
            "resolution": self.resolution,
        }
        torch.save(data, path)
    
    def load(self, path: str) -> None:
        """Load hex mapper state."""
        data = torch.load(path)
        self.resolution = data["resolution"]
        self._build_index(data["hex_ids"])
