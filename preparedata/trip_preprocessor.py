import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .cache_builder import HexCache, HexCacheBuilder

LOGGER = logging.getLogger(__name__)


@dataclass
class TripChunkSummary:
    chunk_id: int
    num_rows: int
    output_path: Path


class TripPreprocessor:
    """Convert trip parquet data into compact numpy caches."""

    REQUIRED_COLUMNS = [
        "trip_id",
        "pickup_hex",
        "dropoff_hex",
        "pickup_time",
        "fare",
        "distance_km",
        "duration_min",
    ]

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        cache_dir: Path,
        parquet_file: str = "trips_processed.parquet",
        chunk_size_rows: int = 500_000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        percentage: float = 1.0,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.parquet_path = self.data_dir / parquet_file
        self.chunk_size_rows = chunk_size_rows
        self.start_dt = pd.to_datetime(start_date) if start_date else None
        self.end_dt = pd.to_datetime(end_date) if end_date else None
        self.percentage = max(0.0, min(1.0, percentage))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hex_cache = HexCacheBuilder.load_cache(self.cache_dir)

    def _iter_dataframes(self) -> Iterable[pd.DataFrame]:
        """Yield dataframes per row group."""
        parquet = pq.ParquetFile(self.parquet_path)
        columns = list({*self.REQUIRED_COLUMNS, "pickup_datetime"})
        for rg in range(parquet.num_row_groups):
            table = parquet.read_row_group(rg, columns=columns)
            df = table.to_pandas()
            yield df

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # ensure pickup_time column exists
        if "pickup_time" not in df.columns and "pickup_datetime" in df.columns:
            df["pickup_time"] = pd.to_datetime(df["pickup_datetime"])
        else:
            df["pickup_time"] = pd.to_datetime(df["pickup_time"])

        if self.start_dt is not None:
            df = df[df["pickup_time"] >= self.start_dt]
        if self.end_dt is not None:
            df = df[df["pickup_time"] <= self.end_dt]

        if self.percentage < 1.0 and len(df) > 0:
            sample_n = max(1, int(len(df) * self.percentage))
            df = df.sample(n=sample_n, random_state=42)

        return df

    def _encode_hex_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        pickup_idx = df["pickup_hex"].map(self.hex_cache.hex_to_idx)
        dropoff_idx = df["dropoff_hex"].map(self.hex_cache.hex_to_idx)
        df = df[pickup_idx.notna() & dropoff_idx.notna()].copy()
        df["pickup_idx"] = pickup_idx.loc[df.index].astype(np.int32)
        df["dropoff_idx"] = dropoff_idx.loc[df.index].astype(np.int32)
        return df

    def _build_arrays(self, df: pd.DataFrame) -> dict:
        pickup_ts = df["pickup_time"].view("int64") // 10**9  # seconds
        arrays = {
            "trip_id": df["trip_id"].astype(str).values,
            "pickup_idx": df["pickup_idx"].to_numpy(dtype=np.int32),
            "dropoff_idx": df["dropoff_idx"].to_numpy(dtype=np.int32),
            "pickup_ts": pickup_ts.to_numpy(dtype=np.int64),
            "fare": df["fare"].fillna(0.0).to_numpy(dtype=np.float32),
            "distance_km": df["distance_km"].fillna(0.0).to_numpy(dtype=np.float32),
            "duration_min": df["duration_min"].fillna(0.0).to_numpy(dtype=np.float32),
        }
        return arrays

    def process(self) -> List[TripChunkSummary]:
        """Process parquet into chunked npz files."""
        summaries: List[TripChunkSummary] = []
        chunk_id = 0
        rows_buffer: List[pd.DataFrame] = []
        rows_count = 0

        for df in self._iter_dataframes():
            df = self._normalize_dataframe(df)
            if df.empty:
                continue
            df = self._encode_hex_indices(df)
            if df.empty:
                continue

            rows_buffer.append(df)
            rows_count += len(df)

            if rows_count >= self.chunk_size_rows:
                summaries.append(self._flush_buffer(rows_buffer, chunk_id))
                chunk_id += 1
                rows_buffer = []
                rows_count = 0

        if rows_buffer:
            summaries.append(self._flush_buffer(rows_buffer, chunk_id))
        return summaries

    def _flush_buffer(self, dfs: List[pd.DataFrame], chunk_id: int) -> TripChunkSummary:
        chunk_df = pd.concat(dfs, ignore_index=True)
        chunk_df.sort_values("pickup_time", inplace=True, ignore_index=True)
        arrays = self._build_arrays(chunk_df)
        output_path = self.output_dir / f"trips_chunk_{chunk_id:03d}.npz"
        LOGGER.info("Writing %s rows to %s", len(chunk_df), output_path)
        np.savez_compressed(output_path, **arrays)
        return TripChunkSummary(chunk_id=chunk_id, num_rows=len(chunk_df), output_path=output_path)

