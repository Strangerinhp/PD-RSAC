## Prepare Data Module

Goal: precompute and cache heavy data transformations (hex coordinates, trip tensors, episode filters) ahead of training to shrink CPU prep time.

### Proposed components
1. `cache_builder.py`
   - Loads hexagon list (`Wasserstein/data/nyc_full/hexagons.txt`) and computes:
     - `hex_ids` ordered list
     - `hex_to_idx` mapping
     - `hex_latlng` tensor `[num_hexes, 2]`
   - Saves cache to `preparedata/cache/hex_cache.npz`

2. `trip_preprocessor.py`
   - Reads `trips_processed.parquet`
   - Keeps needed columns (pickup_hex, dropoff_hex, fare, distance_km, pickup_time, trip_id, etc.)
   - Converts pickup/dropoff hex to indices using cache.
   - Computes additional fields (pickup_timestamp, travel_time_est, energy estimates).
   - Saves to chunked `.npz` or `.parquet` ready for GPU adapter.

3. `cli.py`
   - Provides commands (via argparse) to build caches:
     - `python -m preparedata.cli build --hex --trips`
   - Options for input/output directories, resolution, filters (date range, radius).

4. Integration plan
   - Modify `GPUProductionAdapter` / `FullSACAgent` to read from cache when available (future work).

### Next Steps
1. Implement cache builder + CLI skeleton.
2. Wire logs/progress.
3. Update training pipeline to consume cache (later task).

