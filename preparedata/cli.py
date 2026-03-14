"""
# NYC Real (1 tháng data)
python3 -m preparedata.cli hex \
  --data-dir Wasserstein/data/nyc_real \
  --output-dir preparedata/cache \
  --min-trips 100

# NYC Full (toàn bộ data - dataset bạn đang dùng)
python3 -m preparedata.cli hex \
  --data-dir Wasserstein/data/nyc_full \
  --output-dir preparedata/cache \
  --min-trips 100

--min-trips <số>      # Ngưỡng trips tối thiểu (default: 100)
                      # Thử các giá trị: 50, 100, 200, 500
                      # Set 0 để disable filtering

--data-dir <path>     # Thư mục chứa hexagons.txt và trips_processed.parquet

--output-dir <path>   # Nơi lưu hex_cache.npz
                      # Training sẽ load cache từ đây

Để thử resolution 9:
# Bước 1: Regenerate hex grid với resolution 9
# (Cần reprocess NYC data với H3 resolution 9)

# Bước 2: Build cache với resolution 9 + frequency filter
python3 -m preparedata.cli hex \
  --data-dir Wasserstein/data/nyc_full \
  --output-dir preparedata/cache \
  --resolution 9 \
  --min-trips 100
  
"""     
import argparse
import logging
from pathlib import Path

from .cache_builder import HexCacheBuilder
from .trip_preprocessor import TripPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
LOGGER = logging.getLogger("preparedata.cli")


def build_hex_cache(args: argparse.Namespace):
    builder = HexCacheBuilder(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        hex_file=args.hex_file,
        resolution=args.resolution,
    )
    cache = builder.build_cache(
        use_concave_hull=args.use_concave_hull,
        alpha=args.alpha,
        buffer_distance=args.buffer,
        min_trip_frequency=args.min_trips,
    )
    builder.save_cache(cache)
    LOGGER.info("Hex cache built with %s hexes", cache.size)


def preprocess_trips(args: argparse.Namespace):
    preprocessor = TripPreprocessor(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        parquet_file=args.trips_file,
        chunk_size_rows=args.chunk_size,
        start_date=args.start_date,
        end_date=args.end_date,
        percentage=args.percentage,
    )
    summaries = preprocessor.process()
    total_rows = sum(s.num_rows for s in summaries)
    LOGGER.info("Wrote %s rows across %s chunks", total_rows, len(summaries))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare data caches for GPU training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hex_parser = subparsers.add_parser("hex", help="Build hexagon cache.")
    hex_parser.add_argument("--data-dir", type=str, default="Wasserstein/data/nyc_full")
    hex_parser.add_argument("--output-dir", type=str, default="preparedata/cache")
    hex_parser.add_argument("--hex-file", type=str, default="hexagons.txt")
    hex_parser.add_argument("--resolution", type=int, default=None)
    hex_parser.add_argument(
        "--min-trips", type=int, default=100,
        help="Minimum trips per hex to keep (default: 100, set 0 to disable)"
    )
    hex_parser.add_argument(
        "--use-concave-hull", action="store_true", default=False,
        help="Filter hexes to concave hull of data points (default: False)"
    )
    hex_parser.add_argument(
        "--no-concave-hull", dest="use_concave_hull", action="store_false",
        help="Disable concave hull filtering"
    )
    hex_parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Alpha parameter for concave hull (smaller=more concave, default: 0.05)"
    )
    hex_parser.add_argument(
        "--buffer", type=float, default=0.005,
        help="Buffer distance in degrees around hull (~0.01=1km, default: 0.005)"
    )
    hex_parser.set_defaults(func=build_hex_cache)

    trip_parser = subparsers.add_parser("trips", help="Preprocess trip parquet.")
    trip_parser.add_argument("--data-dir", type=str, default="Wasserstein/data/nyc_full")
    trip_parser.add_argument("--output-dir", type=str, default="preparedata/trips_cache")
    trip_parser.add_argument("--cache-dir", type=str, default="preparedata/cache")
    trip_parser.add_argument("--trips-file", type=str, default="trips_processed.parquet")
    trip_parser.add_argument("--chunk-size", type=int, default=500_000)
    trip_parser.add_argument("--start-date", type=str, default=None)
    trip_parser.add_argument("--end-date", type=str, default=None)
    trip_parser.add_argument("--percentage", type=float, default=1.0)
    trip_parser.set_defaults(func=preprocess_trips)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

