#!/usr/bin/env python3
"""
Extract trip data for a specific day to CSV file.

Usage:
    python extract_day_to_csv.py --date 2009-01-15
    python extract_day_to_csv.py --date 2009-01-15 --input data/nyc_full/trips_processed.parquet --output day_2009-01-15.csv
    python extract_day_to_csv.py --date 2009-01-15 --columns trip_id,pickup_time,pickup_hex,dropoff_hex,fare


# Xem tất cả các ngày có sẵn
python extract_day_to_csv.py --list-dates

# Trích xuất dữ liệu ngày 2009-01-15
python extract_day_to_csv.py --date 2009-01-15

# Chỉ định input/output file
python extract_day_to_csv.py --date 2009-01-15 \
    --input Wasserstein/data/nyc_full/trips_processed.parquet \
    --output day_2009-01-15.csv

# Chỉ export một số columns
python extract_day_to_csv.py --date 2009-01-15 \
    --columns trip_id,pickup_time,pickup_hex,dropoff_hex,fare,distance_km

# Sample 10% dữ liệu
python extract_day_to_csv.py --date 2009-01-15 --sample 0.1
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract trip data for a specific day to CSV',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--date',
        type=str,
        required=True,
        help='Date to extract (YYYY-MM-DD format, e.g., 2009-01-15)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='Wasserstein/data/nyc_full/trips_processed.parquet',
        help='Input parquet file path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: day_YYYY-MM-DD.csv)'
    )
    
    parser.add_argument(
        '--columns',
        type=str,
        default=None,
        help='Comma-separated list of columns to export (default: all columns)'
    )
    
    parser.add_argument(
        '--sample',
        type=float,
        default=1.0,
        help='Sample ratio (0.0-1.0, default: 1.0 = all data)'
    )
    
    parser.add_argument(
        '--list-dates',
        action='store_true',
        help='List all available dates in the data file'
    )
    
    return parser.parse_args()


def list_available_dates(parquet_path: Path) -> None:
    """List all available dates in the parquet file."""
    print(f"Loading data from: {parquet_path}")
    
    if not parquet_path.exists():
        print(f"Error: File not found: {parquet_path}")
        return
    
    try:
        # Read parquet file
        print("Reading parquet file...")
        df = pd.read_parquet(parquet_path)
        
        # Parse pickup_time if needed
        if not pd.api.types.is_datetime64_any_dtype(df['pickup_time']):
            df['pickup_time'] = pd.to_datetime(df['pickup_time'])
        
        # Get unique dates
        dates = df['pickup_time'].dt.date.unique()
        dates = sorted(dates)
        
        print(f"\nFound {len(dates)} unique dates:")
        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"\nAvailable dates:")
        for i, date in enumerate(dates, 1):
            # Count trips for this date
            mask = df['pickup_time'].dt.date == date
            count = mask.sum()
            print(f"  {i:3d}. {date} ({count:,} trips)")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()


def extract_day_to_csv(
    date_str: str,
    input_path: str,
    output_path: str = None,
    columns: str = None,
    sample_ratio: float = 1.0
) -> None:
    """Extract trips for a specific day and save to CSV."""
    
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Parse date
    try:
        target_date = pd.to_datetime(date_str).date()
    except Exception as e:
        print(f"Error: Invalid date format '{date_str}'. Use YYYY-MM-DD format.")
        return
    
    # Default output filename
    if output_path is None:
        output_path = f"day_{date_str}.csv"
    
    output_file = Path(output_path)
    
    print("=" * 60)
    print("Extract Day Data to CSV")
    print("=" * 60)
    print(f"Date: {date_str} ({target_date})")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Sample ratio: {sample_ratio * 100:.0f}%")
    
    # Load data
    print("\nLoading parquet file...")
    try:
        df = pd.read_parquet(input_file)
        print(f"Loaded {len(df):,} total trips")
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Parse datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['pickup_time']):
        print("Parsing pickup_time...")
        df['pickup_time'] = pd.to_datetime(df['pickup_time'])
    
    # Filter by date
    print(f"Filtering for date {target_date}...")
    mask = df['pickup_time'].dt.date == target_date
    day_df = df[mask].copy()
    
    if len(day_df) == 0:
        print(f"\n❌ No trips found for date {target_date}")
        print("\nAvailable dates in the file:")
        available_dates = sorted(df['pickup_time'].dt.date.unique())
        for d in available_dates[:10]:
            count = (df['pickup_time'].dt.date == d).sum()
            print(f"  - {d} ({count:,} trips)")
        if len(available_dates) > 10:
            print(f"  ... and {len(available_dates) - 10} more dates")
        return
    
    print(f"Found {len(day_df):,} trips for {target_date}")
    
    # Sample if needed
    if sample_ratio < 1.0:
        n = int(len(day_df) * sample_ratio)
        day_df = day_df.sample(n=n, random_state=42)
        print(f"Sampled {len(day_df):,} trips ({sample_ratio * 100:.0f}%)")
    
    # Select columns if specified
    if columns:
        column_list = [c.strip() for c in columns.split(',')]
        # Check which columns exist
        available_columns = [c for c in column_list if c in day_df.columns]
        missing_columns = [c for c in column_list if c not in day_df.columns]
        
        if missing_columns:
            print(f"\n⚠️  Warning: Columns not found: {missing_columns}")
            print(f"Available columns: {list(day_df.columns)}")
        
        if available_columns:
            day_df = day_df[available_columns]
            print(f"Selected {len(available_columns)} columns: {', '.join(available_columns)}")
        else:
            print("❌ No valid columns specified. Exporting all columns.")
    
    # Sort by pickup_time
    day_df = day_df.sort_values('pickup_time').reset_index(drop=True)
    
    # Save to CSV
    print(f"\nSaving to CSV...")
    try:
        day_df.to_csv(output_file, index=False)
        print(f"✅ Successfully saved {len(day_df):,} trips to {output_file}")
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        print(f"Total trips: {len(day_df):,}")
        
        if 'fare' in day_df.columns:
            print(f"Fare - Min: ${day_df['fare'].min():.2f}, Max: ${day_df['fare'].max():.2f}, Avg: ${day_df['fare'].mean():.2f}")
        
        if 'distance_km' in day_df.columns:
            print(f"Distance - Min: {day_df['distance_km'].min():.2f} km, Max: {day_df['distance_km'].max():.2f} km, Avg: {day_df['distance_km'].mean():.2f} km")
        
        if 'pickup_time' in day_df.columns:
            time_range = day_df['pickup_time'].max() - day_df['pickup_time'].min()
            print(f"Time range: {day_df['pickup_time'].min()} to {day_df['pickup_time'].max()}")
            print(f"Duration: {time_range}")
        
        if 'pickup_hex' in day_df.columns:
            unique_hexes = len(set(day_df['pickup_hex'].unique()) | set(day_df['dropoff_hex'].unique()))
            print(f"Unique hexes: {unique_hexes}")
        
        print(f"\nFile size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_args()
    
    # List dates if requested
    if args.list_dates:
        list_available_dates(Path(args.input))
        return
    
    # Extract day data
    extract_day_to_csv(
        date_str=args.date,
        input_path=args.input,
        output_path=args.output,
        columns=args.columns,
        sample_ratio=args.sample
    )


if __name__ == '__main__':
    main()

