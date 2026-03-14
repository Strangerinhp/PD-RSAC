#!/usr/bin/env python3
"""
Calculate Maximum Possible Profit from Dataset

This script calculates the maximum possible profit (sum of all fares) from a dataset
for a specific day. This represents the theoretical upper bound if we could serve
100% of trips.

Usage:
    python3 calculate_max_profit.py \
        --real-data data/nyc_full/trips_processed.parquet \
        --date 2009-01-15 \
        --output results/max_profit.json
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate maximum possible profit from trip dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--real-data', type=str, required=True,
                        help='Path to trip data parquet file')
    parser.add_argument('--date', type=str, required=True,
                        help='Target date (format: YYYY-MM-DD, e.g., 2009-01-15)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: print to stdout)')
    parser.add_argument('--trip-sample', type=float, default=1.0,
                        help='Sample ratio for trip data (0.0-1.0). Default: 1.0 (all trips)')
    
    return parser.parse_args()


def calculate_max_profit(
    parquet_path: str,
    target_date: str,
    sample_ratio: float = 1.0
) -> Dict:
    """
    Calculate maximum possible profit from dataset.
    
    Args:
        parquet_path: Path to parquet file with trip data
        target_date: Target date (YYYY-MM-DD)
        sample_ratio: Sample ratio for trips (0.0-1.0)
    
    Returns:
        Dictionary with max profit statistics
    """
    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print(f"Total trips in dataset: {len(df):,}")
    
    # Parse date
    target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
    
    # Convert pickup_time to date
    if 'pickup_time' in df.columns:
        df['pickup_time'] = pd.to_datetime(df['pickup_time'])
        df['date'] = df['pickup_time'].dt.date
    else:
        raise ValueError("Column 'pickup_time' not found in dataset")
    
    # Filter by date
    filtered_df = df[df['date'] == target_date_obj].copy()
    print(f"Trips on {target_date}: {len(filtered_df):,}")
    
    if len(filtered_df) == 0:
        print(f"WARNING: No trips found for date {target_date}")
        return {
            'date': target_date,
            'total_trips': 0,
            'total_fare': 0.0,
            'avg_fare': 0.0,
            'min_fare': 0.0,
            'max_fare': 0.0,
            'median_fare': 0.0,
        }
    
    # Sample if needed
    if sample_ratio < 1.0:
        original_count = len(filtered_df)
        filtered_df = filtered_df.sample(frac=sample_ratio, random_state=42)
        print(f"Sampled {len(filtered_df):,} trips ({sample_ratio*100:.1f}% of {original_count:,})")
    
    # Calculate statistics
    total_fare = filtered_df['fare'].sum()
    avg_fare = filtered_df['fare'].mean()
    min_fare = filtered_df['fare'].min()
    max_fare = filtered_df['fare'].max()
    median_fare = filtered_df['fare'].median()
    
    # Hourly breakdown
    filtered_df['hour'] = filtered_df['pickup_time'].dt.hour
    hourly_stats = filtered_df.groupby('hour').agg({
        'fare': ['count', 'sum', 'mean']
    }).reset_index()
    hourly_stats.columns = ['hour', 'trip_count', 'total_fare', 'avg_fare']
    hourly_breakdown = hourly_stats.to_dict('records')
    
    result = {
        'date': target_date,
        'total_trips': int(len(filtered_df)),
        'total_fare': float(total_fare),
        'avg_fare': float(avg_fare),
        'min_fare': float(min_fare),
        'max_fare': float(max_fare),
        'median_fare': float(median_fare),
        'sample_ratio': sample_ratio,
        'hourly_breakdown': hourly_breakdown
    }
    
    return result


def main():
    args = parse_args()
    
    # Validate inputs
    if not Path(args.real_data).exists():
        raise FileNotFoundError(f"Data file not found: {args.real_data}")
    
    # Calculate max profit
    result = calculate_max_profit(
        parquet_path=args.real_data,
        target_date=args.date,
        sample_ratio=args.trip_sample
    )
    
    # Print summary
    print("\n" + "="*60)
    print("MAXIMUM POSSIBLE PROFIT SUMMARY")
    print("="*60)
    print(f"Date: {result['date']}")
    print(f"Total trips: {result['total_trips']:,}")
    print(f"Total fare (max possible revenue): ${result['total_fare']:,.2f}")
    print(f"Average fare: ${result['avg_fare']:.2f}")
    print(f"Median fare: ${result['median_fare']:.2f}")
    print(f"Min fare: ${result['min_fare']:.2f}")
    print(f"Max fare: ${result['max_fare']:.2f}")
    print("="*60)
    
    # Save to file if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    else:
        # Print JSON to stdout
        print("\n" + json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

