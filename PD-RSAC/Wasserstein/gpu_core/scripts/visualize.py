#!/usr/bin/env python3
"""
Visualization script for EV Fleet RL training results.

Usage:
    python visualize.py --metrics training_metrics.json --output plots/
    python visualize.py --benchmark benchmark_results.json
"""

import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.utils.visualizer import (
    TrainingMetrics,
    TrainingVisualizer,
    PerformanceVisualizer,
    create_training_report
)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize training results')
    
    parser.add_argument('--metrics', type=str, default=None,
                        help='Path to training metrics JSON file')
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Path to benchmark results JSON file')
    parser.add_argument('--output', type=str, default='plots',
                        help='Output directory for plots')
    
    parser.add_argument('--window', type=int, default=100,
                        help='Smoothing window for reward plots')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output format for plots')
    
    parser.add_argument('--report', action='store_true',
                        help='Generate full training report')
    
    return parser.parse_args()


def visualize_training(args):
    """Generate training visualizations."""
    print(f"Loading metrics from: {args.metrics}")
    metrics = TrainingMetrics.load(args.metrics)
    
    print(f"Loaded {len(metrics.episodes)} episodes")
    
    viz = TrainingVisualizer(metrics, output_dir=args.output)
    
    viz.plot_rewards(window=args.window)
    
    if metrics.actor_losses or metrics.critic_losses:
        viz.plot_losses()
    
    if metrics.alpha_values or metrics.entropies:
        viz.plot_alpha_entropy()
    
    if metrics.trips_served or metrics.avg_soc:
        viz.plot_fleet_metrics()
    
    if args.report:
        report_dir = Path(args.output) / 'report'
        create_training_report(metrics, str(report_dir))


def visualize_benchmark(args):
    """Generate benchmark visualizations."""
    print(f"Loading benchmark results from: {args.benchmark}")
    
    with open(args.benchmark, 'r') as f:
        results = json.load(f)
    
    viz = PerformanceVisualizer(output_dir=args.output)
    
    if 'scaling' in results and 'vehicle_counts' in results['scaling']:
        vehicle_data = results['scaling']['vehicle_counts']
        counts = []
        throughputs = []
        
        for count_str, data in sorted(vehicle_data.items(), key=lambda x: int(x[0])):
            if 'oom' not in data:
                counts.append(int(count_str))
                throughputs.append(data['steps_per_second'])
        
        if counts:
            viz.plot_scaling(counts, throughputs)
            print(f"Generated scaling plot")
    
    if 'memory' in results and 'batch_sizes' in results['memory']:
        batch_data = results['memory']['batch_sizes']
        sizes = []
        memories = []
        
        for size_str, data in sorted(batch_data.items(), key=lambda x: int(x[0])):
            if 'oom' not in data:
                sizes.append(int(size_str))
                memories.append(data['max_allocated_mb'])
        
        if sizes:
            viz.plot_memory_scaling(sizes, memories)
            print(f"Generated memory scaling plot")
    
    print(f"Plots saved to: {args.output}")


def main():
    args = parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.metrics:
        visualize_training(args)
    
    if args.benchmark:
        visualize_benchmark(args)
    
    if not args.metrics and not args.benchmark:
        print("Please specify --metrics or --benchmark")
        return 1
    
    print("Visualization complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
