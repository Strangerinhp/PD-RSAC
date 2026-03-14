#!/usr/bin/env python3
"""
Benchmark script for GPU-accelerated EV Fleet RL training.

Measures performance metrics:
- Simulation throughput (steps/second)
- Training throughput (updates/second)
- GPU memory usage
- Scaling with fleet size
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import Config, ConfigLoader
from gpu_core.networks.sac import SACAgent
from gpu_core.features.replay_buffer import GPUReplayBuffer
from gpu_core.simulator.environment import GPUEnvironment
from gpu_core.spatial.grid import HexGrid
from gpu_core.utils.profiler import (
    GPUProfiler,
    ThroughputMeter,
    MemoryTracker,
    estimate_max_batch_size
)


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark GPU training performance')
    
    parser.add_argument('--benchmark', type=str, default='all',
                        choices=['all', 'simulation', 'training', 'memory', 'scaling'],
                        help='Type of benchmark to run')
    
    parser.add_argument('--num-vehicles', type=int, default=100,
                        help='Number of vehicles')
    parser.add_argument('--num-hexes', type=int, default=200,
                        help='Number of hexagons')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    
    parser.add_argument('--warmup-iterations', type=int, default=10,
                        help='Warmup iterations before timing')
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='Number of iterations to benchmark')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to benchmark on')
    
    return parser.parse_args()


def create_components(
    num_vehicles: int,
    num_hexes: int,
    batch_size: int,
    device: str
) -> Tuple[GPUEnvironment, SACAgent, GPUReplayBuffer]:
    """Create components for benchmarking."""
    config = Config()
    config.environment.num_vehicles = num_vehicles
    config.environment.num_hexes = num_hexes
    config.training.batch_size = batch_size
    
    # Initialize HexGrid with synthetic data for benchmarking
    hex_grid = HexGrid(device=device)
    fake_hex_ids = [f"hex_{i}" for i in range(num_hexes)]
    hex_grid._hex_ids = fake_hex_ids
    hex_grid._hex_to_idx = {h: i for i, h in enumerate(fake_hex_ids)}
    hex_grid._latitudes = torch.zeros(num_hexes, device=device)
    hex_grid._longitudes = torch.zeros(num_hexes, device=device)
    hex_grid._initialized = True
    hex_grid.distance_matrix._distances = torch.rand(num_hexes, num_hexes, device=device) * 10.0
    hex_grid.distance_matrix._num_hexes = num_hexes
    
    env = GPUEnvironment(config=config, hex_grid=hex_grid, device=device)
    
    state_dim = num_vehicles * 16 + num_hexes * 8 + 32
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=4,
        num_hexes=num_hexes,
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
        device=device
    ).to(device)
    
    replay_buffer = GPUReplayBuffer(
        capacity=100000,
        num_vehicles=num_vehicles,
        vehicle_feature_dim=16,
        num_hexes=num_hexes,
        hex_feature_dim=8,
        context_dim=32,
        device=device
    )
    
    return env, agent, replay_buffer


def benchmark_simulation(
    env: GPUEnvironment,
    warmup: int,
    iterations: int,
    device: str
) -> Dict:
    """Benchmark environment simulation speed."""
    print("\n" + "=" * 60)
    print("Simulation Benchmark")
    print("=" * 60)
    
    profiler = GPUProfiler(device=device)
    meter = ThroughputMeter()
    
    state = env.reset()
    
    for _ in range(warmup):
        action_type = torch.randint(0, 4, (env.num_vehicles,), device=device)
        reposition_target = torch.randint(0, env.num_hexes, (env.num_vehicles,), device=device)
        state, _, done, _ = env.step(action_type, reposition_target)
        if done.item():
            state = env.reset()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for i in range(iterations):
        with profiler.profile("step"):
            action_type = torch.randint(0, 4, (env.num_vehicles,), device=device)
            reposition_target = torch.randint(0, env.num_hexes, (env.num_vehicles,), device=device)
            state, reward, done, info = env.step(action_type, reposition_target)
            
            if done.item():
                state = env.reset()
        
        meter.record_step(env.num_vehicles)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start_time
    
    results = {
        'simulation_steps': iterations,
        'total_time_s': elapsed,
        'steps_per_second': iterations / elapsed,
        'vehicle_steps_per_second': (iterations * env.num_vehicles) / elapsed,
        'avg_step_time_ms': profiler.get_stats('step')['gpu_time_mean_ms']
    }
    
    print(f"Steps: {iterations}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Steps/second: {results['steps_per_second']:.1f}")
    print(f"Vehicle-steps/second: {results['vehicle_steps_per_second']:.1f}")
    print(f"Avg step time: {results['avg_step_time_ms']:.3f}ms")
    
    return results


def benchmark_training(
    agent: SACAgent,
    replay_buffer: GPUReplayBuffer,
    batch_size: int,
    warmup: int,
    iterations: int,
    device: str,
    num_vehicles: int,
    num_hexes: int
) -> Dict:
    """Benchmark training update speed."""
    print("\n" + "=" * 60)
    print("Training Benchmark")
    print("=" * 60)
    
    profiler = GPUProfiler(device=device)
    
    # Calculate dimensions from config
    vehicle_feature_dim = 16
    hex_feature_dim = 8
    context_dim = 32
    
    # Fill replay buffer with synthetic data
    for _ in range(batch_size * 10):
        state = {
            'vehicle': torch.randn(num_vehicles, vehicle_feature_dim, device=device),
            'hex': torch.randn(num_hexes, hex_feature_dim, device=device),
            'context': torch.randn(context_dim, device=device)
        }
        # Actions: [num_vehicles, 2] - action_type (0-3) and reposition_target (hex_idx)
        action = torch.zeros(num_vehicles, 2, dtype=torch.long, device=device)
        action[:, 0] = torch.randint(0, 4, (num_vehicles,), device=device)
        action[:, 1] = torch.randint(0, num_hexes, (num_vehicles,), device=device)
        
        reward = torch.randn(1, device=device).item()
        next_state = {
            'vehicle': torch.randn(num_vehicles, vehicle_feature_dim, device=device),
            'hex': torch.randn(num_hexes, hex_feature_dim, device=device),
            'context': torch.randn(context_dim, device=device)
        }
        done = False
        
        replay_buffer.push(state, action, reward, next_state, done)
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    
    # Compute state dimension
    state_dim = num_vehicles * vehicle_feature_dim + num_hexes * hex_feature_dim + context_dim
    
    for _ in range(warmup):
        batch = replay_buffer.sample(batch_size)
        # Flatten states for the agent
        vehicle_flat = batch.states['vehicle'].view(batch_size, -1)
        hex_flat = batch.states['hex'].view(batch_size, -1)
        context_flat = batch.states['context']
        states_flat = torch.cat([vehicle_flat, hex_flat, context_flat], dim=-1)
        
        # Get action type for critic (first column, one-hot encoded as action_dim=4)
        action_types = batch.actions[:, :, 0].view(batch_size, num_vehicles)  # [batch, num_vehicles]
        # Use mean action type across vehicles for simple critic input
        mean_action = action_types.float().mean(dim=1).long()  # [batch]
        
        q_values = agent.critic(states_flat, mean_action)
        loss = q_values[0].mean() + q_values[1].mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    forward_times = []
    backward_times = []
    
    for i in range(iterations):
        batch = replay_buffer.sample(batch_size)
        # Flatten states for the agent
        vehicle_flat = batch.states['vehicle'].view(batch_size, -1)
        hex_flat = batch.states['hex'].view(batch_size, -1)
        context_flat = batch.states['context']
        states_flat = torch.cat([vehicle_flat, hex_flat, context_flat], dim=-1)
        
        # Get action type for critic
        action_types = batch.actions[:, :, 0].view(batch_size, num_vehicles)
        mean_action = action_types.float().mean(dim=1).long()
        
        # Flatten next states
        next_vehicle_flat = batch.next_states['vehicle'].view(batch_size, -1)
        next_hex_flat = batch.next_states['hex'].view(batch_size, -1)
        next_context_flat = batch.next_states['context']
        next_states_flat = torch.cat([next_vehicle_flat, next_hex_flat, next_context_flat], dim=-1)
        
        rewards = batch.rewards
        dones = batch.dones.float()  # Convert to float for math operations
        
        with profiler.profile("forward"):
            q1, q2 = agent.critic(states_flat, mean_action)
            
            with torch.no_grad():
                # Actor returns (action_type, action_log_prob, reposition_target, reposition_log_prob)
                next_action_type, next_log_prob, _, _ = agent.actor(next_states_flat)
                next_q1, next_q2 = agent.critic_target(next_states_flat, next_action_type)
                next_q = torch.min(next_q1, next_q2)
                target_q = rewards + 0.99 * (1 - dones) * next_q
            
            critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
        
        with profiler.profile("backward"):
            optimizer.zero_grad()
            critic_loss.backward()
            optimizer.step()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start_time
    
    forward_stats = profiler.get_stats('forward')
    backward_stats = profiler.get_stats('backward')
    
    results = {
        'training_updates': iterations,
        'batch_size': batch_size,
        'total_time_s': elapsed,
        'updates_per_second': iterations / elapsed,
        'samples_per_second': (iterations * batch_size) / elapsed,
        'avg_forward_time_ms': forward_stats['gpu_time_mean_ms'],
        'avg_backward_time_ms': backward_stats['gpu_time_mean_ms']
    }
    
    print(f"Updates: {iterations}")
    print(f"Batch size: {batch_size}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Updates/second: {results['updates_per_second']:.1f}")
    print(f"Samples/second: {results['samples_per_second']:.1f}")
    print(f"Avg forward time: {results['avg_forward_time_ms']:.3f}ms")
    print(f"Avg backward time: {results['avg_backward_time_ms']:.3f}ms")
    
    return results


def benchmark_memory(
    num_vehicles: int,
    num_hexes: int,
    batch_sizes: List[int],
    device: str
) -> Dict:
    """Benchmark GPU memory usage."""
    print("\n" + "=" * 60)
    print("Memory Benchmark")
    print("=" * 60)
    
    vehicle_feature_dim = 16
    hex_feature_dim = 8
    context_dim = 32
    
    results = {'batch_sizes': {}}
    
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            env, agent, replay_buffer = create_components(
                num_vehicles, num_hexes, batch_size, device
            )
            
            for _ in range(batch_size * 2):
                state = {
                    'vehicle': torch.randn(num_vehicles, vehicle_feature_dim, device=device),
                    'hex': torch.randn(num_hexes, hex_feature_dim, device=device),
                    'context': torch.randn(context_dim, device=device)
                }
                action = torch.zeros(num_vehicles, 2, dtype=torch.long, device=device)
                action[:, 0] = torch.randint(0, 4, (num_vehicles,), device=device)
                action[:, 1] = torch.randint(0, num_hexes, (num_vehicles,), device=device)
                reward = torch.randn(1, device=device).item()
                next_state = {
                    'vehicle': torch.randn(num_vehicles, vehicle_feature_dim, device=device),
                    'hex': torch.randn(num_hexes, hex_feature_dim, device=device),
                    'context': torch.randn(context_dim, device=device)
                }
                done = False
                replay_buffer.push(state, action, reward, next_state, done)
            
            batch = replay_buffer.sample(batch_size)
            # Flatten states
            vehicle_flat = batch.states['vehicle'].view(batch_size, -1)
            hex_flat = batch.states['hex'].view(batch_size, -1)
            context_flat = batch.states['context']
            states_flat = torch.cat([vehicle_flat, hex_flat, context_flat], dim=-1)
            
            action_types = batch.actions[:, :, 0].view(batch_size, num_vehicles)
            mean_action = action_types.float().mean(dim=1).long()
            
            _ = agent.critic(states_flat, mean_action)
            
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            
            results['batch_sizes'][batch_size] = {
                'allocated_mb': allocated,
                'max_allocated_mb': max_allocated,
                'reserved_mb': reserved
            }
            
            print(f"Batch {batch_size:4d}: {allocated:.1f} MB allocated, "
                  f"{max_allocated:.1f} MB peak, {reserved:.1f} MB reserved")
            
            del env, agent, replay_buffer
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch {batch_size:4d}: OOM")
                results['batch_sizes'][batch_size] = {'oom': True}
            else:
                raise
    
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
    results['total_gpu_memory_mb'] = total_memory
    print(f"\nTotal GPU memory: {total_memory:.1f} MB")
    
    return results


def benchmark_scaling(
    vehicle_counts: List[int],
    num_hexes: int,
    batch_size: int,
    iterations: int,
    device: str
) -> Dict:
    """Benchmark scaling with fleet size."""
    print("\n" + "=" * 60)
    print("Scaling Benchmark")
    print("=" * 60)
    
    results = {'vehicle_counts': {}}
    
    for num_vehicles in vehicle_counts:
        print(f"\nTesting {num_vehicles} vehicles...")
        
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            env, agent, replay_buffer = create_components(
                num_vehicles, num_hexes, batch_size, device
            )
            
            state = env.reset()
            
            start = time.perf_counter()
            for _ in range(iterations):
                action_type = torch.randint(0, 4, (num_vehicles,), device=device)
                reposition_target = torch.randint(0, num_hexes, (num_vehicles,), device=device)
                state, _, done, _ = env.step(action_type, reposition_target)
                if done.item():
                    state = env.reset()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start
            
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
            
            results['vehicle_counts'][num_vehicles] = {
                'steps_per_second': iterations / elapsed,
                'vehicle_steps_per_second': (iterations * num_vehicles) / elapsed,
                'peak_memory_mb': peak_memory
            }
            
            print(f"  Steps/s: {iterations / elapsed:.1f}")
            print(f"  Vehicle-steps/s: {(iterations * num_vehicles) / elapsed:.1f}")
            print(f"  Peak memory: {peak_memory:.1f} MB")
            
            del env, agent, replay_buffer
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM at {num_vehicles} vehicles")
                results['vehicle_counts'][num_vehicles] = {'oom': True}
                break
            else:
                raise
    
    return results


def main():
    args = parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("GPU Fleet RL Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    print(f"Vehicles: {args.num_vehicles}")
    print(f"Hexes: {args.num_hexes}")
    print(f"Batch size: {args.batch_size}")
    
    results = {
        'device': device,
        'num_vehicles': args.num_vehicles,
        'num_hexes': args.num_hexes,
        'batch_size': args.batch_size
    }
    
    if args.benchmark in ['all', 'simulation']:
        env, agent, buffer = create_components(
            args.num_vehicles, args.num_hexes, args.batch_size, device
        )
        results['simulation'] = benchmark_simulation(
            env, args.warmup_iterations, args.num_iterations, device
        )
        del env, agent, buffer
    
    if args.benchmark in ['all', 'training']:
        env, agent, buffer = create_components(
            args.num_vehicles, args.num_hexes, args.batch_size, device
        )
        results['training'] = benchmark_training(
            agent, buffer, args.batch_size,
            args.warmup_iterations, args.num_iterations, device,
            args.num_vehicles, args.num_hexes
        )
        del env, agent, buffer
    
    if args.benchmark in ['all', 'memory']:
        results['memory'] = benchmark_memory(
            args.num_vehicles, args.num_hexes,
            [32, 64, 128, 256, 512, 1024],
            device
        )
    
    if args.benchmark in ['all', 'scaling']:
        results['scaling'] = benchmark_scaling(
            [100, 200, 500, 1000, 2000],
            args.num_hexes,
            args.batch_size,
            args.num_iterations // 2,
            device
        )
    
    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
