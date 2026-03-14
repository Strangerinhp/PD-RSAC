#!/usr/bin/env python3
"""
GPU-accelerated EV Fleet RL Training Script.

Usage:
    python train.py --config config.yaml
    python train.py --num-vehicles 1000 --num-hexes 1300 --gpus 0,1
    python train.py --real-data ./data/nyc_real/trips_processed.parquet
    python train.py --help
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List
from collections import deque
import time
from collections import defaultdict
import csv

# Timing profiler
_timing_stats = defaultdict(list)

def _time_section(name):
    """Context manager for timing code sections."""
    class TimingContext:
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start = time.time()
            return self
        def __exit__(self, *args):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - self.start
            _timing_stats[name].append(elapsed)
    return TimingContext(name)



import torch
import torch.distributed as dist

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import ConfigLoader, Config, EnvironmentConfig, TrainingConfig, DistributedConfig
from gpu_core.networks.sac import SACAgent
from gpu_core.features.replay_buffer import GPUReplayBuffer
from gpu_core.features.builder import FeatureBuilder
from gpu_core.training.trainer import SACTrainer
from gpu_core.training.enhanced_trainer import EnhancedSACTrainer, EnhancedTrainingConfig
from gpu_core.training.enhanced_collector import EnhancedEpisodeCollector
from gpu_core.training.distributed import (
    DistributedTrainer,
    MixedPrecisionTrainer,
    setup_distributed,
    cleanup_distributed,
    spawn_distributed_training
)
from gpu_core.training.episode_collector import EpisodeCollector, BatchedEpisodeCollector
from gpu_core.simulator.environment import GPUEnvironment, GPUEnvironmentV2, BatchedGPUEnvironment
from gpu_core.spatial.grid import HexGrid
from gpu_core.spatial.distance import DistanceMatrix
from gpu_core.data.real_trip_loader import RealTripLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train EV Fleet RL Agent with GPU acceleration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    
    parser.add_argument('--num-vehicles', type=int, default=None,
                        help='Number of vehicles in fleet')
    parser.add_argument('--num-hexes', type=int, default=None,
                        help='Number of hexagons in grid')
    parser.add_argument('--compact-state', action='store_true',
                        help='Use compact state (aggregate vehicle features per hex) for faster training')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Total training episodes')
    parser.add_argument('--episode-duration-hours', type=float, default=None,
                        help='Duration of each episode in hours (e.g., 24.0 for full day, 10.0 for 10 hours)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size')
    
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs (e.g., "0,1,2")')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training (DDP - advanced)')
    parser.add_argument('--data-parallel', action='store_true',
                        help='Enable DataParallel for multi-GPU (simpler than DDP)')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training (AMP)')
    
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--start-episode', type=int, default=None,
                        help='Override start episode (use when checkpoint has wrong episode counter)')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Save checkpoint every N episodes')
    
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log metrics every N episodes')
    parser.add_argument('--eval-interval', type=int, default=50,
                        help='Evaluate agent every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Number of episodes for evaluation')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='Number of random steps to warmup replay buffer')
    parser.add_argument('--update-ratio', type=int, default=1,
                        help='Number of steps per training update (higher = faster but less learning)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    # Logging options
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard logging')
    parser.add_argument('--tensorboard-dir', type=str, default='runs',
                        help='TensorBoard log directory')
    parser.add_argument('--log-loss-interval', type=int, default=100,
                        help='Log loss every N training updates')
    
    # Real data options
    parser.add_argument('--real-data', type=str, default=None,
                        help='Path to real trip data parquet file (e.g., ./data/nyc_real/trips_processed.parquet)')
    parser.add_argument('--trip-sample', type=float, default=None,
                        help='Sample ratio for trip data (0.0-1.0). Use 0.1 for 10%%, 0.5 for 50%%, 1.0 for 100%%')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Filter trips from this date (format: YYYY-MM-DD, e.g., 2009-01-15)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Filter trips until this date inclusive (format: YYYY-MM-DD, e.g., 2009-01-20)')
    parser.add_argument('--target-h3-resolution', type=int, default=None,
                        help='Coarsen hex grid to target H3 resolution (e.g., 7, 8). Lower = larger hexes.')
    parser.add_argument('--max-hex-count', type=int, default=None,
                        help='Limit number of hexes by keeping most frequent (e.g., 5000)')
    
    # Paper-compliant training options
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use greedy (argmax) action selection instead of stochastic sampling')
    parser.add_argument('--milp', action='store_true', default=False,
                        help='Enable exact MILP assignment via Gurobi (IEEE_T_IV.pdf formulation)')
    parser.add_argument('--semi-mdp', action='store_true', default=True,  # CHANGED: Enable by default per paper
                        help='Enable Semi-MDP duration discounting (paper Eq. 12-13)')
    parser.add_argument('--no-semi-mdp', dest='semi_mdp', action='store_false',
                        help='Disable Semi-MDP discounting')
    parser.add_argument('--wdro', action='store_true', default=True,  # CHANGED: Enable by default per paper
                        help='Enable WDRO robust targets (paper Section 4)')
    parser.add_argument('--no-wdro', dest='wdro', action='store_false',
                        help='Disable WDRO robust targets')
    parser.add_argument('--wdro-rho', type=float, default=0.3,
                        help='WDRO ambiguity radius (default: 0.3)')
    parser.add_argument('--temperature-annealing', action='store_true', default=True,
                        help='Enable temperature annealing (paper Eq. 16)')
    parser.add_argument('--initial-temperature', type=float, default=1.0,
                        help='Initial softmax temperature')
    parser.add_argument('--final-temperature', type=float, default=0.1,
                        help='Final softmax temperature')
    parser.add_argument('--temperature-decay-episodes', type=int, default=200,
                        help='Episodes over which temperature anneals from initial to final (default: 200; was hardcoded 500)')
    parser.add_argument('--env-v2', action='store_true', default=False,
                        help='Use GPUEnvironmentV2 (refactored modular environment)')
    
    return parser.parse_args()


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(args) -> torch.device:
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
        return torch.device(f'cuda:{gpu_ids[0]}')
    elif torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def create_config(args) -> Config:
    if args.config:
        config = ConfigLoader.from_yaml(args.config)
    else:
        config = Config()
    
    if args.num_vehicles is not None:
        config.environment.num_vehicles = args.num_vehicles
    if args.num_hexes is not None:
        config.environment.num_hexes = args.num_hexes
    if args.compact_state:
        config.environment.compact_state = True
    if args.episodes is not None:
        config.training.total_episodes = args.episodes
    if args.episode_duration_hours is not None:
        config.episode.duration_hours = args.episode_duration_hours
        print(f"[Episode] Duration set to {args.episode_duration_hours} hours ({config.episode.steps_per_episode} steps @ {config.episode.step_duration_minutes} min/step)")
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    if args.gpus:
        config.distributed.gpus = [int(x) for x in args.gpus.split(',')]
    if args.distributed:
        config.distributed.enabled = True
        config.distributed.world_size = len(config.distributed.gpus) if config.distributed.gpus else torch.cuda.device_count()
    if args.mixed_precision:
        config.training.mixed_precision = True
    
    config.checkpoint.checkpoint_dir = args.checkpoint_dir
    config.checkpoint.save_interval = args.save_interval
    config.logging.log_interval = args.log_interval
    
    if args.warmup_steps is not None:
        config.training.warmup_steps = args.warmup_steps
    
    return config


def create_agent(config: Config, device: torch.device, env=None) -> SACAgent:
    env_config = config.environment
    entropy_config = config.entropy
    
    # Get dimensions from environment if available, otherwise use defaults
    if env is not None:
        vehicle_feature_dim = env._vehicle_feature_dim
        hex_feature_dim = env._hex_feature_dim
        context_dim = env._context_dim
    else:
        # Default V2 dimensions
        vehicle_feature_dim = 16  # FIX: Match feature builder output (was 14)
        hex_feature_dim = 5
        context_dim = 9
    
    # State dimension: vehicle features + hex features + context
    state_dim = (
        env_config.num_vehicles * vehicle_feature_dim +
        env_config.num_hexes * hex_feature_dim +
        context_dim
    )
    
    print(f"[State Dim] Vehicle: {env_config.num_vehicles}x{vehicle_feature_dim}, "
          f"Hex: {env_config.num_hexes}x{hex_feature_dim}, Context: {context_dim}, Total: {state_dim}")
    
    # Calculate target entropy from config
    action_dim = 3  # SERVE=0, CHARGE=1, REPOSITION=2 (IDLE removed)
    target_entropy = -entropy_config.target_entropy_ratio * torch.log(torch.tensor(float(action_dim))).item()
    
    # GCN-based actor (paper Section 5.1.2)
    # Use config network dimensions if available, otherwise paper defaults
    actor_hidden_dims = config.network.actor_hidden_dims if hasattr(config, 'network') and hasattr(config.network, 'actor_hidden_dims') else [128, 128]
    critic_hidden_dims = config.network.critic_hidden_dims if hasattr(config, 'network') and hasattr(config.network, 'critic_hidden_dims') else [256, 256]
    dropout_rate = config.network.dropout if hasattr(config, 'network') and hasattr(config.network, 'dropout') else 0.1
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_hexes=env_config.num_hexes,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        gamma=config.training.gamma,
        tau=config.training.tau,
        alpha=entropy_config.initial_alpha,
        auto_alpha=entropy_config.auto_alpha,
        target_entropy=target_entropy,
        lr_actor=config.training.learning_rate.actor,
        lr_critic=config.training.learning_rate.critic,
        lr_alpha=config.training.learning_rate.alpha,
        dropout=dropout_rate,
        device=str(device),
        num_vehicles=env_config.num_vehicles,
        # GCN feature dimensions
        vehicle_feature_dim=vehicle_feature_dim,
        hex_feature_dim=hex_feature_dim,
        context_dim=context_dim,
        # Configurable limits from config (no hardcoding!)
        max_trips=env_config.max_trips_per_step,
        num_stations=env_config.num_stations,
        max_serve=env_config.max_serve_per_step,
        max_charge=env_config.max_charge_per_step,
        min_alpha=getattr(entropy_config, 'min_alpha', 0.05),
        max_alpha=getattr(entropy_config, 'max_alpha', 1.0),
        repos_aux_weight=getattr(config.training, 'repos_aux_weight', 0.1),
    )
    
    return agent.to(device)


def create_replay_buffer(config: Config, device: torch.device, env=None) -> GPUReplayBuffer:
    env_config = config.environment
    buffer_config = config.replay_buffer
    
    # Get dimensions from environment if available, otherwise use defaults
    if env is not None:
        vehicle_feature_dim = env._vehicle_feature_dim
        hex_feature_dim = env._hex_feature_dim
        context_dim = env._context_dim
    else:
        # Default V1 dimensions
        vehicle_feature_dim = 16
        hex_feature_dim = 8
        context_dim = 32
    
    # Auto-calculate buffer capacity based on available GPU memory
    # Each transition needs: vehicle_states + hex_states + context + actions + rewards + dones
    # Estimate: ~(num_vehicles * vdim + num_hexes * hdim + cdim) * 4 bytes * 2 (state + next_state)
    bytes_per_transition = (env_config.num_vehicles * vehicle_feature_dim + 
                            env_config.num_hexes * hex_feature_dim + context_dim + 
                            env_config.num_vehicles * 2 + 1 + 1) * 4 * 2
    
    # Dynamically detect GPU memory instead of hardcoding 4GB
    # Use 50% of total VRAM for replay buffer, leave rest for model + environment
    try:
        gpu_index = 0
        if isinstance(device, torch.device) and device.index is not None:
            gpu_index = device.index
        elif isinstance(device, str) and 'cuda:' in device:
            gpu_index = int(device.split(':')[1])
        total_vram = torch.cuda.get_device_properties(gpu_index).total_memory
        # Reserve 50% for replay buffer
        max_buffer_memory = int(total_vram * 0.50)
        vram_gb = total_vram / (1024 ** 3)
        buffer_gb = max_buffer_memory / (1024 ** 3)
        print(f"GPU VRAM detected: {vram_gb:.1f} GB → allocating {buffer_gb:.1f} GB for replay buffer")
    except Exception as e:
        # Fallback to 4GB if GPU detection fails
        max_buffer_memory = 4 * 1024 * 1024 * 1024
        print(f"GPU VRAM detection failed ({e}), using 4GB fallback for replay buffer")
    
    max_capacity = max_buffer_memory // bytes_per_transition

    # Replay buffer capacity should respect config, only bounded by max memory
    capacity = min(buffer_config.capacity, max_capacity)
    
    print(f"Replay buffer capacity: {capacity:,} (auto-adjusted for {env_config.num_vehicles} vehicles to fit in memory)")
    
    return GPUReplayBuffer(
        capacity=capacity,
        num_vehicles=env_config.num_vehicles,
        vehicle_feature_dim=vehicle_feature_dim,
        num_hexes=env_config.num_hexes,
        hex_feature_dim=hex_feature_dim,
        context_dim=context_dim,
        prioritized=buffer_config.prioritized,
        alpha=buffer_config.alpha,
        beta_start=buffer_config.beta_start,
        beta_end=buffer_config.beta_end,
        device=str(device)
    )


def create_trainer(
    agent: SACAgent,
    replay_buffer: GPUReplayBuffer,
    config: Config,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1
):
    training_config = config.training
    
    if world_size > 1:
        if config.training.mixed_precision:
            return MixedPrecisionTrainer(
                agent=agent,
                replay_buffer=replay_buffer,
                training_config=training_config,
                distributed_config=config.distributed,
                rank=rank,
                world_size=world_size,
                checkpoint_config=config.checkpoint,
                logging_config=config.logging,
                use_amp=True
                # device is set by DistributedTrainer based on rank
            )
        else:
            return DistributedTrainer(
                agent=agent,
                replay_buffer=replay_buffer,
                training_config=training_config,
                distributed_config=config.distributed,
                rank=rank,
                world_size=world_size,
                checkpoint_config=config.checkpoint,
                logging_config=config.logging
                # device is set by DistributedTrainer based on rank
            )
    else:
        return SACTrainer(
            agent=agent,
            replay_buffer=replay_buffer,
            training_config=training_config,
            checkpoint_config=config.checkpoint,
            logging_config=config.logging,
            device=str(device)
        )


def create_environment(config: Config, device: torch.device, args, trip_loader: Optional[RealTripLoader] = None) -> GPUEnvironment:
    """Create GPU environment for training.
    
    Args:
        config: Configuration object
        device: Torch device
        args: Command line arguments
        trip_loader: Optional real trip data loader
        
    Returns:
        GPUEnvironment instance
    """
    num_hexes = config.environment.num_hexes
    device_str = str(device)
    hex_grid = HexGrid(device=device_str)
    
    if trip_loader and trip_loader.is_loaded:
        hex_ids = trip_loader.hex_ids
        lats, lons = trip_loader.get_hex_coordinates()
        num_hexes = len(hex_ids)
        config.environment.num_hexes = num_hexes
        hex_grid._hex_ids = hex_ids
        hex_grid._hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
        hex_grid._latitudes = lats.to(device)
        hex_grid._longitudes = lons.to(device)
        hex_grid._initialized = True
        hex_grid.distance_matrix.compute(hex_grid._latitudes, hex_grid._longitudes, hex_ids=hex_ids)
    else:
        # Fallback: synthetic grid covering ~20km^2 of NYC
        grid_size = int(num_hexes ** 0.5) + 1
        hex_spacing_km = 20.0 / max(grid_size, 1)
        fake_hex_ids = [f"hex_{i}" for i in range(num_hexes)]
        hex_grid._hex_ids = fake_hex_ids
        hex_grid._hex_to_idx = {h: i for i, h in enumerate(fake_hex_ids)}
        base_lat, base_lon = 40.7128, -74.0060
        lat_per_km, lon_per_km = 0.009, 0.012
        indices = torch.arange(num_hexes, device=device)
        rows = indices // grid_size
        cols = indices % grid_size
        lats = base_lat + rows.float() * hex_spacing_km * lat_per_km
        lons = base_lon + cols.float() * hex_spacing_km * lon_per_km
        hex_grid._latitudes = lats
        hex_grid._longitudes = lons
        hex_grid._initialized = True
        lat_diff = lats.unsqueeze(1) - lats.unsqueeze(0)
        lon_diff = lons.unsqueeze(1) - lons.unsqueeze(0)
        lat_km = lat_diff / lat_per_km
        lon_km = lon_diff / lon_per_km
        distances = torch.sqrt(lat_km**2 + lon_km**2)
        distances = distances * (0.8 + 0.4 * torch.rand_like(distances))
        distances.fill_diagonal_(0)
        hex_grid.distance_matrix._distances = distances
        hex_grid.distance_matrix._num_hexes = num_hexes
    
    # Choose environment version based on flag
    if args.env_v2:
        print("[Environment] Using GPUEnvironmentV2 (modular refactored version)")
        env = GPUEnvironmentV2(
            config=config,
            hex_grid=hex_grid,
            trip_loader=trip_loader,
            device=str(device)
        )
    else:
        env = GPUEnvironment(
            config=config,
            hex_grid=hex_grid,
            trip_loader=trip_loader,  # Pass real trip loader if available
            device=str(device)
        )
    
    return env


def warmup_replay_buffer(
    env: GPUEnvironment,
    replay_buffer: GPUReplayBuffer,
    num_steps: int,
    device: torch.device
):
    """Fill replay buffer with random transitions."""
    print(f"Warming up replay buffer with {num_steps} random transitions...")
    
    state = env.reset()
    
    for step in range(num_steps):
        # SERVE=0, CHARGE=1, REPOSITION=2 (IDLE removed)
        action_type = torch.randint(0, 3, (env.num_vehicles,), device=device)
        reposition_target = torch.randint(0, env.num_hexes, (env.num_vehicles,), device=device)
        selected_trip = torch.randint(0, 500, (env.num_vehicles,), device=device)  # NEW: random trip selections
        
        next_state, reward, done, info = env.step(action_type, reposition_target, selected_trip)
        
        # Convert flat state to dict format for replay buffer
        state_dict = env._build_state_dict(state)
        next_state_dict = env._build_state_dict(next_state)
        
        # Action: [num_vehicles, 2]
        action = torch.zeros(env.num_vehicles, 2, dtype=torch.long, device=device)
        action[:, 0] = action_type
        action[:, 1] = reposition_target
        
        replay_buffer.push(
            state=state_dict,
            action=action,
            reward=reward.item() if isinstance(reward, torch.Tensor) else reward,
            next_state=next_state_dict,
            done=done.item() if isinstance(done, torch.Tensor) else done
        )
        
        if done.item() if isinstance(done, torch.Tensor) else done:
            state = env.reset()
        else:
            state = next_state
        
        if (step + 1) % 1000 == 0:
            print(f"  Warmup: {step + 1}/{num_steps} transitions")
    
    print(f"Warmup complete. Buffer size: {len(replay_buffer)}")


def train_single_gpu(args):
    print("=" * 60)
    print("EV Fleet RL Training - GPU Accelerated")
    print("=" * 60)
    
    setup_seed(args.seed)
    device = get_device(args)
    print(f"Device: {device}")
    
    config = create_config(args)
    
    # Load real trip data if provided (CLI overrides YAML)
    trip_loader = None
    data_path = args.real_data or config.data.parquet_path
    sample_ratio = args.trip_sample if args.trip_sample is not None else config.data.trip_percentage
    target_res = args.target_h3_resolution if hasattr(args, 'target_h3_resolution') and args.target_h3_resolution is not None else config.data.target_h3_resolution
    max_hex_count = args.max_hex_count if hasattr(args, 'max_hex_count') and args.max_hex_count is not None else config.data.max_hex_count
    if data_path:
        resolved_path = Path(data_path)
        resolved_str = str(resolved_path)
        sample_ratio = sample_ratio if sample_ratio is not None else 1.0
        print(f"\n[Real Data] Loading from: {resolved_str}")
        if sample_ratio < 1.0:
            print(f"[Real Data] Using {sample_ratio*100:.0f}% sample of trips")
        if target_res is not None:
            print(f"[Real Data] Target H3 resolution: {target_res}")
        if max_hex_count is not None:
            print(f"[Real Data] Max hex count: {max_hex_count}")
        if not resolved_path.exists():
            print(f"[Real Data] File not found: {resolved_str}. Falling back to synthetic data.")
        else:
            try:
                trip_loader = RealTripLoader(
                    parquet_path=resolved_str,
                    device=str(device),
                    sample_ratio=sample_ratio,
                    target_h3_resolution=target_res,
                    max_hex_count=max_hex_count,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
                trip_loader.load()
            except Exception as exc:
                print(f"[Real Data] Failed to load due to: {exc}. Falling back to synthetic data.")
                trip_loader = None
        if trip_loader and trip_loader.is_loaded:
            config.environment.num_hexes = trip_loader.num_hexes
            print(f"[Real Data] Loaded {trip_loader.total_trips:,} trips | Hexes: {trip_loader.num_hexes:,}")
            print(f"[Real Data] Fare range: ${trip_loader.fare_stats['min']:.2f} - ${trip_loader.fare_stats['max']:.2f}")
        elif trip_loader is None:
            print("[Real Data] Warning: proceeding with synthetic trip generator")
    else:
        print("\n[Data] Using synthetic trips (set data.parquet_path or --real-data for NYC trips)")
    
    print(f"Vehicles: {config.environment.num_vehicles}")
    print(f"Hexes: {config.environment.num_hexes}")
    print(f"Episodes: {config.training.total_episodes}")
    print(f"Batch Size: {config.training.batch_size}")
    
    env = create_environment(config, device, args, trip_loader=trip_loader)
    print(f"\nEnvironment created with {env.num_vehicles} vehicles")
    
    # Create agent with GCN support
    agent = create_agent(config, device, env=env)
    
    # Set adjacency matrix for GCN actor
    if hasattr(env, '_adjacency_matrix') and env._adjacency_matrix is not None:
        agent.set_adjacency(env._adjacency_matrix)
        print(f"[GCN] Adjacency matrix set for GCN actor")
    else:
        print(f"[GCN] Warning: No adjacency matrix available, GCN may not work properly")
    print(f"[GCN] Using GCN-based actor for per-vehicle spatial reasoning")
    
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    gpu_ids = [int(x) for x in args.gpus.split(',')] if args.gpus else None
    
    # Auto-enable data_parallel if multiple GPUs were passed without --data-parallel flag
    if gpu_ids and len(gpu_ids) > 1 and not (args.data_parallel or args.distributed):
        print(f"\n[Warning] Multiple GPUs ({gpu_ids}) specified but no parallel flag was used.")
        print(f"[Warning] Auto-enabling --data-parallel to utilize multiple GPUs!\n")
        args.data_parallel = True

    # DataParallel for multi-GPU training (simpler than DDP)
    if args.data_parallel and (torch.cuda.device_count() > 1 or (gpu_ids and len(gpu_ids) > 1)):
        num_gpus = len(gpu_ids) if gpu_ids else torch.cuda.device_count()
        print(f"[DataParallel] Using {num_gpus} GPUs (IDs: {gpu_ids or 'All visible'}) for training")
        agent.actor = torch.nn.DataParallel(agent.actor, device_ids=gpu_ids)
        agent.critic = torch.nn.DataParallel(agent.critic, device_ids=gpu_ids)
        agent.critic_target = torch.nn.DataParallel(agent.critic_target, device_ids=gpu_ids)
        print(f"[DataParallel] Models wrapped - effective split batch size: {args.batch_size}")
        print(f"[DataParallel Note] DataParallel keeps the Replay Buffer and Environment on GPU 0.")
        print(f"[DataParallel Note] VRAM usage will appear unbalanced (GPU 0 maxed, GPU 1 low usage).")
        print(f"[DataParallel Note] For balanced VRAM usage, use --distributed instead of --data-parallel.\n")
    
    # Compile models for faster execution (PyTorch 2.0+)
    # Note: torch.compile with DataParallel may have issues
    if hasattr(torch, 'compile') and not args.data_parallel:
        try:
            print("[Optimization] Compiling models with torch.compile...")
            agent.actor = torch.compile(agent.actor, mode='reduce-overhead', options={"triton.cudagraphs": False})
            agent.critic = torch.compile(agent.critic, mode='reduce-overhead', options={"triton.cudagraphs": False})
            agent.critic_target = torch.compile(agent.critic_target, mode='reduce-overhead', options={"triton.cudagraphs": False})
            print("[Optimization] Model compilation complete")
        except Exception as e:
            print(f"[Optimization] Warning: torch.compile failed: {e}")
            print("[Optimization] Continuing without compilation")
    
    replay_buffer = create_replay_buffer(config, device, env=env)
    # Capacity already printed by create_replay_buffer() with auto-adjustment info
    
    warmup_steps = config.training.warmup_steps
    if warmup_steps > 0 and not args.resume:
        warmup_replay_buffer(env, replay_buffer, warmup_steps, device)
    
    # Create trainer - use EnhancedSACTrainer if Semi-MDP or WDRO is enabled
    use_enhanced = args.semi_mdp or args.wdro
    if use_enhanced:
        enhanced_config = EnhancedTrainingConfig(
            use_semi_mdp=args.semi_mdp,
            use_wdro=args.wdro,
            wdro_rho=args.wdro_rho,
            use_temperature_annealing=args.temperature_annealing,
            initial_temperature=args.initial_temperature,
            final_temperature=args.final_temperature,
            temperature_decay_episodes=args.temperature_decay_episodes,
            use_milp=args.milp,
            use_amp=True,  # Enable mixed precision training
        )
        # Get adjacency matrix for WDRO graph-aligned metric (paper Eq. 17)
        adj_matrix = getattr(env, '_adjacency_matrix', None)
        trainer = EnhancedSACTrainer(
            agent=agent,
            replay_buffer=replay_buffer,
            training_config=config.training,
            enhanced_config=enhanced_config,
            checkpoint_config=config.checkpoint,
            logging_config=config.logging,
            device=str(device),
            adjacency_matrix=adj_matrix
        )
        print(f"[Enhanced] Semi-MDP: {args.semi_mdp}, WDRO: {args.wdro}, MILP: {args.milp}")
        print(f"[Optimization] Mixed Precision Training (AMP): Enabled")
    else:
        trainer = create_trainer(agent, replay_buffer, config, device)
    
    # Create collector - use EnhancedEpisodeCollector for duration tracking if Semi-MDP enabled
    if args.semi_mdp:
        collector = EnhancedEpisodeCollector(
            env=env,
            replay_buffer=replay_buffer,
            device=str(device),
            use_milp=args.milp,
            track_durations=True
        )
        print(f"[Enhanced] Using EnhancedEpisodeCollector with duration tracking")
    else:
        collector = EpisodeCollector(
            env=env,
            replay_buffer=replay_buffer,
            device=str(device),
            use_milp=args.milp
        )
        if args.milp:
            print(f"[Assignment] MILP EXACT ASSIGNMENT ENABLED (Gurobi)")
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    print("=" * 60)
    print("Training started...")
    print("=" * 60)
    
    # Setup TensorBoard and CSV logging
    writer = None
    csv_file = None
    csv_writer = None
    
    # Setup log directory
    log_dir = os.path.join(config.logging.log_dir, f"run_{int(time.time())}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize CSV logging
    csv_path = os.path.join(log_dir, "training_history.csv")
    try:
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        # Write CSV header
        csv_writer.writerow([
            'Episode', 'Reward', 'Avg100', 'Profit', 'Revenue', 'Serve_Pct', 'Steps', 'Trips_Served', 
            'Trips_Loaded', 'Avg_SOC', 'Actor_Loss', 'Critic_Loss', 
            'Alpha', 'Idle_Pct', 'Serve_Act_Pct', 'Charge_Pct', 'Repos_Pct', 'Elapsed_Sec'
        ])
        print(f"CSV logging to: {csv_path}")
    except Exception as e:
        print(f"Failed to initialize CSV logging: {e}")
        
    if args.tensorboard and HAS_TENSORBOARD:
        run_name = f"ev_fleet_{config.environment.num_vehicles}v_{config.environment.num_hexes}h_{int(time.time())}"
        writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_dir, run_name))
        print(f"TensorBoard logging to: {args.tensorboard_dir}/{run_name}")
    elif args.tensorboard and not HAS_TENSORBOARD:
        print("Warning: TensorBoard requested but not installed. Run: pip install tensorboard")
    
    start_time = time.time()
    total_steps = getattr(trainer, 'global_step', 0)
    total_updates = getattr(trainer, 'global_step', 0)
    best_eval_reward = getattr(trainer, 'best_reward', float('-inf'))
    best_train_reward = float('-inf')  # Track best training reward too
    
    # Metrics tracking
    recent_rewards = deque(maxlen=100)
    recent_losses = {
        'actor_loss': deque(maxlen=100),
        'critic_loss': deque(maxlen=100),
        'alpha': deque(maxlen=100),
        'q_mean': deque(maxlen=100),
    }
    
    start_episode = getattr(trainer, 'episode', 0)
    if args.start_episode is not None:
        start_episode = args.start_episode
        trainer.episode = start_episode
    if start_episode > 0:
        print(f"Resuming training loop from episode {start_episode}")
    
    # Track time per 20 episodes
    episode_batch_start_time = time.time()
    episode_batch_count = 0
        
    for episode in range(start_episode, config.training.total_episodes):
        # Track episode in trainer so checkpoints save the correct episode number
        trainer.episode = episode
        
        # Lower epsilon-greedy noise since we now have per-vehicle stochastic sampling
        # The policy entropy (via alpha) provides main exploration
        # Fixed low noise - SAC entropy (alpha + temperature) handles exploration
        exploration_noise = getattr(config.training, 'exploration_noise', 0.01)
        
        # Get temperature for this episode (if using temperature annealing)
        if use_enhanced and args.temperature_annealing:
            temperature = trainer.get_temperature()
        else:
            temperature = 1.0
        
        with _time_section("episode_collection"):
            episode_stats = collector.collect_episode(
                agent=agent,
                exploration_noise=exploration_noise,
                seed=args.seed + episode,
                temperature=temperature,
                deterministic=args.deterministic
            )
        
        total_steps += episode_stats.steps
        
        if len(replay_buffer) >= config.training.batch_size:
            # Use configurable update-to-data ratio
            # Default: 1 update per step for balance between learning and throughput
            update_ratio = getattr(args, 'update_ratio', getattr(config.training, 'update_every', 1))
            num_updates = max(1, episode_stats.steps // update_ratio)
            
            for update_i in range(num_updates):
                # Use enhanced train_step if available
                if use_enhanced:
                    with _time_section("training_step"):
                        train_info = trainer.train_step_enhanced()
                else:
                        with _time_section("training_step"):
                            train_info = trainer.train_step()
                total_updates += 1
                
                # Track loss metrics
                if train_info:
                    for key in ['actor_loss', 'critic_loss', 'alpha', 'q_mean']:
                        if key in train_info:
                            recent_losses[key].append(train_info[key])
                    
                    # Log to TensorBoard
                    if writer and total_updates % args.log_loss_interval == 0:
                        for key, value in train_info.items():
                            writer.add_scalar(f'Loss/{key}', value, total_updates)
                    
                    # Print detailed loss periodically
                    if args.debug and total_updates % (args.log_loss_interval * 10) == 0:
                        print(f"  [Update {total_updates}] "
                              f"Actor: {train_info.get('actor_loss', 0):.4f} | "
                              f"Critic: {train_info.get('critic_loss', 0):.4f} | "
                              f"Alpha: {train_info.get('alpha', 0):.4f} | "
                              f"Q: {train_info.get('q_mean', 0):.2f}")
        
        # Track episode reward
        recent_rewards.append(episode_stats.total_reward)
        episode_batch_count += 1
        
        # Log timing every 20 episodes
        if episode_batch_count == 20:
            batch_elapsed = time.time() - episode_batch_start_time
            print(f"[Timing] Last 20 episodes took {batch_elapsed:.2f}s ({batch_elapsed/20:.2f}s per episode)")
            episode_batch_start_time = time.time()
            episode_batch_count = 0
        
        if (episode + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed
            updates_per_sec = total_updates / elapsed if elapsed > 0 else 0
            
            # Calculate average losses
            avg_actor_loss = sum(recent_losses['actor_loss']) / len(recent_losses['actor_loss']) if recent_losses['actor_loss'] else 0
            avg_critic_loss = sum(recent_losses['critic_loss']) / len(recent_losses['critic_loss']) if recent_losses['critic_loss'] else 0
            avg_alpha = sum(recent_losses['alpha']) / len(recent_losses['alpha']) if recent_losses['alpha'] else 0
            avg_reward_100 = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            
            # Get action distribution
            action_counts = getattr(episode_stats, 'action_counts', None)
            if action_counts:
                total_actions = sum(action_counts.values()) or 1
                action_pct = {k: v * 100.0 / total_actions for k, v in action_counts.items()}
                action_str = f"S:{action_pct['serve']:4.1f}% C:{action_pct['charge']:4.1f}% R:{action_pct['reposition']:4.1f}%"
            else:
                action_str = ""
            
            serve_pct_stat = (episode_stats.trips_served / max(1, episode_stats.trips_loaded)) * 100.0
            print(f"Episode {episode + 1:5d} | "
                f"Reward: {episode_stats.total_reward:8.2f} | "
                f"Avg100: {avg_reward_100:8.2f} | "
                f"Profit: ${episode_stats.profit:6.2f} | "
                f"Rev: ${episode_stats.revenue:6.2f} | "
                f"Serve%: {serve_pct_stat:4.1f}% | "
                f"SOC: {episode_stats.avg_soc:5.1f}%")
            print(f"         Actor: {avg_actor_loss:7.4f} | "
                f"Critic: {avg_critic_loss:7.4f} | "
                f"Alpha: {avg_alpha:5.3f} | "
                f"Actions: {action_str}")
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('Episode/reward', episode_stats.total_reward, episode + 1)
                writer.add_scalar('Episode/avg_reward_100', avg_reward_100, episode + 1)
                writer.add_scalar('Episode/steps', episode_stats.steps, episode + 1)
                writer.add_scalar('Episode/trips_served', episode_stats.trips_served, episode + 1)
                writer.add_scalar('Episode/trips_loaded', episode_stats.trips_loaded, episode + 1)
                writer.add_scalar('Episode/avg_soc', episode_stats.avg_soc, episode + 1)
                writer.add_scalar('Speed/steps_per_sec', steps_per_sec, episode + 1)
                writer.add_scalar('Speed/updates_per_sec', updates_per_sec, episode + 1)
                writer.add_scalar('Loss/avg_actor_loss', avg_actor_loss, episode + 1)
                writer.add_scalar('Loss/avg_critic_loss', avg_critic_loss, episode + 1)
                writer.add_scalar('Loss/avg_alpha', avg_alpha, episode + 1)
                # Log action distribution
                if action_counts:
                    for action_name, count in action_counts.items():
                        writer.add_scalar(f'Actions/{action_name}', count, episode + 1)
                        writer.add_scalar(f'ActionPct/{action_name}', action_pct[action_name], episode + 1)
                        
            # Log to CSV
            if csv_writer and csv_file:
                idle_pct = action_pct.get('idle', 0) if action_counts else 0
                serve_act_pct = action_pct.get('serve', 0) if action_counts else 0
                charge_pct = action_pct.get('charge', 0) if action_counts else 0
                repos_pct = action_pct.get('reposition', 0) if action_counts else 0
                
                serve_pct_stat = (episode_stats.trips_served / max(1, episode_stats.trips_loaded)) * 100.0
                csv_writer.writerow([
                    episode + 1,
                    round(episode_stats.total_reward, 2),
                    round(avg_reward_100, 2),
                    round(episode_stats.profit, 2),
                    round(episode_stats.revenue, 2),
                    round(serve_pct_stat, 1),
                    episode_stats.steps,
                    episode_stats.trips_served,
                    episode_stats.trips_loaded,
                    round(episode_stats.avg_soc, 1),
                    round(avg_actor_loss, 4),
                    round(avg_critic_loss, 4),
                    round(avg_alpha, 4),
                    round(idle_pct, 1),
                    round(serve_act_pct, 1),
                    round(charge_pct, 1),
                    round(repos_pct, 1),
                    round(elapsed, 1)
                ])
                csv_file.flush()  # Ensure data is written immediately
        
        if (episode + 1) % args.eval_interval == 0:
            eval_reward = evaluate_agent(
                agent, env, args.eval_episodes, device
            )
            print(f"  [Eval] Avg Reward: {eval_reward:.2f}")
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                trainer.save_checkpoint('best_eval.pt')
                print(f"  [Best Eval] New best eval model saved!")
        
        # Save best model based on training reward (Avg100)
        avg_reward_100 = sum(recent_rewards) / len(recent_rewards) if recent_rewards else float('-inf')
        if len(recent_rewards) >= 50 and avg_reward_100 > best_train_reward:
            best_train_reward = avg_reward_100
            trainer.save_checkpoint('best.pt')
            print(f"  [Best Train] New best model saved! Avg100: {avg_reward_100:.2f}")
        
        if (episode + 1) % args.save_interval == 0:
            trainer.save_checkpoint(f'checkpoint_{episode + 1}.pt')
    
    elapsed = time.time() - start_time
    steps_per_sec = total_steps / elapsed
    
    print("=" * 60)
    print("Training complete!")
    print(f"Total episodes: {config.training.total_episodes}")
    print(f"Total steps: {total_steps:,}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Average steps/second: {steps_per_sec:.1f}")
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print(f"Best train reward (Avg100): {best_train_reward:.2f}")
    print("=" * 60)
    
    # Print timing report
    print("\n" + "=" * 60)
    print("TIMING STATISTICS")
    print("=" * 60)
    for name, times in sorted(_timing_stats.items(), key=lambda x: sum(x[1]), reverse=True):
        total = sum(times)
        mean = total / len(times) if times else 0
        max_time = max(times) if times else 0
        count = len(times)
        print(f"{name:<30} Total: {total:>8.2f}s  Mean: {mean:>7.4f}s  Max: {max_time:>7.4f}s  Count: {count:>6}")
    print("=" * 60)
    
    trainer.save_checkpoint('final.pt')
    print(f"Final model saved to: {config.checkpoint.checkpoint_dir}/final.pt")
    
    # Close TensorBoard writer and CSV file
    if writer:
        writer.close()
        print(f"TensorBoard logs saved. View with: tensorboard --logdir={args.tensorboard_dir}")
        
    if csv_file:
        csv_file.close()
        print(f"CSV logs saved to: {csv_path}")


def evaluate_agent(
    agent: SACAgent,
    env: GPUEnvironment,
    num_episodes: int,
    device: torch.device
) -> float:
    """Evaluate agent performance."""
    total_reward = 0.0
    total_revenue = 0.0
    total_profit = 0.0
    total_served = 0
    total_loaded = 0
    
    agent.eval()
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            with torch.no_grad():
                # Get available action mask
                available_mask = env.get_available_actions()
                
                # Generate reposition mask (spatial limitation)
                vehicle_hex_ids = env.fleet_state.positions.long()
                num_vehicles = vehicle_hex_ids.shape[0]
                reposition_mask = torch.ones(num_vehicles, env.num_hexes, dtype=torch.bool, device=device)
                if hasattr(env, 'hex_grid') and hasattr(env.hex_grid, 'distance_matrix') and env.hex_grid.distance_matrix is not None:
                    try:
                        # Get threshold
                        threshold = getattr(env, 'max_pickup_distance', 5.0)
                        
                        # Most memory efficient approach: Create target tensor and use get_distances_batch
                        # This avoids slicing a 1000x3985 float tensor (~15MB per step)
                        all_hexes = torch.arange(env.num_hexes, device=device).unsqueeze(0).expand(num_vehicles, -1)
                        veh_hexes_expanded = vehicle_hex_ids.unsqueeze(1).expand(-1, env.num_hexes)
                        
                        # Memory efficient batch distance calculation
                        veh_distances = env.hex_grid.distance_matrix.get_distances_batch(
                            veh_hexes_expanded.reshape(-1), 
                            all_hexes.reshape(-1)
                        ).view(num_vehicles, env.num_hexes)
                        
                        reposition_mask = veh_distances <= threshold
                        reposition_mask.scatter_(1, vehicle_hex_ids.unsqueeze(1), False)
                    except AttributeError:
                        # Fallback if get_distances_batch is missing, though less memory efficient
                        distance_matrix = env.hex_grid.distance_matrix._distances
                        if distance_matrix is not None:
                            veh_distances = distance_matrix[vehicle_hex_ids]
                            threshold = getattr(env, 'max_pickup_distance', 5.0)
                            reposition_mask = veh_distances <= threshold
                            reposition_mask.scatter_(1, vehicle_hex_ids.unsqueeze(1), False)
                
                # Generate trip mask (2D Spatial Match)
                trip_mask = None
                if hasattr(env, 'trip_state'):
                    unassigned_mask = env.trip_state.get_unassigned_mask()
                    if unassigned_mask.any():
                        actor_module = agent.actor.module if hasattr(agent.actor, 'module') else agent.actor
                        max_trips = getattr(actor_module, 'max_trips', 500)
                        trip_mask = torch.zeros(num_vehicles, max_trips, dtype=torch.bool, device=device)
                        
                        available_indices = unassigned_mask.nonzero(as_tuple=True)[0]
                        num_available = min(len(available_indices), max_trips)
                        
                        if num_available > 0:
                            pickup_hexes = env.trip_state.pickup_hex[available_indices[:num_available]]
                            
                            if hasattr(env, 'hex_grid') and hasattr(env.hex_grid, 'distance_matrix') and env.hex_grid.distance_matrix is not None:
                                try:
                                    # Memory efficient batch distance calculation
                                    veh_hexes_expanded = vehicle_hex_ids.unsqueeze(1).expand(-1, num_available)
                                    pickup_hexes_expanded = pickup_hexes.unsqueeze(0).expand(num_vehicles, -1)
                                    
                                    veh_distances = env.hex_grid.distance_matrix.get_distances_batch(
                                        veh_hexes_expanded.reshape(-1),
                                        pickup_hexes_expanded.reshape(-1)
                                    ).view(num_vehicles, num_available)
                                    
                                    threshold = getattr(env, 'max_pickup_distance', 5.0)
                                    within = veh_distances <= threshold
                                except AttributeError:
                                    # Fallback
                                    dist_matrix = env.hex_grid.distance_matrix._distances
                                    within = dist_matrix[vehicle_hex_ids[:, None], pickup_hexes[None, :]]
                                    threshold = getattr(env, 'max_pickup_distance', 5.0)
                                    within = within <= threshold
                            else:
                                within = torch.ones(num_vehicles, num_available, dtype=torch.bool, device=device)
                                
                            trip_mask[:, :num_available] = within
                            
                            # Fallback: vehicles with zero reachable trips get all available unmasked
                            # This allows them to output a valid softmax, action_mask prevents SERVE
                            no_reachable = ~trip_mask.any(dim=1)
                            if no_reachable.any():
                                trip_mask[no_reachable, :num_available] = True

                # Use agent's select_action_gcn if available to get full capabilities
                if hasattr(agent, 'select_action_gcn'):
                    output = agent.select_action_gcn(
                        state=state,
                        action_mask=available_mask,
                        reposition_mask=reposition_mask,
                        trip_mask=trip_mask,
                        vehicle_hex_ids=vehicle_hex_ids,
                        deterministic=False
                    )
                else:
                    output = agent.select_action(
                        state, 
                        action_mask=available_mask,
                        reposition_mask=reposition_mask,
                        deterministic=False
                    )
                    
                action_type = output.action_type
                reposition_target = output.reposition_target
                selected_trip = getattr(output, 'selected_trip', None)  # get trip selections from GCN
                
                # GCN outputs [batch, num_vehicles], flatten to [num_vehicles]
                if action_type.dim() > 1:
                    action_type = action_type.squeeze(0)
                    reposition_target = reposition_target.squeeze(0)
                    if selected_trip is not None:
                        selected_trip = selected_trip.squeeze(0)
                
                # If still scalar (single action), expand to all vehicles
                if action_type.numel() == 1:
                    action_type = action_type.expand(env.num_vehicles)
                    reposition_target = reposition_target.expand(env.num_vehicles)
                    if selected_trip is not None and selected_trip.numel() == 1:
                        selected_trip = selected_trip.expand(env.num_vehicles)
            
            next_state, reward, done_tensor, info = env.step(action_type, reposition_target, selected_trip)
            done = done_tensor.item()
            episode_reward += reward.item()
            state = next_state
        
        total_reward += episode_reward
        total_revenue += info.revenue
        total_profit += info.revenue - getattr(info, 'driving_cost', 0.0) - info.energy_cost
        total_served += info.trips_served
        total_loaded += info.trips_loaded
    
    agent.train()
    
    avg_reward = total_reward / num_episodes
    avg_revenue = total_revenue / num_episodes
    avg_profit = total_profit / num_episodes
    serve_pct = (total_served / max(1, total_loaded)) * 100.0
    
    print(f"  [Eval Detail] Profit: ${avg_profit:.2f} | Revenue: ${avg_revenue:.2f} | Serve: {serve_pct:.1f}%")
    return avg_reward


def train_distributed(rank: int, world_size: int, args):
    # setup_distributed already called by _distributed_worker_wrapper
    try:
        device = torch.device(f'cuda:{rank}')
        setup_seed(args.seed + rank)
        
        if rank == 0:
            print("=" * 60)
            print("EV Fleet RL Training - Distributed (Multi-GPU)")
            print("=" * 60)
        config = create_config(args)
        
        # Load real trip data if provided
        trip_loader = None
        data_path = args.real_data or config.data.parquet_path
        sample_ratio = args.trip_sample if args.trip_sample is not None else config.data.trip_percentage
        target_res = args.target_h3_resolution if hasattr(args, 'target_h3_resolution') and args.target_h3_resolution is not None else config.data.target_h3_resolution
        max_hex_count = args.max_hex_count if hasattr(args, 'max_hex_count') and args.max_hex_count is not None else config.data.max_hex_count
        if data_path:
            resolved_path = Path(data_path)
            resolved_str = str(resolved_path)
            sample_ratio = sample_ratio if sample_ratio is not None else 1.0
            if rank == 0:
                print(f"\n[Real Data] Loading from: {resolved_str}")
                if sample_ratio < 1.0:
                    print(f"[Real Data] Using {sample_ratio*100:.0f}% sample of trips")
            if resolved_path.exists():
                try:
                    trip_loader = RealTripLoader(
                        parquet_path=resolved_str,
                        device=str(device),
                        sample_ratio=sample_ratio,
                        target_h3_resolution=target_res,
                        max_hex_count=max_hex_count,
                        start_date=args.start_date,
                        end_date=args.end_date,
                    )
                    trip_loader.load()
                except Exception as exc:
                    if rank == 0:
                        print(f"[Real Data] Failed to load: {exc}. Falling back to synthetic data.")
                    trip_loader = None
            if trip_loader and trip_loader.is_loaded:
                config.environment.num_hexes = trip_loader.num_hexes
                if rank == 0:
                    print(f"[Real Data] Loaded {trip_loader.total_trips:,} trips | Hexes: {trip_loader.num_hexes:,}")
        
        env = create_environment(config, device, args, trip_loader=trip_loader)
        if rank == 0:
            print(f"Vehicles: {config.environment.num_vehicles}")
            print(f"Hexes: {config.environment.num_hexes}")
            print(f"Episodes: {config.training.total_episodes}")
        
        agent = create_agent(config, device, env=env)
        
        # Set adjacency matrix for GCN
        if hasattr(env, '_adjacency_matrix') and env._adjacency_matrix is not None:
            agent.set_adjacency(env._adjacency_matrix)
        
        if rank == 0:
            print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")
        
        replay_buffer = create_replay_buffer(config, device, env=env)
        
        warmup_steps = config.training.warmup_steps
        if warmup_steps > 0 and not args.resume:
            if rank == 0:
                print(f"Warming up replay buffer with {warmup_steps} random steps...")
            warmup_replay_buffer(env, replay_buffer, warmup_steps, device)
        
        # Create trainer - use EnhancedSACTrainer if Semi-MDP or WDRO is enabled
        use_enhanced = args.semi_mdp or args.wdro
        if use_enhanced:
            enhanced_config = EnhancedTrainingConfig(
                use_semi_mdp=args.semi_mdp,
                use_wdro=args.wdro,
                wdro_rho=args.wdro_rho,
                use_temperature_annealing=args.temperature_annealing,
                initial_temperature=args.initial_temperature,
                final_temperature=args.final_temperature,
                temperature_decay_episodes=args.temperature_decay_episodes,
                use_amp=True,
            )
            adj_matrix = getattr(env, '_adjacency_matrix', None)
            trainer = EnhancedSACTrainer(
                agent=agent,
                replay_buffer=replay_buffer,
                training_config=config.training,
                enhanced_config=enhanced_config,
                checkpoint_config=config.checkpoint,
                logging_config=config.logging,
                device=str(device),
                adjacency_matrix=adj_matrix
            )
            if rank == 0:
                print(f"[Enhanced] Semi-MDP: {args.semi_mdp}, WDRO: {args.wdro}")
                print(f"[Optimization] Mixed Precision Training (AMP): Enabled")
        else:
            trainer = create_trainer(
                agent, replay_buffer, config, device,
                rank=rank, world_size=world_size
            )
        
        if args.resume:
            trainer.load_checkpoint(args.resume)
            if rank == 0:
                print(f"Resumed from checkpoint: {args.resume}")
        
        # Create collector
        if args.semi_mdp:
            from gpu_core.training.enhanced_collector import EnhancedEpisodeCollector
            collector = EnhancedEpisodeCollector(
                env=env, replay_buffer=replay_buffer, device=str(device),
                use_milp=args.milp, track_durations=True
            )
        else:
            from gpu_core.training.episode_collector import EpisodeCollector
            collector = EpisodeCollector(
                env=env, replay_buffer=replay_buffer, device=str(device),
                use_milp=args.milp
            )
        
        # === CSV Logging (rank 0 only) ===
        csv_file = None
        csv_writer = None
        log_dir = None
        if rank == 0:
            log_dir = os.path.join(config.logging.log_dir, f"run_{int(time.time())}")
            os.makedirs(log_dir, exist_ok=True)
            csv_path = os.path.join(log_dir, "training_history.csv")
            try:
                csv_file = open(csv_path, mode='w', newline='')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([
                    'Episode', 'Reward', 'Avg100', 'Profit', 'Revenue', 'Serve_Pct', 'Steps', 'Trips_Served',
                    'Trips_Loaded', 'Avg_SOC', 'Actor_Loss', 'Critic_Loss',
                    'Alpha', 'Idle_Pct', 'Serve_Act_Pct', 'Charge_Pct', 'Repos_Pct', 'Elapsed_Sec'
                ])
                print(f"CSV logging to: {csv_path}")
            except Exception as e:
                print(f"Failed to initialize CSV logging: {e}")
        
        # === Tracking ===
        recent_rewards = deque(maxlen=100)
        recent_losses = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'alpha': deque(maxlen=100),
        }
        best_eval_reward = float('-inf')
        best_train_reward = float('-inf')
        total_steps = 0
        total_updates = 0
        start_time = time.time()
        
        if rank == 0:
            print("=" * 60)
            print(f"Distributed training with {world_size} GPUs started...")
            print("=" * 60)
        
        num_episodes = config.training.total_episodes
        start_episode = getattr(trainer, 'episode', 0)
        if args.start_episode is not None:
            start_episode = args.start_episode
            trainer.episode = start_episode
        if rank == 0 and start_episode > 0:
            print(f"Resuming training loop from episode {start_episode}")
        action_pct = {}
        
        # Track time per 20 episodes
        episode_batch_start_time = time.time()
        episode_batch_count = 0
        
        for episode in range(start_episode, num_episodes):
            # Track episode in trainer so checkpoints save the correct episode number
            trainer.episode = episode
            
            exploration_noise = getattr(config.training, 'exploration_noise', 0.01)
            
            # Get temperature for this episode (if using temperature annealing)
            if use_enhanced and args.temperature_annealing:
                temperature = trainer.get_temperature()
            else:
                temperature = 1.0
            
            # Collect episode
            episode_stats = collector.collect_episode(
                agent=agent, exploration_noise=exploration_noise,
                seed=args.seed + episode, temperature=temperature,
                deterministic=args.deterministic
            )
            total_steps += episode_stats.steps
            
            # Check if all ranks have enough data to train
            local_ready = 1 if len(replay_buffer) >= config.training.batch_size else 0
            if args.distributed:
                ready_tensor = torch.tensor([local_ready], dtype=torch.int32, device=device)
                dist.all_reduce(ready_tensor, op=dist.ReduceOp.MIN)
                global_ready = (ready_tensor.item() == 1)
            else:
                global_ready = (local_ready == 1)
                
            # Train on collected data
            if global_ready:
                update_ratio = getattr(config.training, 'update_every', 1)
                local_updates = max(1, episode_stats.steps // update_ratio)
                
                # Make sure all ranks perform the exact same number of updates to prevent DDP deadlocks
                if args.distributed:
                    updates_tensor = torch.tensor([local_updates], dtype=torch.float32, device=device)
                    dist.all_reduce(updates_tensor, op=dist.ReduceOp.MAX)
                    num_updates = int(updates_tensor.item())
                else:
                    num_updates = local_updates
                
                for _ in range(num_updates):
                    if use_enhanced:
                        train_info = trainer.train_step_enhanced()
                    else:
                        train_info = trainer.train_step()
                    total_updates += 1
                    
                    if train_info:
                        for key in ['actor_loss', 'critic_loss', 'alpha']:
                            if key in train_info:
                                recent_losses[key].append(train_info[key])
            
            # Track reward
            recent_rewards.append(episode_stats.total_reward)
            episode_batch_count += 1
            
            # Log timing every 20 episodes (rank 0 only)
            if rank == 0 and episode_batch_count == 20:
                batch_elapsed = time.time() - episode_batch_start_time
                print(f"[Timing] Last 20 episodes took {batch_elapsed:.2f}s ({batch_elapsed/20:.2f}s per episode)")
                episode_batch_start_time = time.time()
                episode_batch_count = 0
            
            # === Logging (rank 0 only) ===
            if rank == 0 and (episode + 1) % args.log_interval == 0:
                avg_actor_loss = sum(recent_losses['actor_loss']) / len(recent_losses['actor_loss']) if recent_losses['actor_loss'] else 0
                avg_critic_loss = sum(recent_losses['critic_loss']) / len(recent_losses['critic_loss']) if recent_losses['critic_loss'] else 0
                avg_alpha = sum(recent_losses['alpha']) / len(recent_losses['alpha']) if recent_losses['alpha'] else 0
                avg_reward_100 = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                
                # Action distribution
                action_counts = getattr(episode_stats, 'action_counts', None)
                if action_counts:
                    total_actions = sum(action_counts.values()) or 1
                    action_pct = {k: v * 100.0 / total_actions for k, v in action_counts.items()}
                    action_str = f"S:{action_pct['serve']:4.1f}% C:{action_pct['charge']:4.1f}% R:{action_pct['reposition']:4.1f}%"
                else:
                    action_str = ""
                    action_pct = {}
                
                serve_pct_stat = (episode_stats.trips_served / max(1, episode_stats.trips_loaded)) * 100.0
                print(f"Episode {episode + 1:5d} | "
                    f"Reward: {episode_stats.total_reward:8.2f} | "
                    f"Avg100: {avg_reward_100:8.2f} | "
                    f"Profit: ${episode_stats.profit:6.2f} | "
                    f"Rev: ${episode_stats.revenue:6.2f} | "
                    f"Serve%: {serve_pct_stat:4.1f}% | "
                    f"SOC: {episode_stats.avg_soc:5.1f}%")
                print(f"         Actor: {avg_actor_loss:7.4f} | "
                    f"Critic: {avg_critic_loss:7.4f} | "
                    f"Alpha: {avg_alpha:5.3f} | "
                    f"Actions: {action_str}")
                
                # CSV logging
                if csv_writer and csv_file:
                    idle_pct = action_pct.get('idle', 0)
                    serve_act_pct = action_pct.get('serve', 0)
                    charge_pct = action_pct.get('charge', 0)
                    repos_pct = action_pct.get('reposition', 0)
                    
                    dist_elapsed = time.time() - start_time
                    csv_writer.writerow([
                        episode + 1,
                        round(episode_stats.total_reward, 2),
                        round(avg_reward_100, 2),
                        round(episode_stats.profit, 2),
                        round(episode_stats.revenue, 2),
                        round(serve_pct_stat, 1),
                        episode_stats.steps,
                        episode_stats.trips_served,
                        episode_stats.trips_loaded,
                        round(episode_stats.avg_soc, 1),
                        round(avg_actor_loss, 4),
                        round(avg_critic_loss, 4),
                        round(avg_alpha, 4),
                        round(idle_pct, 1),
                        round(serve_act_pct, 1),
                        round(charge_pct, 1),
                        round(repos_pct, 1),
                        round(dist_elapsed, 1)
                    ])
                    csv_file.flush()
            
            # === Evaluation ===
            if (episode + 1) % args.eval_interval == 0:
                # All ranks must evaluate to keep NCCL in sync
                eval_reward = evaluate_agent(agent, env, args.eval_episodes, device)
                
                # Make absolutely certain all ranks agree on the exact same eval score
                eval_tensor = torch.tensor([eval_reward], dtype=torch.float32, device=device)
                dist.all_reduce(eval_tensor, op=dist.ReduceOp.SUM)
                agreed_eval_reward = (eval_tensor / world_size).item()
                
                if rank == 0:
                    print(f"  [Eval] Avg Reward: {agreed_eval_reward:.2f}")
                
                # ALL ranks must determine if this is the best eval and save
                if agreed_eval_reward > best_eval_reward:
                    best_eval_reward = agreed_eval_reward
                    if rank == 0:
                        print(f"  [Best Eval] New best eval model saved!")
                    trainer.save_checkpoint('best_eval.pt')
            
            # === Checkpoint saving ===
            # ALL ranks must compute the average reward to decide whether to save
            # This is critical because save_checkpoint contains a dist.barrier()
            avg_reward_100 = sum(recent_rewards) / len(recent_rewards) if recent_rewards else float('-inf')
            
            # Make sure all ranks agree exactly on the average
            if args.distributed:
                avg_tensor = torch.tensor([avg_reward_100], dtype=torch.float32, device=device)
                dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
                avg_reward_100 = (avg_tensor / world_size).item()
            
            save_best = False
            if len(recent_rewards) >= 50 and avg_reward_100 > best_train_reward:
                best_train_reward = avg_reward_100
                save_best = True
                if rank == 0:
                    print(f"  [Best Train] New best model saved! Avg100: {avg_reward_100:.2f}")
            
            if save_best:
                trainer.save_checkpoint('best.pt')
            
            if (episode + 1) % args.save_interval == 0:
                trainer.save_checkpoint(f'checkpoint_{episode + 1}.pt')
        
        # === Training complete ===
        if rank == 0:
            elapsed = time.time() - start_time
            print("=" * 60)
            print("Training complete!")
            print(f"Total episodes: {num_episodes}")
            print(f"Total steps: {total_steps:,}")
            print(f"Total time: {elapsed:.1f}s")
            print(f"Best eval reward: {best_eval_reward:.2f}")
            print(f"Best train reward (Avg100): {best_train_reward:.2f}")
            print("=" * 60)
            
        trainer.save_checkpoint('final.pt')
        
        if rank == 0:
            print(f"Final model saved to: {getattr(config, 'checkpoint', type('obj', (object,), {'checkpoint_dir': 'checkpoints'})).checkpoint_dir}/final.pt")
            
            if csv_file:
                csv_file.close()
                print(f"CSV logs saved to: {csv_path}")
            
    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        pass  # cleanup_distributed called by wrapper


def main():
    args = parse_args()
    
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    
    if args.distributed:
        gpus = [int(x) for x in args.gpus.split(',')] if args.gpus else list(range(torch.cuda.device_count()))
        world_size = len(gpus)
        
        print(f"Launching distributed training on GPUs: {gpus}")
        spawn_distributed_training(
            train_distributed,
            world_size=world_size,
            args=(args,)
        )
    else:
        train_single_gpu(args)


if __name__ == '__main__':
    main()
