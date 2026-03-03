#!/usr/bin/env python3
"""
Evaluate trained EV Fleet RL Agent on Real Data for a Full Time Span

Usage:
    python evaluate.py \
        --checkpoint checkpoints/best.pt \
        --real-data data/nyc_full/trips_processed.parquet \
        --start-date 2009-01-15 \
        --end-date 2009-01-15 \
        --episode-duration-hours 24
"""

import argparse
import sys
import time
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import ConfigLoader, Config
from gpu_core.networks.sac import SACAgent
from gpu_core.spatial.grid import HexGrid
from gpu_core.simulator.environment import GPUEnvironment
from gpu_core.simulator.environment_v2 import GPUEnvironmentV2
from gpu_core.data.real_trip_loader import RealTripLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate trained EV Fleet RL Agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides checkpoint config)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of evaluation passes (mostly 1 for deterministic evaluation)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic actions (argmax)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Save results to JSON file')
    
    # Environment + Data args similar to heuristic_matching
    parser.add_argument('--num-vehicles', type=int, default=None,
                        help='Number of vehicles in fleet')
    parser.add_argument('--num-hexes', type=int, default=None,
                        help='Number of hexagons in grid')
    parser.add_argument('--episode-duration-hours', type=float, default=None,
                        help='Episode duration in hours (e.g., 720 for a 30-day month)')
    parser.add_argument('--env-v2', action='store_true', default=False,
                        help='Use GPUEnvironmentV2')
    
    parser.add_argument('--real-data', type=str, default=None,
                        help='Path to real trip data parquet file')
    parser.add_argument('--trip-sample', type=float, default=1.0,
                        help='Sample ratio for trip data (0.0-1.0)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Filter trips from this date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Filter trips until this date (YYYY-MM-DD)')
    parser.add_argument('--target-h3-resolution', type=int, default=None,
                        help='Target H3 resolution')
    parser.add_argument('--max-hex-count', type=int, default=None,
                        help='Maximum number of hexes')
    
    return parser.parse_args()


def create_config(args, checkpoint_path: str) -> Config:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = checkpoint.get('config', {})
    
    if args.config:
        config = ConfigLoader.from_yaml(args.config)
    else:
        # Generate default config
        config = Config()
        
    if args.num_vehicles is not None:
        config.environment.num_vehicles = args.num_vehicles
    elif 'environment' in ckpt_config and 'num_vehicles' in ckpt_config['environment']:
        config.environment.num_vehicles = ckpt_config['environment']['num_vehicles']
        
    if args.num_hexes is not None:
        config.environment.num_hexes = args.num_hexes
    elif 'environment' in ckpt_config and 'num_hexes' in ckpt_config['environment']:
        config.environment.num_hexes = ckpt_config['environment']['num_hexes']
        
    if args.start_date and args.end_date:
        try:
            from datetime import datetime
            start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
            days = max(1, (end_dt - start_dt).days + 1)
            calculated_duration = float(days * 24)
            config.episode.duration_hours = calculated_duration
            print(f"[Auto-Detect] Set episode duration to {config.episode.duration_hours} hours ({days} days)")
            if args.episode_duration_hours is not None and args.episode_duration_hours != calculated_duration:
                print(f"[Warning] Overriding requested --episode-duration-hours {args.episode_duration_hours} with calculated duration {calculated_duration}")
        except Exception as e:
            print(f"[Auto-Detect] Failed to parse dates for duration: {e}")
            if args.episode_duration_hours is not None:
                config.episode.duration_hours = args.episode_duration_hours
            elif 'episode' in ckpt_config and 'duration_hours' in ckpt_config['episode']:
                config.episode.duration_hours = ckpt_config['episode']['duration_hours']
    elif args.episode_duration_hours is not None:
        config.episode.duration_hours = args.episode_duration_hours
    elif 'episode' in ckpt_config and 'duration_hours' in ckpt_config['episode']:
        config.episode.duration_hours = ckpt_config['episode']['duration_hours']
        
    return config


def create_environment(config: Config, device: torch.device, args, trip_loader: Optional[RealTripLoader] = None):
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
        grid_size = int(num_hexes ** 0.5) + 1
        fake_hex_ids = [f"hex_{i}" for i in range(num_hexes)]
        hex_grid._hex_ids = fake_hex_ids
        hex_grid._hex_to_idx = {h: i for i, h in enumerate(fake_hex_ids)}
        base_lat, base_lon = 40.7128, -74.0060
        lat_per_km, lon_per_km = 0.009, 0.012
        lats = torch.zeros(num_hexes, device=device)
        lons = torch.zeros(num_hexes, device=device)
        for i in range(num_hexes):
            row = i // grid_size
            col = i % grid_size
            lats[i] = base_lat + row * 0.5 * lat_per_km
            lons[i] = base_lon + col * 0.5 * lon_per_km
        hex_grid._latitudes = lats
        hex_grid._longitudes = lons
        hex_grid._initialized = True
        lat_diff = lats.unsqueeze(1) - lats.unsqueeze(0)
        lon_diff = lons.unsqueeze(1) - lons.unsqueeze(0)
        lat_km = lat_diff / lat_per_km
        lon_km = lon_diff / lon_per_km
        distances = torch.sqrt(lat_km**2 + lon_km**2)
        distances.fill_diagonal_(0)
        hex_grid.distance_matrix._distances = distances
        hex_grid.distance_matrix._num_hexes = num_hexes
    
    use_v2 = args.env_v2
    if not use_v2 and hasattr(args, 'checkpoint') and args.checkpoint:
        try:
            chkpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if 'agent' in chkpt:
                for k, v in chkpt['agent'].items():
                    if 'actor.gcn' in k and 'weight' in k and v.dim() == 2:
                        if v.shape[1] == 5:
                            use_v2 = True
                            print("[Auto-Detect] Model has hex_feature_dim=5. Using GPUEnvironmentV2.")
                        elif v.shape[1] == 8:
                            print("[Auto-Detect] Model has hex_feature_dim=8. Using GPUEnvironment(V1).")
                        break
        except Exception as e:
            print(f"[Auto-Detect] Failed to infer env version: {e}")

    if use_v2:
        env = GPUEnvironmentV2(
            config=config,
            hex_grid=hex_grid,
            trip_loader=trip_loader,
            device=device_str
        )
    else:
        env = GPUEnvironment(
            config=config,
            hex_grid=hex_grid,
            trip_loader=trip_loader,
            device=device_str
        )
    
    return env


def load_agent(checkpoint_path: str, config: Config, env, device: torch.device) -> SACAgent:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    env_config = config.environment
    entropy_config = config.entropy
    
    # The model was trained with specific feature dimensions
    vehicle_feature_dim = getattr(env, '_vehicle_feature_dim', 16)
    hex_feature_dim = getattr(env, '_hex_feature_dim', 5)
    context_dim = getattr(env, '_context_dim', 9)
    
    state_dim = (
        env_config.num_vehicles * vehicle_feature_dim +
        env_config.num_hexes * hex_feature_dim +
        context_dim
    )
    
    action_dim = 4
    target_entropy = -entropy_config.target_entropy_ratio * torch.log(torch.tensor(float(action_dim))).item()
    
    # We must match the same exact shapes the agent trained with
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_hexes=env_config.num_hexes,
        actor_hidden_dims=[128, 128], # Matching train.py GCN specs
        critic_hidden_dims=[256, 256],
        gamma=config.training.gamma,
        tau=config.training.tau,
        alpha=entropy_config.initial_alpha,
        auto_alpha=entropy_config.auto_alpha,
        target_entropy=target_entropy,
        device=str(device),
        num_vehicles=env_config.num_vehicles,
        vehicle_feature_dim=vehicle_feature_dim,
        hex_feature_dim=hex_feature_dim,
        context_dim=context_dim,
        max_trips=env_config.max_trips_per_step,
        num_stations=env_config.num_stations,
        max_serve=env_config.max_serve_per_step,
        max_charge=env_config.max_charge_per_step,
    )
    
    if 'agent' in checkpoint:
        agent.load_state_dict(checkpoint['agent'])
        
    return agent.to(device)


def evaluate(args):
    print("=" * 60)
    print("EV Fleet RL Agent Evaluation (SAC Financial Analysis)")
    print("=" * 60)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    config = create_config(args, args.checkpoint)
    
    # Setup data
    trip_loader = None
    data_path = args.real_data or config.data.parquet_path
    if data_path:
        resolved_path = Path(data_path)
        print(f"\n[Real Data] Loading from: {resolved_path}")
        if not resolved_path.exists():
            print(f"[Real Data] File not found: {resolved_path}. Falling back to synthetic.")
        else:
            try:
                trip_loader = RealTripLoader(
                    parquet_path=str(resolved_path),
                    device=str(device),
                    sample_ratio=args.trip_sample,
                    target_h3_resolution=args.target_h3_resolution,
                    max_hex_count=args.max_hex_count,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
                trip_loader.load()
            except Exception as exc:
                print(f"[Real Data] Failed to load due to: {exc}. Falling back to synthetic data.")
                trip_loader = None
                
    # Create enviroment
    env = create_environment(config, device, args, trip_loader=trip_loader)
    env_name = "GPUEnvironmentV2" if isinstance(env, GPUEnvironmentV2) else "GPUEnvironment V1"
    print(f"\n[Environment] Created {env_name} with {env.num_vehicles} vehicles and {env.num_hexes} hexes")
    print(f"[Environment] Time span: {config.episode.duration_hours} hours")
    
    # Create agent
    print(f"Loading checkpoint: {args.checkpoint}")
    agent = load_agent(args.checkpoint, config, env, device)
    
    # Ensure adjacency matrix is generated and set for GCN
    if hasattr(env, '_adjacency_matrix') and env._adjacency_matrix is not None:
        agent.set_adjacency(env._adjacency_matrix)
        print(f"[GCN] Adjacency matrix loaded into actor")
    else:
        # GPUEnvironment (v1) doesn't have _compute_adjacency_matrix
        print("[GCN] Computing missing adjacency matrix...")
        distance_matrix = env.hex_grid.distance_matrix._distances
        adjacency_threshold = 3.0
        adj = (distance_matrix < adjacency_threshold).float()
        adj = adj + torch.eye(env.num_hexes, device=device)
        adj = torch.clamp(adj, 0, 1)
        
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        adj_matrix = D_inv_sqrt @ adj @ D_inv_sqrt
        agent.set_adjacency(adj_matrix)
        print(f"[GCN] Manual adjacency matrix loaded into actor")
        
    agent.eval()
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print("=" * 60)
    
    all_results = []
    
    for ep in range(args.episodes):
        print(f"\nEvaluating Episode {ep + 1}/{args.episodes}...")
        
        action_counts = defaultdict(int)
        hourly_metrics = defaultdict(lambda: {'revenue': 0.0, 'driving_cost': 0.0, 'energy_cost': 0.0})
        # Track daily metrics if duration is long
        daily_metrics = defaultdict(lambda: {'revenue': 0.0, 'driving_cost': 0.0, 'energy_cost': 0.0})
        
        state = env.reset()
        max_steps = config.episode.steps_per_episode
        step_duration_minutes = config.episode.step_duration_minutes
        
        prev_revenue = 0.0
        prev_driving_cost = 0.0
        prev_energy_cost = 0.0
        
        # for env_v2 metric tracking (separate from cumulative accumulators)
        prev_env_revenue = 0.0
        prev_env_driving_cost = 0.0  # FIX: was prev_driving_cost, collided with accumulator
        prev_env_energy_cost = 0.0
        prev_env_reward = 0.0
        
        done = False
        step = 0
        start_time = time.time()
        
        while not done and step < max_steps:
            # Mask actions if environment supports it
            action_mask = env.get_available_actions() if hasattr(env, 'get_available_actions') else None
            
            # Get available (non-busy) vehicle mask BEFORE step for accurate action counting
            if hasattr(env, 'fleet_state') and env.fleet_state is not None:
                available_for_count = env.fleet_state.get_available_mask(env.current_step)
            else:
                available_for_count = None
            
            with torch.no_grad():
                output = agent.select_action(state, action_mask=action_mask, deterministic=args.deterministic)
                action_type = output.action_type
                reposition_target = output.reposition_target
                selected_trip = output.selected_trip
                
                # Check formatting from networks 
                if action_type.dim() > 1:
                    action_type = action_type.squeeze(0)
                    reposition_target = reposition_target.squeeze(0)
                    if selected_trip is not None:
                        selected_trip = selected_trip.squeeze(0)
                
                if action_type.numel() == 1:
                    action_type = action_type.expand(env.num_vehicles)
                    reposition_target = reposition_target.expand(env.num_vehicles)
                    if selected_trip is not None and selected_trip.numel() == 1:
                        selected_trip = selected_trip.expand(env.num_vehicles)
            
            # Count actions only for vehicles that are actually available (not busy).
            # Busy vehicles' chosen actions are suppressed by available_mask in env.step(),
            # so counting them inflates IDLE artificially.
            action_names = ['IDLE', 'SERVE', 'CHARGE', 'REPOSITION']
            if available_for_count is not None:
                available_actions = action_type[available_for_count].cpu().numpy()
            else:
                available_actions = action_type.cpu().numpy()
            for action_id in available_actions:
                if 0 <= action_id < 4:
                    action_counts[action_names[action_id]] += 1
                    
            next_state, reward, done_tensor, info = env.step(action_type, reposition_target, selected_trip)
            done = done_tensor.item()
            state = next_state
            
            # Accumulate step metrics via delta tracking
            step_revenue = 0.0
            step_serve_cost = 0.0
            step_charge_cost = 0.0
            
            if isinstance(env, GPUEnvironmentV2):
                # Using modular GPUEnvironmentV2
                metrics = env.get_metrics()
                
                # Extract cumulative metrics from env and calculate step delta
                cur_rev = metrics.get('revenue', 0.0)
                cur_drive = metrics.get('driving_cost', 0.0)
                cur_energy = metrics.get('energy_cost', 0.0)
                cur_reward = metrics.get('episode_reward', 0.0)
                
                step_revenue = cur_rev - prev_env_revenue
                step_serve_cost = cur_drive - prev_env_driving_cost  # FIX: use dedicated V2 tracker
                step_charge_cost = cur_energy - prev_env_energy_cost
                
                prev_env_revenue = cur_rev
                prev_env_driving_cost = cur_drive  # FIX: update dedicated V2 tracker
                prev_env_energy_cost = cur_energy
                prev_env_reward = cur_reward
                
            elif isinstance(info, list):
                # Using GPUEnvironment (V1) where info is a list
                for step_info in info:
                    if 'financial' in step_info:
                        step_revenue += step_info['financial'].get('revenue', 0.0)
                        step_serve_cost += step_info['financial'].get('serve_cost', 0.0)
                        step_charge_cost += step_info['financial'].get('charge_cost', 0.0)
                    elif 'reward_components' in step_info:
                        step_revenue += step_info['reward_components'].get('Revenue', 0.0)
                        step_serve_cost += step_info['reward_components'].get('ServeDrive', 0.0)
                        step_charge_cost += step_info['reward_components'].get('Charge', 0.0)
            elif isinstance(info, dict):
                # Using GPUEnvironment (V1) where info is a dict
                if 'financial' in info:
                    step_revenue += info['financial'].get('revenue', 0.0)
                    step_serve_cost += info['financial'].get('serve_cost', 0.0)
                    step_charge_cost += info['financial'].get('charge_cost', 0.0)
                elif 'reward_components' in info:
                    step_revenue += info['reward_components'].get('Revenue', 0.0)
                    step_serve_cost += info['reward_components'].get('ServeDrive', 0.0)
                    step_charge_cost += info['reward_components'].get('Charge', 0.0)
                    
            current_hour = int((step * step_duration_minutes) // 60) % 24
            hourly_metrics[current_hour]['revenue'] += step_revenue
            hourly_metrics[current_hour]['driving_cost'] += step_serve_cost
            hourly_metrics[current_hour]['energy_cost'] += step_charge_cost
            
            # Accumulate daily metrics
            current_day = int((step * step_duration_minutes) // (60 * 24))
            daily_metrics[current_day]['revenue'] += step_revenue
            daily_metrics[current_day]['driving_cost'] += step_serve_cost
            daily_metrics[current_day]['energy_cost'] += step_charge_cost
            
            prev_revenue += step_revenue
            prev_driving_cost += step_serve_cost
            prev_energy_cost += step_charge_cost
            
            step += 1
            if step % 50 == 0:
                print(f"  Step {step}/{max_steps}: Revenue=${prev_revenue:.2f}, "
                      f"Driving Cost=${prev_driving_cost:.2f}, Charge Cost=${prev_energy_cost:.2f}")

        elapsed_time = time.time() - start_time
        
        # net_profit computed after V2 override below (or using step-accumulators for V1)
        net_profit = prev_revenue - prev_driving_cost - prev_energy_cost
        
        # Format hourly metrics
        hourly_results = {}
        for hour in sorted(hourly_metrics.keys()):
            hour_data = hourly_metrics[hour]
            hourly_net_profit = hour_data['revenue'] - hour_data['driving_cost'] - hour_data['energy_cost']
            hourly_results[int(hour)] = {
                'revenue': float(hour_data['revenue']),
                'driving_cost': float(hour_data['driving_cost']),
                'energy_cost': float(hour_data['energy_cost']),
                'net_profit': float(hourly_net_profit)
            }
            
        # Format daily metrics
        daily_results = {}
        for day in sorted(daily_metrics.keys()):
            day_data = daily_metrics[day]
            daily_net_profit = day_data['revenue'] - day_data['driving_cost'] - day_data['energy_cost']
            daily_results[int(day)] = {
                'revenue': float(day_data['revenue']),
                'driving_cost': float(day_data['driving_cost']),
                'energy_cost': float(day_data['energy_cost']),
                'net_profit': float(daily_net_profit)
            }
        if isinstance(env, GPUEnvironmentV2):
            # Fetch final cumulative metrics directly from EnvV2
            metrics = env.get_metrics()
            
            # Override step-accumulator totals with authoritative env totals
            prev_revenue = metrics.get('revenue', 0.0)
            prev_driving_cost = metrics.get('driving_cost', 0.0)
            prev_energy_cost = metrics.get('energy_cost', 0.0)
            
            # FIX: Recompute net_profit AFTER the V2 override so it uses the correct finals
            net_profit = prev_revenue - prev_driving_cost - prev_energy_cost
            
            trips_served = metrics.get('trips_served', 0)
            trips_loaded = metrics.get('trips_loaded', 0)
            trips_dropped = metrics.get('trips_dropped', 0)
            avg_soc = metrics.get('avg_soc', 0.0)
            
            # Diagnostic: log reward component weights so the user can tune behavior
            if hasattr(env, '_reward_computer'):
                rc = env._reward_computer
                print(f"[Diagnostics] RewardComputer penalties: "
                      f"reposition={getattr(rc, 'reposition_penalty', 'N/A')}, "
                      f"idle={getattr(rc, 'idle_penalty', 'N/A')}, "
                      f"charge_high_soc={getattr(rc, 'high_soc_charge_penalty', 'N/A')}")
        else:
            trips_served = env.total_trips_served if hasattr(env, 'total_trips_served') else (
                env.trip_state.total_trips_served if hasattr(env, 'trip_state') and hasattr(env.trip_state, 'total_trips_served') else 0
            )
            trips_loaded = env.total_trips_loaded if hasattr(env, 'total_trips_loaded') else (
                env.trip_state.total_trips_loaded if hasattr(env, 'trip_state') and hasattr(env.trip_state, 'total_trips_loaded') else 0
            )
            trips_dropped = env.total_trips_dropped if hasattr(env, 'total_trips_dropped') else (
                env.trip_state.total_trips_dropped if hasattr(env, 'trip_state') and hasattr(env.trip_state, 'total_trips_dropped') else 0
            )
            avg_soc = env.vehicle_socs.mean().item() if hasattr(env, 'vehicle_socs') else (
                env.fleet_state.socs.mean().item() if hasattr(env, 'fleet_state') else 0.0
            )

        service_rate = float(trips_served) / max(float(trips_loaded), 1.0)
            
        result = {
            'episode': ep,
            'total_trips_loaded': int(trips_loaded),
            'total_trips_served': int(trips_served),
            'total_trips_dropped': int(trips_dropped),
            'service_rate': float(service_rate),
            'total_revenue': float(prev_revenue),
            'total_driving_cost': float(prev_driving_cost),
            'total_charging_cost': float(prev_energy_cost),
            'net_profit': float(net_profit),
            'final_avg_soc': float(avg_soc),
            'action_counts': dict(action_counts),
            'simulation_time_seconds': float(elapsed_time),
            'steps_completed': int(step),
            'hourly_metrics': hourly_results,
            'daily_metrics': daily_results
        }
        
        print("\n--- Episode Summary ---")
        print(f" Revenue: ${result['total_revenue']:.2f}")
        print(f" Costs: ${result['total_driving_cost'] + result['total_charging_cost']:.2f} (Drive: ${result['total_driving_cost']:.2f}, Charge: ${result['total_charging_cost']:.2f})")
        print(f" Net Profit: ${result['net_profit']:.2f}")
        print(f" Trips: Served {result['total_trips_served']} / Loaded {result['total_trips_loaded']} (Rate: {result['service_rate']*100:.1f}%)")
        print(f" Action Counts: {result['action_counts']}")
        
        # If duration > 48 hours, print daily metrics to avoid flooding the console
        if 'daily_metrics' in result and result['daily_metrics'] and config.episode.duration_hours > 48.0:
            print(f"\nDaily metrics:")
            for day in sorted(result['daily_metrics'].keys()):
                day_data = result['daily_metrics'][day]
                print(f"  Day {day+1:2d}: Revenue=${day_data['revenue']:>10,.2f}, "
                      f"Net Profit=${day_data['net_profit']:>10,.2f}")
        
        all_results.append(result)
        
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")


def main():
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
