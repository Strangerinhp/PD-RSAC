"""
Episode collector for GPU-native RL training.

Collects experience from environment using the agent.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time

from ..simulator.environment import GPUEnvironment, BatchedGPUEnvironment, EnvInfo
from ..features.replay_buffer import GPUReplayBuffer
from ..features.builder import FeatureBuilder
from .smart_assignment import HybridAssignment, AssignmentResult


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode_id: int = 0
    total_reward: float = 0.0
    steps: int = 0
    trips_served: int = 0
    trips_dropped: int = 0
    trips_loaded: int = 0  # Total trips that appeared this episode
    avg_soc: float = 0.0
    revenue: float = 0.0
    driving_cost: float = 0.0
    energy_cost: float = 0.0
    profit: float = 0.0
    collection_time: float = 0.0
    # Action distribution tracking
    action_counts: Dict[str, int] = field(default_factory=lambda: {'idle': 0, 'serve': 0, 'charge': 0, 'reposition': 0})
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_id': self.episode_id,
            'total_reward': self.total_reward,
            'steps': self.steps,
            'trips_served': self.trips_served,
            'trips_dropped': self.trips_dropped,
            'trips_loaded': self.trips_loaded,
            'avg_soc': self.avg_soc,
            'revenue': self.revenue,
            'driving_cost': self.driving_cost,
            'energy_cost': self.energy_cost,
            'profit': self.profit,
            'collection_time': self.collection_time
        }


@dataclass
class CollectionMetrics:
    """Metrics for experience collection."""
    episodes_collected: int = 0
    total_steps: int = 0
    total_time: float = 0.0
    steps_per_second: float = 0.0
    avg_episode_reward: float = 0.0
    avg_episode_length: float = 0.0
    episode_stats: List[EpisodeStats] = field(default_factory=list)


class EpisodeCollector:
    """
    Collects episodes from environment using agent policy.
    
    Features:
    - Batched collection for efficiency
    - GPU-native experience storage
    - Integration with replay buffer
    - Smart assignment (auction-based) for ~95% MILP optimality
    """
    
    def __init__(
        self,
        env: GPUEnvironment,
        replay_buffer: GPUReplayBuffer,
        device: str = "cuda",
        use_smart_assignment: bool = True  # Enable auction-based assignment
    ):
        self.env = env
        self.replay_buffer = replay_buffer
        self.device = torch.device(device)
        self.use_smart_assignment = use_smart_assignment
        
        self.total_episodes = 0
        self.total_steps = 0
        
        # Get dimensions for state conversion
        self._vehicle_feature_dim = env._vehicle_feature_dim
        self._hex_feature_dim = env._hex_feature_dim
        self._context_dim = env._context_dim
        self._num_vehicles = env.num_vehicles
        self._num_hexes = env.num_hexes
        
        # Initialize smart assignment module
        if use_smart_assignment:
            self._assigner = HybridAssignment(
                num_vehicles=env.num_vehicles,
                num_hexes=env.num_hexes,
                num_stations=50,
                device=device
            )
        
        # Initialize feature builder for EnhancedActor
        self._feature_builder = FeatureBuilder(
            hex_grid=env.hex_grid,
            num_vehicles=env.num_vehicles,
            device=device
        )
    
    def _get_reposition_mask(self, vehicle_hex_ids: torch.Tensor) -> torch.Tensor:
        """Create reposition mask: restrict targets to local neighborhood (paper Eq 3).
        
        Args:
            vehicle_hex_ids: [num_vehicles] current positions
            
        Returns:
            reposition_mask: [num_vehicles, num_hexes] True for valid hexes
        """
        num_vehicles = vehicle_hex_ids.shape[0]
        
        # Try to use distance matrix if available to restrict reposition targets
        # The paper restricts to N(h_{i,t}), which we approximate as hexes within max_pickup_distance
        if hasattr(self.env, 'hex_grid') and hasattr(self.env.hex_grid, 'distance_matrix') and self.env.hex_grid.distance_matrix is not None:
            try:
                # Use same distance threshold as max_pickup_distance (typically 5.0 km)
                threshold = getattr(self.env, 'max_pickup_distance', 5.0)
                
                # Most memory efficient approach: Create target tensor and use get_distances_batch
                # This avoids slicing a 1000x3985 float tensor (~15MB per step per environment)
                all_hexes = torch.arange(self._num_hexes, device=self.device).unsqueeze(0).expand(num_vehicles, -1)
                veh_hexes_expanded = vehicle_hex_ids.unsqueeze(1).expand(-1, self._num_hexes)
                
                # Memory efficient batch distance calculation
                veh_distances = self.env.hex_grid.distance_matrix.get_distances_batch(
                    veh_hexes_expanded.reshape(-1), 
                    all_hexes.reshape(-1)
                ).view(num_vehicles, self._num_hexes)
                
                reposition_mask = veh_distances <= threshold
                
                # Prevent repositioning to the exact same hex (force them to use IDLE action instead)
                # This explicitly forces the network to learn the difference between IDLE and REPOSITION
                reposition_mask.scatter_(1, vehicle_hex_ids.unsqueeze(1), False)
                return reposition_mask
                
            except AttributeError:
                # Fallback if get_distances_batch is missing, though less memory efficient
                distance_matrix = self.env.hex_grid.distance_matrix._distances
                if distance_matrix is not None:
                    veh_distances = distance_matrix[vehicle_hex_ids]
                    threshold = getattr(self.env, 'max_pickup_distance', 5.0)
                    reposition_mask = veh_distances <= threshold
                    reposition_mask.scatter_(1, vehicle_hex_ids.unsqueeze(1), False)
                    return reposition_mask
                
        # Fallback if no distance matrix (should not happen in real training run)
        reposition_mask = torch.ones(num_vehicles, self._num_hexes, 
                                     dtype=torch.bool, device=self.device)
        return reposition_mask
    
    def _flat_to_dict_state(self, flat_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert flat state tensor to dict format for replay buffer."""
        vehicle_size = self._num_vehicles * self._vehicle_feature_dim
        hex_size = self._num_hexes * self._hex_feature_dim
        
        vehicle_features = flat_state[:vehicle_size].view(self._num_vehicles, self._vehicle_feature_dim)
        hex_features = flat_state[vehicle_size:vehicle_size + hex_size].view(self._num_hexes, self._hex_feature_dim)
        context_features = flat_state[vehicle_size + hex_size:]
        
        return {
            'vehicle': vehicle_features,
            'hex': hex_features,
            'context': context_features
        }
    
    def collect_episode(
        self,
        agent: Any,
        exploration_noise: float = 0.1,
        seed: Optional[int] = None,
        render: bool = False,
        temperature: float = 1.0
    ) -> EpisodeStats:
        """
        Collect a single episode of experience.
        
        Args:
            agent: RL agent with select_action method
            exploration_noise: Noise for exploration
            seed: Random seed for episode
            render: Whether to render environment
            temperature: Temperature for annealed softmax (paper Eq. 16)
            
        Returns:
            Episode statistics
        """
        start_time = time.time()
        
        state = self.env.reset(seed=seed)
        
        stats = EpisodeStats(episode_id=self.total_episodes)
        done = False
        step_count = 0
        
        while not done:
            # Temperature annealing within episode (optional - can also be done across episodes)
            # Start warmer, cool down as episode progresses
            step_temperature = max(0.5, temperature * (1.0 - 0.3 * step_count / self.env.episode_steps))
            
            with torch.no_grad():
                # Use smart assignment if enabled (~95% MILP optimality)
                if self.use_smart_assignment and hasattr(self, '_assigner'):
                    action_type, reposition_target, selected_trip = self._select_action_smart(
                        agent, state, temperature=step_temperature
                    )
                else:
                    action_type, reposition_target, selected_trip = self._select_action(
                        agent, state, exploration_noise, temperature=step_temperature
                    )
            
            # Track action distribution (only for AVAILABLE vehicles)
            # BUSY vehicles have action=-1 from _select_action
            # OPTIMIZED: Use tensor operations instead of Python loop
            action_names = ['idle', 'serve', 'charge', 'reposition']
            available_mask = self.env.get_available_actions()
            is_available = available_mask.any(dim=1)  # [num_vehicles]
            
            # Vectorized action counting: mask invalid actions, then bincount
            action_flat = action_type.view(-1)
            valid_actions = action_flat[(action_flat >= 0) & (action_flat < 4) & is_available]
            if len(valid_actions) > 0:
                counts = torch.bincount(valid_actions, minlength=4)
                for i, name in enumerate(action_names):
                    stats.action_counts[name] += counts[i].item()
            
            step_count += 1
            
            # Use step_with_preferences if agent has EnhancedActor (serve_scores, charge_scores)
            if hasattr(agent, 'use_enhanced_actor') and agent.use_enhanced_actor:
                # Build trip and station features for actor
                with torch.no_grad():
                    # Build features from environment state
                    actor_module = agent.actor.module if hasattr(agent.actor, 'module') else agent.actor
                    trip_features, trip_mask = self._feature_builder.build_trip_features(
                        self.env.trip_state,
                        self.env.fleet_state,
                        max_trips=actor_module.max_trips  # From Actor config (use parameter!)
                    )
                    station_features, station_mask = self._feature_builder.build_station_features(
                        self.env.station_state,
                        self.env.fleet_state,
                        current_step=step_count
                    )
                    
                    # Add batch dimension
                    trip_features = trip_features.unsqueeze(0)  # [1, max_trips, 8]
                    trip_mask = trip_mask.unsqueeze(0)  # [1, max_trips]
                    station_features = station_features.unsqueeze(0)  # [1, num_stations, 6]
                    station_mask = station_mask.unsqueeze(0)  # [1, num_stations]
                    
                    # Get serve/charge scores from actor with features
                    state_input = state.unsqueeze(0) if state.dim() == 1 else state
                    actor_output = agent.actor(
                        state_input,
                        trip_features=trip_features,
                        trip_mask=trip_mask,
                        station_features=station_features,
                        station_mask=station_mask
                    )
                    serve_scores = actor_output.serve_scores if actor_output.serve_scores is not None else None
                    charge_scores = actor_output.charge_scores if actor_output.charge_scores is not None else None
                
                next_state, reward, done_tensor, info = self.env.step_with_preferences(
                    action_type, reposition_target, serve_scores, charge_scores
                )
            else:
                next_state, reward, done_tensor, info = self.env.step(
                    action_type, reposition_target, selected_trip  # NEW: pass selected_trip
                )
            done = done_tensor.item()
            
            action = self._encode_action(action_type, reposition_target)
            
            # Convert flat states to dict format for replay buffer
            state_dict = self._flat_to_dict_state(state)
            next_state_dict = self._flat_to_dict_state(next_state)
            
            # Extract assignment info from EnvInfo
            serve_assignments = getattr(info, 'serve_assignments', None)
            charge_assignments = getattr(info, 'charge_assignments', None)
            
            # Store in replay buffer with assignment info
            reward_scalar = reward.item() if isinstance(reward, torch.Tensor) else reward
            self.replay_buffer.push(
                state=state_dict,
                action=action,
                reward=reward_scalar,
                next_state=next_state_dict,
                done=done,
                serve_assignments=serve_assignments,
                charge_assignments=charge_assignments,
            )
            
            state = next_state
            stats.total_reward += reward_scalar
            stats.steps += 1
            
            if render:
                self.env.render()
        
        stats.trips_served = info.trips_served
        stats.trips_dropped = info.trips_dropped
        stats.trips_loaded = info.trips_loaded
        stats.avg_soc = info.avg_soc
        stats.revenue = info.revenue
        stats.driving_cost = getattr(info, 'driving_cost', 0.0)
        stats.energy_cost = info.energy_cost
        stats.profit = stats.revenue - stats.driving_cost - stats.energy_cost
        stats.collection_time = time.time() - start_time
        
        self.total_episodes += 1
        self.total_steps += stats.steps
        
        return stats
    
    def collect_episodes(
        self,
        agent: Any,
        num_episodes: int,
        exploration_noise: float = 0.1,
        progress_callback: Optional[callable] = None
    ) -> CollectionMetrics:
        """
        Collect multiple episodes.
        
        Args:
            agent: RL agent
            num_episodes: Number of episodes to collect
            exploration_noise: Exploration noise
            progress_callback: Optional callback for progress updates
            
        Returns:
            Collection metrics
        """
        metrics = CollectionMetrics()
        start_time = time.time()
        
        for i in range(num_episodes):
            episode_stats = self.collect_episode(
                agent,
                exploration_noise=exploration_noise,
                seed=self.total_episodes
            )
            
            metrics.episode_stats.append(episode_stats)
            metrics.total_steps += episode_stats.steps
            
            if progress_callback:
                progress_callback(i + 1, num_episodes, episode_stats)
        
        metrics.episodes_collected = num_episodes
        metrics.total_time = time.time() - start_time
        metrics.steps_per_second = metrics.total_steps / max(metrics.total_time, 1e-6)
        
        if num_episodes > 0:
            metrics.avg_episode_reward = sum(
                s.total_reward for s in metrics.episode_stats
            ) / num_episodes
            metrics.avg_episode_length = sum(
                s.steps for s in metrics.episode_stats
            ) / num_episodes
        
        return metrics
    
    def _select_action(
        self,
        agent: Any,
        state: torch.Tensor,
        exploration_noise: float,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Select action using GCN actor with per-vehicle decisions.
        
        Returns:
            action_type, reposition_target, selected_trip
        """
        available_mask = self.env.get_available_actions()  # [num_vehicles, action_dim]
        
        # Always use GCN actor (EnhancedActor removed)
        return self._select_action_gcn(agent, state, available_mask, temperature)
    
    def _select_action_gcn(
        self,
        agent: Any,
        state: torch.Tensor,
        available_mask: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # NOW returns 3 values
        """Select action using GCN actor (per-vehicle decisions).
        
        Returns:
            action_type, reposition_target, selected_trip
        """
        with torch.no_grad():
            # Get vehicle positions (hex IDs)
            vehicle_hex_ids = self.env.fleet_state.positions.long()
            
            # Get reposition mask (vehicles can't reposition to hex they're already in)
            reposition_mask = self._get_reposition_mask(vehicle_hex_ids)
            
            # CRITICAL: Get trip mask to tell GCN which trips are actually available
            # Without this, GCN learns to select from full [0, 500] range but only
            # a subset of trips actually exist → high ServeFail rate
            trip_mask = None
            if hasattr(self.env, 'trip_state'):
                unassigned_mask = self.env.trip_state.get_unassigned_mask()
                if unassigned_mask.any():
                    # Create mask: True for available trips, False otherwise
                    actor_module = agent.actor.module if hasattr(agent.actor, 'module') else agent.actor
                    max_trips = actor_module.max_trips  # From Actor config (use parameter!)
                    trip_mask = torch.zeros(max_trips, dtype=torch.bool, device=self.device)
                    # OPTIMIZED: Directly copy first N available trips (vectorized)
                    available_indices = unassigned_mask.nonzero(as_tuple=True)[0]
                    num_to_copy = min(len(available_indices), max_trips)
                    trip_mask[:num_to_copy] = True
            
            # Use agent's select_action_gcn method
            output = agent.select_action_gcn(
                state=state,
                action_mask=available_mask,
                reposition_mask=reposition_mask,
                trip_mask=trip_mask,  # NEW: tell GCN which trips are valid
                vehicle_hex_ids=vehicle_hex_ids,
                temperature=temperature,
                deterministic=False
            )
            
            action_type = output.action_type  # [num_vehicles]
            reposition_target = output.reposition_target  # [num_vehicles]
            selected_trip = output.selected_trip  # NEW: [num_vehicles] trip selections
            
            # Squeeze batch dimension if present
            if action_type.dim() > 1:
                action_type = action_type.squeeze(0)
            if reposition_target.dim() > 1:
                reposition_target = reposition_target.squeeze(0)
            if selected_trip is not None and selected_trip.dim() > 1:
                selected_trip = selected_trip.squeeze(0)
            
            # SOC constraints per paper Section 2.2 (Eq. 3)
            socs = self.env.fleet_state.socs
            
            # SOC guard: vehicles with SOC < 20% must CHARGE
            low_soc_mask = socs < 20.0
            action_type = torch.where(
                low_soc_mask, 
                torch.tensor(2, device=self.device),  # CHARGE
                action_type
            )
            
            # High SOC vehicles (>90%) should not CHARGE
            high_soc_mask = socs > 90.0
            charging_high_soc = (action_type == 2) & high_soc_mask
            serve_available = available_mask[:, 1]
            redirect_action = torch.where(
                serve_available,
                torch.tensor(1, device=self.device),  # SERVE
                torch.tensor(0, device=self.device)   # IDLE
            )
            action_type = torch.where(charging_high_soc, redirect_action, action_type)
            
        return action_type, reposition_target, selected_trip
    
    def _encode_action(
        self,
        action_type: torch.Tensor,
        reposition_target: torch.Tensor
    ) -> torch.Tensor:
        """Encode action for replay buffer storage."""
        # Replay buffer expects [num_vehicles, action_dim] with long dtype
        action = torch.zeros(self._num_vehicles, 2, dtype=torch.long, device=self.device)
        action[:, 0] = action_type.long()
        action[:, 1] = reposition_target.long()
        return action


class BatchedEpisodeCollector:
    """
    Collects episodes in parallel using batched environments.
    """
    
    def __init__(
        self,
        envs: BatchedGPUEnvironment,
        replay_buffer: GPUReplayBuffer,
        device: str = "cuda"
    ):
        self.envs = envs
        self.replay_buffer = replay_buffer
        self.device = torch.device(device)
        self.num_envs = envs.num_envs
        
        # Get dimensions from first environment
        first_env = envs.envs[0]
        self._vehicle_feature_dim = first_env._vehicle_feature_dim
        self._hex_feature_dim = first_env._hex_feature_dim
        self._context_dim = first_env._context_dim
        self._num_vehicles = first_env.num_vehicles
        self._num_hexes = first_env.num_hexes
    
    def _flat_to_dict_state(self, flat_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert flat state tensor to dict format for replay buffer."""
        vehicle_size = self._num_vehicles * self._vehicle_feature_dim
        hex_size = self._num_hexes * self._hex_feature_dim
        
        vehicle_features = flat_state[:vehicle_size].view(self._num_vehicles, self._vehicle_feature_dim)
        hex_features = flat_state[vehicle_size:vehicle_size + hex_size].view(self._num_hexes, self._hex_feature_dim)
        context_features = flat_state[vehicle_size + hex_size:]
        
        return {
            'vehicle': vehicle_features,
            'hex': hex_features,
            'context': context_features
        }
    
    def collect_steps(
        self,
        agent: Any,
        num_steps: int,
        exploration_noise: float = 0.1
    ) -> Dict[str, float]:
        """
        Collect fixed number of steps from all environments.
        
        Args:
            agent: RL agent
            num_steps: Steps to collect per environment
            exploration_noise: Exploration noise
            
        Returns:
            Collection statistics
        """
        start_time = time.time()
        states = self.envs.reset()
        
        total_rewards = torch.zeros(self.num_envs, device=self.device)
        episode_counts = torch.zeros(self.num_envs, device=self.device)
        
        for step in range(num_steps):
            with torch.no_grad():
                actions = self._select_batched_actions(
                    agent, states, exploration_noise
                )
            
            action_types = actions[:, :, 0].long()
            reposition_targets = (actions[:, :, 1] * self.envs.envs[0].num_hexes).long()
            
            next_states, rewards, dones, infos = self.envs.step(
                action_types, reposition_targets
            )
            
            self._store_transitions(
                states, actions, rewards, next_states, dones
            )
            
            total_rewards += rewards
            episode_counts += dones.float()
            
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    next_states[i] = self.envs.envs[i].reset()
            
            states = next_states
        
        elapsed = time.time() - start_time
        total_steps = num_steps * self.num_envs
        
        return {
            'total_steps': total_steps,
            'steps_per_second': total_steps / elapsed,
            'avg_reward': total_rewards.mean().item(),
            'episodes_completed': episode_counts.sum().item(),
            'collection_time': elapsed
        }
    
    def _select_batched_actions(
        self,
        agent: Any,
        states: torch.Tensor,
        exploration_noise: float
    ) -> torch.Tensor:
        """Select actions for all environments in batch."""
        num_envs = states.shape[0]
        num_vehicles = self.envs.envs[0].num_vehicles
        
        actions = torch.zeros(num_envs, num_vehicles, 2, device=self.device)
        
        for i in range(num_envs):
            if hasattr(agent, 'select_action'):
                output = agent.select_action(states[i], deterministic=False)
                # Handle SACOutput
                if hasattr(output, 'action_type'):
                    action_type = output.action_type
                    reposition_target = output.reposition_target
                    actions[i, :, 0] = action_type.float() if action_type.dim() == 1 else action_type.argmax(dim=-1).float()
                    actions[i, :, 1] = reposition_target.float() / self.envs.envs[0].num_hexes if reposition_target.dim() == 1 else reposition_target.argmax(dim=-1).float() / self.envs.envs[0].num_hexes
                elif isinstance(output, tuple):
                    action_probs, reposition_probs = output
                    actions[i, :, 0] = action_probs.argmax(dim=-1).float()
                    actions[i, :, 1] = reposition_probs if reposition_probs.dim() == 1 else reposition_probs.argmax(dim=-1).float() / self.envs.envs[0].num_hexes
                else:
                    action_probs = output
                    reposition_probs = torch.rand(num_vehicles, device=self.device)
                    actions[i, :, 0] = action_probs.argmax(dim=-1).float()
                    actions[i, :, 1] = reposition_probs
            else:
                actions[i, :, 0] = torch.randint(0, 4, (num_vehicles,), device=self.device).float()
                actions[i, :, 1] = torch.rand(num_vehicles, device=self.device)
        
        return actions
    
    def _store_transitions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ):
        """Store transitions from all environments."""
        for i in range(self.num_envs):
            # Convert tensor to dict format expected by replay buffer push
            state_dict = self._flat_to_dict_state(states[i])
            next_state_dict = self._flat_to_dict_state(next_states[i])
            
            # Convert action tensor to long dtype for replay buffer
            action_tensor = actions[i].long() if actions[i].dtype != torch.long else actions[i]
            
            self.replay_buffer.push(
                state=state_dict,
                action=action_tensor.view(self._num_vehicles, 2),
                reward=rewards[i].item() if isinstance(rewards[i], torch.Tensor) else rewards[i],
                next_state=next_state_dict,
                done=dones[i].item() if isinstance(dones[i], torch.Tensor) else dones[i]
            )
