import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
import time
import os
from pathlib import Path

from ..config import TrainingConfig, CheckpointConfig, LoggingConfig
from ..networks.sac import SACAgent
from ..features.replay_buffer import GPUReplayBuffer, Transition


@dataclass
class TrainingMetrics:
    episode: int = 0
    step: int = 0
    total_reward: float = 0.0
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    alpha: float = 0.0
    entropy: float = 0.0
    q_mean: float = 0.0
    episode_length: int = 0
    fps: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


class SACTrainer:
    def __init__(
        self,
        agent: SACAgent,
        replay_buffer: GPUReplayBuffer,
        training_config: TrainingConfig,
        checkpoint_config: Optional[CheckpointConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
        device: str = 'cuda'
    ):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.config = training_config
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.logging_config = logging_config or LoggingConfig()
        self.device = torch.device(device)
        
        self.global_step = 0
        self.episode = 0
        self.best_reward = float('-inf')
        
        self._compile_if_enabled()
        self._setup_lr_schedulers()
    
    def _setup_lr_schedulers(self):
        """Setup learning rate schedulers for better convergence."""
        # Cosine annealing with warm restarts
        total_steps = getattr(self.config, 'total_steps', 1000000)
        
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.agent.actor_optimizer,
            T_0=total_steps // 10,  # Restart every 10% of training
            T_mult=2,
            eta_min=1e-6
        )
        
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.agent.critic_optimizer,
            T_0=total_steps // 10,
            T_mult=2,
            eta_min=1e-6
        )
    
    def _compile_if_enabled(self):
        use_compile = getattr(self.config, 'compile_model', False)
        if use_compile and hasattr(torch, 'compile'):
            self.agent.actor = torch.compile(self.agent.actor, mode='reduce-overhead', options={"triton.cudagraphs": False})
            self.agent.critic = torch.compile(self.agent.critic, mode='reduce-overhead', options={"triton.cudagraphs": False})
    
    def train_step(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        states = batch.states
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_states
        dones = batch.dones
        
        metrics = self.agent.update(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            # Assignment info for auxiliary loss
            serve_vehicle_idx=batch.serve_vehicle_idx,
            serve_trip_idx=batch.serve_trip_idx,
            num_served=batch.num_served,
            charge_vehicle_idx=batch.charge_vehicle_idx,
            charge_station_idx=batch.charge_station_idx,
            num_charged=batch.num_charged,
        )
        
        self.global_step += 1
        
        # Check if prioritized replay is enabled
        use_prioritized = getattr(self.config, 'use_prioritized_replay', False) or \
                          getattr(self.replay_buffer, 'prioritized', False)
        
        if use_prioritized and batch.indices is not None:
            with torch.no_grad():
                # Flatten states if dict
                if isinstance(states, dict):
                    batch_size = states['vehicle'].shape[0]
                    states_flat = torch.cat([
                        states['vehicle'].view(batch_size, -1),
                        states['hex'].view(batch_size, -1),
                        states['context']
                    ], dim=-1)
                else:
                    states_flat = states
                
                if isinstance(next_states, dict):
                    next_states_flat = torch.cat([
                        next_states['vehicle'].view(batch_size, -1),
                        next_states['hex'].view(batch_size, -1),
                        next_states['context']
                    ], dim=-1)
                else:
                    next_states_flat = next_states
                    
                # Extract per-vehicle action types
                if actions.dim() == 3:
                    per_vehicle_actions = actions[:, :, 0].long()
                else:
                    per_vehicle_actions = actions.long()

                q1, q2 = self.agent.call_critic(self.agent.critic, states_flat, per_vehicle_actions)

                # Get next actions from actor
                actor_output = self.agent.forward(next_states_flat, deterministic=False, compute_trips=False)
                if hasattr(actor_output, 'action_type'):
                    next_action = actor_output.action_type
                    next_log_prob = actor_output.action_log_prob
                else:
                    next_action, next_log_prob, _, _ = actor_output
                if next_log_prob.dim() == 2:
                    next_log_prob = next_log_prob.mean(dim=1)

                next_q1, next_q2 = self.agent.call_critic(self.agent.critic_target, next_states_flat, next_action)
                next_q = torch.min(next_q1, next_q2) - self.agent.alpha * next_log_prob
                target_q = rewards + (1 - dones.float()) * self.config.gamma * next_q
                td_error = (torch.min(q1, q2) - target_q).abs()
            
            self.replay_buffer.update_priorities(
                batch.indices,
                td_error  # Keep as tensor, not numpy
            )
        
        # Step learning rate schedulers
        if hasattr(self, 'actor_scheduler'):
            self.actor_scheduler.step()
        if hasattr(self, 'critic_scheduler'):
            self.critic_scheduler.step()
        
        # Add LR to metrics
        metrics['lr_actor'] = self.agent.actor_optimizer.param_groups[0]['lr']
        metrics['lr_critic'] = self.agent.critic_optimizer.param_groups[0]['lr']
        
        return metrics
    
    def train_episode(
        self,
        env,
        max_steps: int = 1000,
        warmup_steps: int = 0
    ) -> TrainingMetrics:
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        start_time = time.time()
        
        metrics_accum = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'alpha': 0.0,
            'q_mean': 0.0,
            'updates': 0
        }
        
        for step in range(max_steps):
            if self.global_step < warmup_steps:
                action_type = torch.randint(0, self.agent.action_dim, (1,), device=self.device)
                reposition_target = torch.randint(0, self.agent.num_hexes, (1,), device=self.device)
            else:
                output = self.agent.select_action(
                    state.unsqueeze(0) if state.dim() == 1 else state
                )
                action_type = output.action_type
                reposition_target = output.reposition_target
            
            next_state, reward, done, info = env.step(action_type.item(), reposition_target.item())
            
            transition = Transition(
                state=state,
                action=action_type,
                reward=torch.tensor([reward], device=self.device),
                next_state=next_state,
                done=torch.tensor([done], device=self.device)
            )
            self.replay_buffer.add(transition)
            
            episode_reward += reward
            episode_length += 1
            
            if self.global_step >= warmup_steps and len(self.replay_buffer) >= self.config.batch_size:
                for _ in range(self.config.gradient_steps):
                    step_metrics = self.train_step()
                    if step_metrics:
                        metrics_accum['actor_loss'] += step_metrics.get('actor_loss', 0)
                        metrics_accum['critic_loss'] += step_metrics.get('critic_loss', 0)
                        metrics_accum['alpha'] += step_metrics.get('alpha', 0)
                        metrics_accum['updates'] += 1
            
            state = next_state
            
            if done:
                break
        
        elapsed = time.time() - start_time
        fps = episode_length / elapsed if elapsed > 0 else 0
        
        num_updates = max(metrics_accum['updates'], 1)
        
        self.episode += 1
        
        return TrainingMetrics(
            episode=self.episode,
            step=self.global_step,
            total_reward=episode_reward,
            actor_loss=metrics_accum['actor_loss'] / num_updates,
            critic_loss=metrics_accum['critic_loss'] / num_updates,
            alpha=metrics_accum['alpha'] / num_updates,
            episode_length=episode_length,
            fps=fps
        )
    
    def train(
        self,
        env,
        num_episodes: int,
        max_steps_per_episode: int = 1000,
        warmup_steps: int = 1000,
        eval_interval: int = 10,
        eval_episodes: int = 5,
        callback: Optional[Callable[[TrainingMetrics], None]] = None
    ) -> list:
        all_metrics = []
        
        for ep in range(num_episodes):
            metrics = self.train_episode(
                env=env,
                max_steps=max_steps_per_episode,
                warmup_steps=warmup_steps
            )
            all_metrics.append(metrics)
            
            if callback:
                callback(metrics)
            
            if self.logging_config.log_interval > 0 and ep % self.logging_config.log_interval == 0:
                self._log_metrics(metrics)
            
            if eval_interval > 0 and ep % eval_interval == 0:
                eval_reward = self.evaluate(env, eval_episodes, max_steps_per_episode)
                metrics.extra['eval_reward'] = eval_reward
                
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    if self.checkpoint_config.save_best:
                        self.save_checkpoint('best.pt')
            
            if self.checkpoint_config.save_interval > 0 and ep % self.checkpoint_config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_{ep}.pt')
        
        return all_metrics
    
    def evaluate(
        self,
        env,
        num_episodes: int,
        max_steps: int = 1000
    ) -> float:
        self.agent.eval()
        total_reward = 0.0
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            
            for _ in range(max_steps):
                with torch.no_grad():
                    output = self.agent.select_action(
                        state.unsqueeze(0) if state.dim() == 1 else state,
                        deterministic=True
                    )
                    action_type = output.action_type
                    reposition_target = output.reposition_target
                
                next_state, reward, done, _ = env.step(action_type.item(), reposition_target.item())
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            total_reward += episode_reward
        
        self.agent.train()
        return total_reward / num_episodes
    
    def _log_metrics(self, metrics: TrainingMetrics):
        print(f"Episode {metrics.episode} | "
              f"Step {metrics.step} | "
              f"Reward: {metrics.total_reward:.2f} | "
              f"Actor Loss: {metrics.actor_loss:.4f} | "
              f"Critic Loss: {metrics.critic_loss:.4f} | "
              f"Alpha: {metrics.alpha:.4f} | "
              f"FPS: {metrics.fps:.1f}")
    
    def save_checkpoint(self, filename: str):
        path = Path(self.checkpoint_config.checkpoint_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'agent': self.agent.state_dict(),
            'global_step': self.global_step,
            'episode': self.episode,
            'best_reward': self.best_reward,
            'config': {
                'training': self.config.__dict__,
                'checkpoint': self.checkpoint_config.__dict__
            }
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        path = Path(self.checkpoint_config.checkpoint_dir) / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.agent.load_state_dict(checkpoint['agent'])
        self.global_step = checkpoint['global_step']
        self.episode = checkpoint['episode']
        self.best_reward = checkpoint.get('best_reward', float('-inf'))


class BatchedSACTrainer(SACTrainer):
    def __init__(
        self,
        agent: SACAgent,
        replay_buffer: GPUReplayBuffer,
        training_config: TrainingConfig,
        num_parallel_envs: int = 8,
        **kwargs
    ):
        super().__init__(agent, replay_buffer, training_config, **kwargs)
        self.num_parallel_envs = num_parallel_envs
    
    def train_step_batched(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        batch_size = states.shape[0]
        
        for i in range(batch_size):
            transition = Transition(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=next_states[i],
                done=dones[i]
            )
            self.replay_buffer.add(transition)
        
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        return self.train_step()
    
    def collect_batch(
        self,
        envs,
        states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            output = self.agent.select_action(states)
            action_types = output.action_type
            reposition_targets = output.reposition_target
        
        next_states = []
        rewards = []
        dones = []
        
        for i, env in enumerate(envs):
            next_state, reward, done, _ = env.step(
                action_types[i].item(),
                reposition_targets[i].item()
            )
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device)
        
        return states, action_types, rewards, next_states, dones, reposition_targets
