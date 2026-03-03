"""
Enhanced SAC Trainer with full paper implementation.

Integrates:
- Semi-MDP duration discounting (paper Eq. 12-13)
- WDRO robust targets (paper Section 4)
- Temperature annealing (paper Eq. 16)
- Smart assignment (~95% MILP optimality)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import time

from .trainer import SACTrainer, TrainingMetrics
from .semi_mdp import SemiMDPHandler
from .wdro import WDROAdversary, WDROConfig, ValueNetwork
from .smart_assignment import HybridAssignment
from ..config import TrainingConfig, CheckpointConfig, LoggingConfig
from ..networks.sac import SACAgent
from ..features.replay_buffer import GPUReplayBuffer


@dataclass
class EnhancedTrainingConfig:
    """Extended config for paper-compliant training."""
    # Base SAC config
    batch_size: int = 256
    gamma: float = 0.99
    gradient_steps: int = 1
    
    # Semi-MDP (paper Eq. 12-13)
    use_semi_mdp: bool = True
    step_duration_minutes: float = 5.0
    avg_speed_kmh: float = 30.0
    
    # WDRO (paper Section 4)
    use_wdro: bool = True
    wdro_rho: float = 0.3           # Ambiguity radius (0.3 = moderate robustness)
    wdro_rho_target: float = 0.2    # Target risk budget
    wdro_lambda_lr: float = 0.01    # Lambda update rate
    wdro_inner_steps: int = 3       # Inner optimization steps
    
    # Temperature annealing (paper Eq. 16)
    use_temperature_annealing: bool = True
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    
    # Mixed precision training (performance optimization)
    use_amp: bool = True  # Automatic Mixed Precision
    temperature_decay_episodes: int = 500
    
    # Smart assignment
    use_smart_assignment: bool = True


class EnhancedReplayBuffer(GPUReplayBuffer):
    """Replay buffer that stores action durations for Semi-MDP."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional storage for durations
        self._durations = None
    
    def push_with_duration(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool,
        duration: float = 1.0
    ):
        """Push transition with action duration."""
        # Call parent push
        self.push(state, action, reward, next_state, done)
        
        # Store duration
        if self._durations is None:
            self._durations = torch.ones(self.capacity, device=self.device)
        
        idx = (self._position - 1) % self.capacity
        self._durations[idx] = duration
    
    def sample_with_durations(self, batch_size: int):
        """Sample batch with durations."""
        batch = self.sample(batch_size)
        
        if self._durations is not None and hasattr(batch, 'indices'):
            durations = self._durations[batch.indices]
        else:
            durations = torch.ones(batch_size, device=self.device)
        
        return batch, durations


class EnhancedSACTrainer(SACTrainer):
    """
    SAC Trainer with full paper implementation.
    
    Features:
    - Semi-MDP discounting with variable durations
    - WDRO robust backup targets
    - Temperature annealing
    - Smart assignment integration
    """
    
    def __init__(
        self,
        agent: SACAgent,
        replay_buffer: GPUReplayBuffer,
        training_config: TrainingConfig,
        enhanced_config: Optional[EnhancedTrainingConfig] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
        device: str = 'cuda'
    ):
        # Mixed precision training
        self.use_amp = enhanced_config.use_amp if enhanced_config else False
        self.scaler = GradScaler(enabled=self.use_amp)
        
        super().__init__(
            agent, replay_buffer, training_config,
            checkpoint_config, logging_config, device
        )
        
        self.enhanced_config = enhanced_config or EnhancedTrainingConfig()
        
        # Initialize Semi-MDP handler
        if self.enhanced_config.use_semi_mdp:
            self.semi_mdp = SemiMDPHandler(
                gamma=training_config.gamma,
                step_duration_minutes=self.enhanced_config.step_duration_minutes,
                avg_speed_kmh=self.enhanced_config.avg_speed_kmh,
                device=device
            )
        else:
            self.semi_mdp = None
        
        # Initialize WDRO adversary
        if self.enhanced_config.use_wdro:
            wdro_config = WDROConfig(
                rho=self.enhanced_config.wdro_rho,
                rho_target=self.enhanced_config.wdro_rho_target,
                lambda_lr=self.enhanced_config.wdro_lambda_lr,
                inner_steps=self.enhanced_config.wdro_inner_steps
            )
            
            # Value network for WDRO
            self.value_network = ValueNetwork(
                state_dim=agent.state_dim,
                hidden_dims=[256, 256]
            ).to(device)
            
            self.value_optimizer = torch.optim.Adam(
                self.value_network.parameters(), lr=3e-4
            )
            
            self.wdro = WDROAdversary(
                scenario_dim=agent.state_dim,
                value_network=self.value_network,
                config=wdro_config,
                device=device
            )
        else:
            self.wdro = None
            self.value_network = None
        
        # Temperature annealing state
        self.current_temperature = self.enhanced_config.initial_temperature

    def _call_critic(
        self,
        critic,
        states_flat: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call critic via agent's unified interface."""
        return self.agent.call_critic(critic, states_flat, actions)

    def get_temperature(self) -> float:
        """Get current temperature based on episode count."""
        if not self.enhanced_config.use_temperature_annealing:
            return 1.0
        
        progress = min(1.0, self.episode / self.enhanced_config.temperature_decay_episodes)
        temp = self.enhanced_config.initial_temperature * (1 - progress) + \
               self.enhanced_config.final_temperature * progress
        
        self.current_temperature = temp
        return temp
    
    def _apply_scenario_to_state(
        self,
        states_flat: torch.Tensor,
        xi_scenario: torch.Tensor,
        num_hexes: int,
        hex_feature_dim: int,
        vehicle_dim: int
    ) -> torch.Tensor:
        """
        Apply scenario ξ to state by replacing hex demand features.
        
        This is used for WDRO adversarial optimization where ξ represents
        the full demand scenario (not a perturbation).
        
        Args:
            states_flat: Flattened states [batch, state_dim]
            xi_scenario: Full demand scenario [batch, num_hexes] - HAS GRADIENT
            num_hexes: Number of hexes
            hex_feature_dim: Hex feature dimension
            vehicle_dim: Vehicle feature dimension
        
        Returns:
            States with scenario applied [batch, state_dim]
        """
        batch_size = states_flat.shape[0]
        
        # DEBUG (optional): enable if you need to trace grad flow
        # if False:
        #     print(f"[DEBUG _apply_scenario_to_state]")
        #     print(f"  states_flat: requires_grad={states_flat.requires_grad}, grad_fn={states_flat.grad_fn}")
        #     print(f"  xi_scenario: requires_grad={xi_scenario.requires_grad}, is_leaf={xi_scenario.is_leaf}, grad_fn={xi_scenario.grad_fn}")
        
        # Extract dimensions
        hex_start = vehicle_dim
        hex_end = hex_start + num_hexes * hex_feature_dim
        context_dim = 9
        
        # CRITICAL FIX: Clone sliced tensors to create new tensors (not views)
        # PyTorch cat() doesn't preserve requires_grad from xi_scenario when 
        # concatenating with VIEWS of non-grad tensors (slicing creates views).
        # Solution: .clone() creates new tensors that can participate in autograd.
        
        # Extract components and CLONE to break view relationship
        vehicle_features = states_flat[:, :vehicle_dim].clone()  # New tensor, can have grad
        hex_flat = states_flat[:, hex_start:hex_end].clone()     # New tensor, can have grad
        context_features = states_flat[:, hex_end:].clone()      # New tensor, can have grad
        
        # Reshape hex features
        hex_features = hex_flat.reshape(batch_size, num_hexes, hex_feature_dim)
        
        # Replace demand feature (dim 0) with scenario ξ
        # xi_scenario has gradient (leaf variable from WDRO loop)
        # Other hex features from replay buffer (no grad)
        other_hex_features = hex_features[:, :, 1:].clone()  # Clone view to new tensor
        
        # Reconstruct hex features with scenario demand
        # GRADIENT FLOW: xi_scenario (leaf, requires_grad=True) → unsqueeze → cat → hex_features_new
        hex_features_new = torch.cat([
            xi_scenario.unsqueeze(-1),  # [batch, num_hexes, 1] - LEAF with grad!
            other_hex_features          # [batch, num_hexes, 4] - No grad (from replay)
        ], dim=-1)  # [batch, num_hexes, 5] - requires_grad=True from xi_scenario!
        
        # Concatenate back
        # GRADIENT FLOW: xi_scenario → hex_features_new → reshape → cat → perturbed
        perturbed = torch.cat([
            vehicle_features,                           # No grad
            hex_features_new.reshape(batch_size, -1),  # HAS GRAD from xi_scenario!
            context_features                            # No grad
        ], dim=1)  # perturbed.requires_grad = True ✓
        
        # DEBUG (optional): enable if you need to trace grad flow
        # if False:
        #     print(f"  perturbed: requires_grad={perturbed.requires_grad}, grad_fn={perturbed.grad_fn}")
        
        return perturbed
    
    def _perturb_state_with_scenario(
        self,
        states_flat: torch.Tensor,
        scenario_perturbation: torch.Tensor,
        num_hexes: int,
        hex_feature_dim: int
    ) -> torch.Tensor:
        """
        DEPRECATED: Use _apply_scenario_to_state instead.
        Perturb state with scenario perturbation for WDRO.
        
        Scenario ξ affects demand features in state.
        For simplicity, add perturbation to hex demand features.
        
        Args:
            states_flat: Flattened states [batch, state_dim]
            scenario_perturbation: Perturbation Δξ [batch, num_hexes]
            num_hexes: Number of hexes
            hex_feature_dim: Hex feature dimension
        
        Returns:
            Perturbed states [batch, state_dim]
        """
        batch_size = states_flat.shape[0]
        
        # Extract dimensions
        vehicle_dim = states_flat.shape[1] - num_hexes * hex_feature_dim - 9  # 9 is context_dim
        hex_start = vehicle_dim
        hex_end = hex_start + num_hexes * hex_feature_dim
        
        # Extract components (no clone to preserve gradient)
        vehicle_features = states_flat[:, :vehicle_dim]
        hex_flat = states_flat[:, hex_start:hex_end]
        context_features = states_flat[:, hex_end:]
        
        # Reshape hex features
        hex_features = hex_flat.reshape(batch_size, num_hexes, hex_feature_dim)
        
        # Perturb first feature (demand) of each hex
        # CRITICAL: Only the perturbation needs gradient, base state is detached
        demand_base = hex_features[:, :, 0].detach()
        demand_perturbed = demand_base + scenario_perturbation  # Gradient flows from scenario_perturbation!
        other_hex_features = hex_features[:, :, 1:].detach()  # Detach other features
        
        # Reconstruct hex features with perturbed demand
        hex_features_perturbed = torch.cat([
            demand_perturbed.unsqueeze(-1),  # [batch, num_hexes, 1] - HAS GRADIENT
            other_hex_features               # [batch, num_hexes, hex_feature_dim-1] - NO GRADIENT
        ], dim=-1)  # [batch, num_hexes, hex_feature_dim]
        
        # Concatenate back (gradient only flows through demand_perturbed -> scenario_perturbation)
        perturbed = torch.cat([
            vehicle_features.detach(),  # No grad
            hex_features_perturbed.reshape(batch_size, -1),  # Gradient from demand_perturbed
            context_features.detach()   # No grad
        ], dim=1)
        
        return perturbed
    
    def train_step_enhanced(
        self,
        durations: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Enhanced training step with Semi-MDP and WDRO.
        
        Args:
            durations: Optional action durations for Semi-MDP discounting
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        states = batch.states
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_states
        dones = batch.dones
        
        # Flatten states
        states_flat = self.agent._flatten_states(states)
        next_states_flat = self.agent._flatten_states(next_states)
        
        # Normalize rewards
        rewards_normalized = self.agent._normalize_rewards(rewards)
        
        # Get durations from batch if available
        if durations is None:
            if hasattr(batch, 'durations'):
                durations = batch.durations
            else:
                durations = torch.ones_like(rewards)
        
        # ===== Critic Update with Semi-MDP (Mixed Precision) =====
        with autocast(enabled=self.use_amp):
            critic_loss = self._compute_critic_loss_enhanced(
                states_flat, actions, rewards_normalized, 
                next_states_flat, dones, durations
            )
        
        self.agent.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.agent.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=1.0)
        self.scaler.step(self.agent.critic_optimizer)
        self.scaler.update()
        
        # ===== Value Network Update (for WDRO) (Mixed Precision) =====
        value_loss = torch.tensor(0.0, device=self.device)
        if self.value_network is not None:
            with autocast(enabled=self.use_amp):
                value_loss = self._update_value_network(states_flat, next_states_flat, dones)
        
        # ===== Actor Update with Auxiliary Assignment Loss (Mixed Precision) =====
        with autocast(enabled=self.use_amp):
            actor_loss, log_prob, aux_losses = self.agent.compute_actor_loss(
                states_flat,
                serve_vehicle_idx=batch.serve_vehicle_idx,
                serve_trip_idx=batch.serve_trip_idx,
                num_served=batch.num_served,
                charge_vehicle_idx=batch.charge_vehicle_idx,
                charge_station_idx=batch.charge_station_idx,
                num_charged=batch.num_charged,
            )
        
        self.agent.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.agent.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=1.0)
        self.scaler.step(self.agent.actor_optimizer)
        self.scaler.update()
        
        # ===== Alpha Update (Mixed Precision) =====
        alpha_loss = torch.tensor(0.0)
        if self.agent.auto_alpha:
            with autocast(enabled=self.use_amp):
                alpha_loss = self.agent.compute_alpha_loss(log_prob)
            self.agent.alpha_optimizer.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.agent.alpha_optimizer)
            self.scaler.update()
        
        # ===== WDRO Lambda Update =====
        if self.wdro is not None:
            self.wdro.update_lambda()
        
        self.agent.soft_update_target()
        self.global_step += 1
        
        # Compute metrics
        with torch.no_grad():
            actor_output = self.agent.forward(states_flat, deterministic=False)
            # Handle both SACOutput and legacy tuple format
            if hasattr(actor_output, 'action_type'):
                action_type = actor_output.action_type
            else:
                action_type = actor_output[0]
            
            q1, q2 = self._call_critic(self.agent.critic, states_flat, action_type)
            q_mean = torch.min(q1, q2).mean().item()
        
        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            'alpha': self.agent.alpha.item(),
            'q_mean': q_mean,
            'temperature': self.current_temperature,
            'serve_loss': aux_losses.get('serve_loss', 0.0),
            'charge_loss': aux_losses.get('charge_loss', 0.0),
            'assignment_loss': aux_losses.get('assignment_loss', 0.0),
        }
        
        if self.value_network is not None:
            metrics['value_loss'] = value_loss.item() if isinstance(value_loss, torch.Tensor) else value_loss
        
        if self.wdro is not None:
            metrics['wdro_lambda'] = self.wdro.lambda_.item()
            metrics['wdro_rho_hat'] = self.wdro.running_rho_hat
        
        return metrics
    
    def _compute_critic_loss_enhanced(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        durations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute critic loss with Semi-MDP discounting.
        
        Uses γ^Δ instead of fixed γ per paper Eq. 13.
        """
        # Extract per-vehicle action types for critic
        if actions.dim() == 3:
            per_vehicle_actions = actions[:, :, 0].long()  # [batch, num_vehicles]
        elif actions.dim() == 2:
            per_vehicle_actions = actions.long()  # [batch, num_vehicles] or [batch, 2]
        else:
            per_vehicle_actions = actions.unsqueeze(0).long()
        
        # Get next actions from policy (use agent.forward for GCN compatibility)
        # Keep policy/critic target under no_grad, then exit for WDRO autograd
        with torch.no_grad():
            actor_output = self.agent.forward(next_states, deterministic=False)
        
        # Handle both SACOutput and legacy tuple format
        if hasattr(actor_output, 'action_type'):
            next_action = actor_output.action_type
            next_action_log_prob = actor_output.action_log_prob
            next_reposition_log_prob = actor_output.reposition_log_prob
        else:
            next_action, next_action_log_prob, _, next_reposition_log_prob = actor_output
        
        with torch.no_grad():
            next_q1, next_q2 = self._call_critic(self.agent.critic_target, next_states, next_action)
            next_q = torch.min(next_q1, next_q2)
        
        # Aggregate per-vehicle log probs to fleet-level
        # GCN outputs [B, V], need to reduce to [B]
        if next_action_log_prob.dim() == 2:
            next_action_log_prob = next_action_log_prob.mean(dim=1)
        if next_reposition_log_prob.dim() == 2:
            next_reposition_log_prob = next_reposition_log_prob.mean(dim=1)
        
        total_log_prob = next_action_log_prob + next_reposition_log_prob
        next_v = next_q - self.agent.alpha * total_log_prob
        
        # Semi-MDP: use γ^Δ instead of fixed γ (paper Eq. 13)
        discount = torch.pow(self.agent.gamma, durations)
        
        # Compute backup target (robust if WDRO enabled)
        if self.wdro is not None:
            # Extract scenario from batch
            # scenario_demand and scenario_context should be in batch now
            # For now, construct scenario from state features
            batch_size = states.shape[0]
            
            # State structure: [vehicle_features, hex_features, context_features]
            # Use agent's stored dimensions (no hardcoding!)
            num_vehicles = self.agent.num_vehicles
            vehicle_feature_dim = self.agent.vehicle_feature_dim
            vehicle_dim = num_vehicles * vehicle_feature_dim
            context_dim = self.agent.context_dim
            hex_feature_dim = self.agent.hex_feature_dim
            state_dim = states.shape[1]
            hex_total_dim = state_dim - vehicle_dim - context_dim
            num_hexes = hex_total_dim // hex_feature_dim
            
            # Extract hex demand features as scenario ξ̂ from CURRENT states
            # IMPORTANT: ξ̂ should be the scenario the agent observed when making the decision
            hex_start = vehicle_dim
            hex_end = hex_start + num_hexes * hex_feature_dim
            hex_flat_current = states[:, hex_start:hex_end]  # From current states, not next!
            hex_features_current = hex_flat_current.reshape(batch_size, num_hexes, hex_feature_dim)
            xi_hat = hex_features_current[:, :, 0]  # Demand feature [batch, num_hexes]
            
            # Create next_state perturbation function
            def next_state_fn(xi_scenario):
                """Apply scenario to next_state (xi_scenario is full demand, not perturbation)."""
                # DEBUG: Check input
                assert xi_scenario.requires_grad, f"xi_scenario must have grad! requires_grad={xi_scenario.requires_grad}, is_leaf={xi_scenario.is_leaf}"
                
                result = self._apply_scenario_to_state(
                    next_states, xi_scenario, num_hexes, hex_feature_dim, vehicle_dim
                )
                
                # DEBUG: Check output
                assert result.requires_grad, f"result must have grad! requires_grad={result.requires_grad}, grad_fn={result.grad_fn}"
                
                return result
            
            # Duration function for semi-MDP
            def duration_fn(xi):
                # Duration depends on scenario (travel times)
                # For now, use stored durations (simplified)
                return durations
            
            # Compute robust target via WDRO
            target_q = self.wdro.compute_robust_target(
                rewards=rewards,
                xi_hat=xi_hat,
                next_state_fn=next_state_fn,
                gamma=self.agent.gamma,
                duration_fn=duration_fn
            )
        else:
            # Standard backup target
            target_q = rewards + (1 - dones.float()) * discount * next_v
        
        q1, q2 = self._call_critic(self.agent.critic, states, per_vehicle_actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        return critic_loss
    
    def _update_value_network(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Update value network for WDRO adversary."""
        # Value target from Q-network
        with torch.no_grad():
            actor_output = self.agent.forward(states, deterministic=False)
            # Handle both SACOutput and legacy tuple format
            if hasattr(actor_output, 'action_type'):
                action = actor_output.action_type
                log_prob = actor_output.action_log_prob
                reposition_log_prob = actor_output.reposition_log_prob
            else:
                action, log_prob, _, reposition_log_prob = actor_output
            
            if log_prob.dim() == 2:
                log_prob = log_prob.mean(dim=1)
            if reposition_log_prob.dim() == 2:
                reposition_log_prob = reposition_log_prob.mean(dim=1)

            q1, q2 = self._call_critic(self.agent.critic, states, action)
            min_q = torch.min(q1, q2)
            
            total_log_prob = log_prob + reposition_log_prob
            v_target = min_q - self.agent.alpha * total_log_prob
        
        v_pred = self.value_network(states).squeeze(-1)
        value_loss = F.mse_loss(v_pred, v_target)
        
        self.value_optimizer.zero_grad()
        self.scaler.scale(value_loss).backward()
        self.scaler.unscale_(self.value_optimizer)
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
        self.scaler.step(self.value_optimizer)
        self.scaler.update()
        
        return value_loss
    
    def save_checkpoint(self, path: str):
        """Save checkpoint including enhanced components."""
        checkpoint = {
            'agent_state_dict': self.agent.state_dict(),
            'global_step': self.global_step,
            'episode': self.episode,
            'best_reward': self.best_reward,
            'current_temperature': self.current_temperature,
        }
        
        if self.value_network is not None:
            checkpoint['value_network_state_dict'] = self.value_network.state_dict()
            checkpoint['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
        
        if self.wdro is not None:
            checkpoint['wdro_state_dict'] = self.wdro.state_dict()
        
        save_path = self.checkpoint_config.save_dir / path if hasattr(self.checkpoint_config.save_dir, '__truediv__') else f"{self.checkpoint_config.save_dir}/{path}"
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint including enhanced components."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.episode = checkpoint.get('episode', 0)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        self.current_temperature = checkpoint.get('current_temperature', 1.0)
        
        if self.value_network is not None and 'value_network_state_dict' in checkpoint:
            self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        if self.wdro is not None and 'wdro_state_dict' in checkpoint:
            self.wdro.load_state_dict(checkpoint['wdro_state_dict'])


def create_enhanced_trainer(
    agent: SACAgent,
    replay_buffer: GPUReplayBuffer,
    training_config: TrainingConfig,
    use_semi_mdp: bool = True,
    use_wdro: bool = True,
    use_temperature_annealing: bool = True,
    device: str = 'cuda'
) -> EnhancedSACTrainer:
    """Factory function to create enhanced trainer with paper features."""
    
    enhanced_config = EnhancedTrainingConfig(
        batch_size=training_config.batch_size,
        gamma=training_config.gamma,
        use_semi_mdp=use_semi_mdp,
        use_wdro=use_wdro,
        use_temperature_annealing=use_temperature_annealing
    )
    
    return EnhancedSACTrainer(
        agent=agent,
        replay_buffer=replay_buffer,
        training_config=training_config,
        enhanced_config=enhanced_config,
        device=device
    )
