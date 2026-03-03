import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class LossResult:
    loss: torch.Tensor
    metrics: Dict[str, float]


class PolicyLoss(nn.Module):
    def __init__(self, entropy_coef: float = 0.01):
        super().__init__()
        self.entropy_coef = entropy_coef
    
    def forward(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        entropy: Optional[torch.Tensor] = None
    ) -> LossResult:
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        total_loss = policy_loss
        metrics = {'policy_loss': policy_loss.item()}
        
        if entropy is not None:
            entropy_loss = -entropy.mean()
            total_loss = total_loss + self.entropy_coef * entropy_loss
            metrics['entropy'] = entropy.mean().item()
            metrics['entropy_loss'] = entropy_loss.item()
        
        return LossResult(loss=total_loss, metrics=metrics)


class ValueLoss(nn.Module):
    def __init__(self, clip_range: Optional[float] = None):
        super().__init__()
        self.clip_range = clip_range
    
    def forward(
        self,
        values: torch.Tensor,
        targets: torch.Tensor,
        old_values: Optional[torch.Tensor] = None
    ) -> LossResult:
        if self.clip_range is not None and old_values is not None:
            clipped_values = old_values + torch.clamp(
                values - old_values,
                -self.clip_range,
                self.clip_range
            )
            loss1 = F.mse_loss(values, targets, reduction='none')
            loss2 = F.mse_loss(clipped_values, targets, reduction='none')
            value_loss = torch.max(loss1, loss2).mean()
        else:
            value_loss = F.mse_loss(values, targets)
        
        metrics = {
            'value_loss': value_loss.item(),
            'value_mean': values.mean().item(),
            'target_mean': targets.mean().item()
        }
        
        return LossResult(loss=value_loss, metrics=metrics)


class EntropyLoss(nn.Module):
    def __init__(self, target_entropy: float):
        super().__init__()
        self.target_entropy = target_entropy
    
    def forward(
        self,
        log_alpha: torch.Tensor,
        log_probs: torch.Tensor
    ) -> LossResult:
        alpha_loss = -(log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        metrics = {
            'alpha_loss': alpha_loss.item(),
            'alpha': log_alpha.exp().item(),
            'log_prob_mean': log_probs.mean().item()
        }
        
        return LossResult(loss=alpha_loss, metrics=metrics)


class SACLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        auto_alpha: bool = True,
        initial_alpha: float = 0.2
    ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        
        if auto_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(initial_alpha).log())
            self.target_entropy = target_entropy
        else:
            self.register_buffer('log_alpha', torch.tensor(initial_alpha).log())
            self.target_entropy = None
    
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()
    
    def compute_critic_loss(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        rewards: torch.Tensor,
        next_q_target: torch.Tensor,
        next_log_probs: torch.Tensor,
        dones: torch.Tensor
    ) -> LossResult:
        with torch.no_grad():
            next_v = next_q_target - self.alpha * next_log_probs
            target_q = rewards + (1 - dones.float()) * self.gamma * next_v
        
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        
        metrics = {
            'critic_loss': critic_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
            'target_q_mean': target_q.mean().item()
        }
        
        return LossResult(loss=critic_loss, metrics=metrics)
    
    def compute_actor_loss(
        self,
        log_probs: torch.Tensor,
        q_values: torch.Tensor
    ) -> LossResult:
        actor_loss = (self.alpha.detach() * log_probs - q_values).mean()
        
        metrics = {
            'actor_loss': actor_loss.item(),
            'log_prob_mean': log_probs.mean().item(),
            'q_mean': q_values.mean().item()
        }
        
        return LossResult(loss=actor_loss, metrics=metrics)
    
    def compute_alpha_loss(
        self,
        log_probs: torch.Tensor
    ) -> LossResult:
        if not self.auto_alpha:
            return LossResult(
                loss=torch.tensor(0.0, device=log_probs.device),
                metrics={'alpha': self.alpha.item()}
            )
        
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        metrics = {
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item()
        }
        
        return LossResult(loss=alpha_loss, metrics=metrics)


class TDLoss(nn.Module):
    def __init__(self, gamma: float = 0.99, n_step: int = 1):
        super().__init__()
        self.gamma = gamma
        self.n_step = n_step
    
    def compute_n_step_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        batch_size = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        
        running_return = values[:, -1]
        for t in reversed(range(rewards.shape[1])):
            running_return = rewards[:, t] + self.gamma * running_return * (1 - dones[:, t].float())
            returns[:, t] = running_return
        
        return returns
    
    def forward(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> LossResult:
        with torch.no_grad():
            targets = rewards + self.gamma * next_values * (1 - dones.float())
        
        td_loss = F.mse_loss(values, targets)
        td_error = (values - targets).abs()
        
        metrics = {
            'td_loss': td_loss.item(),
            'td_error_mean': td_error.mean().item(),
            'td_error_max': td_error.max().item()
        }
        
        return LossResult(loss=td_loss, metrics=metrics)


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> LossResult:
        loss = F.huber_loss(predictions, targets, delta=self.delta)
        
        metrics = {
            'huber_loss': loss.item(),
            'pred_mean': predictions.mean().item(),
            'target_mean': targets.mean().item()
        }
        
        return LossResult(loss=loss, metrics=metrics)


class QuantileLoss(nn.Module):
    def __init__(self, num_quantiles: int = 32):
        super().__init__()
        self.num_quantiles = num_quantiles
        taus = (torch.arange(num_quantiles) + 0.5) / num_quantiles
        self.register_buffer('taus', taus)
    
    def forward(
        self,
        quantiles: torch.Tensor,
        targets: torch.Tensor
    ) -> LossResult:
        targets = targets.unsqueeze(-1)
        td_error = targets - quantiles
        
        huber_loss = torch.where(
            td_error.abs() <= 1,
            0.5 * td_error.pow(2),
            td_error.abs() - 0.5
        )
        
        quantile_loss = (self.taus - (td_error < 0).float()).abs() * huber_loss
        loss = quantile_loss.sum(dim=-1).mean()
        
        metrics = {
            'quantile_loss': loss.item(),
            'quantile_mean': quantiles.mean().item()
        }
        
        return LossResult(loss=loss, metrics=metrics)
