"""
Wasserstein Distributionally Robust Optimization (WDRO) per paper Section 4.

Implements:
- Graph-Aligned Mahalanobis (MAG) metric (paper Eq. 14)
- Wasserstein-1 ambiguity set with KR dual (paper Eq. 19-20)
- Robust backup with projected gradient ascent (paper Section 4.3)
- Primal-dual risk-budget tracking (paper Eq. 23-24)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class WDROConfig:
    """Configuration for WDRO."""
    rho: float = 1.0                 # Ambiguity radius
    rho_target: float = 0.5          # Target risk budget
    lambda_init: float = 1.0         # Initial dual variable
    lambda_lr: float = 0.01          # Learning rate for lambda update
    inner_steps: int = 5             # Steps for inner maximization
    inner_lr: float = 0.02           # Step size for inner gradient ascent
    beta: float = 0.3                # Graph Laplacian weight
    epsilon: float = 1e-6            # Numerical stability
    support_radius: float = 10.0     # Radius of support set B


class MAGMetric:
    
    def __init__(
        self,
        feature_dim,
        beta=0.3,
        epsilon=1e-6,
        device="cuda"
    ):
        self.feature_dim = feature_dim
        self.beta = beta
        self.epsilon = epsilon
        self.device = torch.device(device)

        self.running_var = torch.ones(feature_dim, device=device)
        self.sum = torch.zeros(feature_dim)
        self.sq_sum = torch.zeros(feature_dim)
        self.count = 0
        self.num_hexes = None

        self.Q_graph = None

        self.Q = None
        self._Q_sqrt = None
        self._Q_inv_sqrt = None
        self.L_spatial = None
        self.L_od = None
    

    def update_statistics(self, features):
        # features: [batch, dim]

        self.sum += features.sum(dim=0)
        self.sq_sum += (features**2).sum(dim=0)
        self.count += features.size(0)

        mean = self.sum / self.count
        var = self.sq_sum / self.count - mean**2

        self.running_var = var.clamp(min=self.epsilon)

        self._build_Q()


    def set_spatial_graph(self, adj):
        adj = adj.to(self.device)
        deg = adj.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-6))

        L = torch.eye(adj.size(0), device=self.device) \
            - D_inv_sqrt @ adj @ D_inv_sqrt

        self.L_spatial = L

    def build_od_graph(self, OD):
        # OD: [H,H]

        # cosine similarity
        OD_norm = F.normalize(OD, dim=1)
        sim = OD_norm @ OD_norm.T

        # threshold
        sim[sim < 0.2] = 0

        deg = sim.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-6))

        L = torch.eye(sim.size(0), device=self.device) \
            - D_inv_sqrt @ sim @ D_inv_sqrt

        self.L_od = L
    

    def _build_Q(self):
        self.Q_graph = self.L_spatial + self.L_od

        # w = inverse variance
        w = 1.0 / (self.running_var + self.epsilon)

        Q_diag = torch.diag(w)

        if self.Q_graph is not None:
            Q = Q_diag + self.beta * self.Q_graph
        else:
            Q = Q_diag

        # Regularize
        Q = Q + self.epsilon * torch.eye(Q.size(0), device=self.device)

        self.Q = Q

        # Cholesky
        L = torch.linalg.cholesky(Q)

        self._Q_sqrt = L
        self._Q_inv_sqrt = torch.cholesky_inverse(L)
    

    def distance(self, xi: torch.Tensor, xi_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute MAG distance d_Q(ξ, ξ').
        
        Args:
            xi: [batch, feature_dim] - perturbed scenario
            xi_hat: [batch, feature_dim] - nominal scenario
            
        Returns:
            distance: [batch] - d_Q(ξ, ξ')
        """
        diff = xi - xi_hat
        z = diff @ self._Q_sqrt.T
        return torch.norm(z, dim=-1)
    

    def subgradient(self, xi: torch.Tensor, xi_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute subgradient of d_Q w.r.t. ξ per paper Section 4.3.
        
        g_d(ξ) = Q(ξ - ξ̂) / max(ε, ||ξ - ξ̂||_Q)
        """
        diff = xi - xi_hat
        dist = self.distance(xi, xi_hat).unsqueeze(-1)
        grad = diff @ self.Q
        return grad / dist.clamp(min=self.epsilon)
    

    def project_to_ball(
        self,
        xi: torch.Tensor,
        center: torch.Tensor,
        radius: float
    ) -> torch.Tensor:
        """
        Project ξ onto Q-ball of given radius centered at center.
        
        Per paper Eq. 21:
        u = Q^{1/2}(ξ - center)
        u ← R * u / max(R, ||u||)
        ξ_proj = center + Q^{-1/2} u
        """
        diff = xi - center

        # u = Q^{1/2} diff
        u = diff @ self._Q_sqrt.T
        norm = torch.norm(u, dim=-1, keepdim=True)
        u = radius * u / norm.clamp(min=radius)

        # xi_proj = center + Q^{-1/2} u
        diff_proj = u @ self._Q_inv_sqrt.T

        return center + diff_proj

class WDROAdversary(nn.Module):
    """
    WDRO Adversary that finds worst-case scenarios per paper Section 4.
    
    Solves the inner maximization:
    max_ξ [γ^Δ(ξ) V(s'(ξ)) - λ d_Q(ξ, ξ̂)]
    
    using projected gradient ascent.
    """
    
    def __init__(
        self,
        scenario_dim: int,
        value_network: nn.Module,
        config: WDROConfig,
        device: str = "cuda"
    ):
        super().__init__()
        self.scenario_dim = scenario_dim
        self.value_network = value_network
        self.config = config
        self.device = torch.device(device)
        
        # MAG metric
        self.mag_metric = MAGMetric(
            feature_dim=scenario_dim,
            num_hexes=1,  # Simplified
            beta=config.beta,
            device=device
        )
        
        adj = load_hex_adjacency()   # [N,N]

        assert adj.shape[0] == scenario_dim, \
            "Adjacency size must match scenario_dim"

        self.mag_metric.set_graph_adjacency(adj)

        # Dual variable λ for Lagrangian
        self.log_lambda = nn.Parameter(torch.tensor(config.lambda_init).log())
        
        # Running statistics for primal-dual update
        self.running_rho_hat = 0.0
        self.update_count = 0
    
    @property
    def lambda_(self) -> torch.Tensor:
        """Get current λ value (positive)."""
        return self.log_lambda.exp()
    
    def find_adversarial_scenario(
        self,
        xi_hat: torch.Tensor,        # [batch, scenario_dim] - nominal scenario
        next_state_fn,               # Function: scenario -> next_state
        gamma: float = 0.99,
        duration_fn=None             # Optional function: scenario -> duration
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find worst-case scenario ξ* via projected gradient ascent.
        
        Returns:
            xi_star: [batch, scenario_dim] - adversarial scenario
            worst_value: [batch] - V(s'(ξ*))
        """
        batch_size = xi_hat.size(0)
        
        # Initialize at nominal
        # CRITICAL: Must detach first to create clean leaf variable
        xi = xi_hat.detach().clone().requires_grad_(True)
        
        # DEBUG: Verify xi is a leaf
        assert xi.is_leaf, f"xi must be leaf! is_leaf={xi.is_leaf}, grad_fn={xi.grad_fn}"
        assert xi.requires_grad, f"xi must require grad! requires_grad={xi.requires_grad}"
        
        for k in range(self.config.inner_steps):
            # Compute value at current scenario
            next_state = next_state_fn(xi)
            
            # DEBUG: Check gradient flow
            assert next_state.requires_grad, f"next_state must require grad at step {k}! requires_grad={next_state.requires_grad}, grad_fn={next_state.grad_fn}"
            
            value = self.value_network(next_state).squeeze(-1)
            
            # DEBUG: Check value gradient
            assert value.requires_grad, f"value must require grad at step {k}! requires_grad={value.requires_grad}, grad_fn={value.grad_fn}"
            
            # Compute duration discount if provided
            if duration_fn is not None:
                duration = duration_fn(xi)
                discount = torch.pow(gamma, duration)
            else:
                discount = gamma
            
            # Objective: γ^Δ V(s'(ξ)) - λ d_Q(ξ, ξ̂)
            distance = self.mag_metric.distance(xi, xi_hat)
            objective = discount * value + self.lambda_ * distance
            
            # Compute gradient w.r.t. ξ of the value part
            grad_v = torch.autograd.grad(objective.sum(), xi, create_graph=False)[0]

            # Compute subgradient of the distance part
            g_d = self.mag_metric.subgradient(xi, xi_hat)
            
            # Gradient descent step (we are minimizing f(ξ) = γ^Δ V(s'(ξ)) + λ d_Q(ξ, ξ̂))
            xi_new = xi.detach() - self.config.inner_lr * (grad_v + self.lambda_ * g_d)
            
            # Project onto support set B (Q-ball)
            xi_new = self.mag_metric.project_to_ball(
                xi_new, xi_hat.detach(), self.config.support_radius
            )
            
            # CRITICAL: Detach and create new leaf variable for next iteration
            xi = xi_new.detach().clone().requires_grad_(True)
        
        # Final value computation (keep grad to satisfy next_state_fn assertion)
        next_state_star = next_state_fn(xi)
        worst_value = self.value_network(next_state_star).squeeze(-1)
        
        if duration_fn is not None:
            duration_star = duration_fn(xi)
            discount_star = torch.pow(gamma, duration_star)
        else:
            discount_star = gamma
        
        distance_star = self.mag_metric.distance(xi, xi_hat)
        
        return xi.detach(), worst_value, discount_star, distance_star
    
    def compute_robust_target(
        self,
        rewards: torch.Tensor,
        xi_hat: torch.Tensor,
        next_state_fn,
        gamma: float = 0.99,
        duration_fn=None
    ) -> torch.Tensor:
        """
        Compute robust backup target per paper Eq. 20.
        
        y_t^rob = r_t + λρ + max_ξ [γ^Δ V(s'(ξ)) - λ d_Q(ξ, ξ̂)]
        """
        xi_star, worst_value, discount, distance = self.find_adversarial_scenario(
            xi_hat, next_state_fn, gamma, duration_fn
        )
        
        # Robust target: r + λρ + (γ^Δ V(s'(ξ*)) - λ d_Q(ξ*, ξ̂))
        robust_term = discount * worst_value + self.lambda_ * distance
        target = rewards - self.lambda_ * self.config.rho + robust_term
        
        # Update running ρ_hat for primal-dual
        rho_hat = distance.mean().item()
        alpha = min(0.1, 1.0 / (self.update_count + 1))
        self.running_rho_hat = (1 - alpha) * self.running_rho_hat + alpha * rho_hat
        self.update_count += 1
        
        return target
    
    def update_lambda(self):
        """
        Update dual variable λ via primal-dual per paper Eq. 24.
        
        λ_{t+1} = [λ_t + η_λ (ρ̂_t - ρ_target)]_+
        """
        # Compute step size with decay
        eta = self.config.lambda_lr / (self.update_count ** 0.5 + 1)
        
        # Gradient: ρ̂ - ρ_target
        grad = self.running_rho_hat - self.config.rho_target
        
        # Update in log space for positivity
        with torch.no_grad():
            new_lambda = (self.lambda_ + eta * grad).clamp(min=1e-6)
            self.log_lambda.data = new_lambda.log()


class ValueNetwork(nn.Module):
    """
    Value network V_φ for WDRO adversary per paper Section 5.1.5.
    
    Used to compute worst-case value: max_ξ γ^Δ(ξ) V(s'(ξ))
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [256, 256],
        dropout: float = 0.1
    ):
        super().__init__()
        
        dims = [state_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state value V(s)."""
        return self.network(state)


class RobustSACAgent(nn.Module):
    """
    SAC Agent with WDRO integration per paper Algorithm 1.
    
    Extends standard SAC with:
    - Robust backup targets using WDRO
    - Value network for adversary
    - Primal-dual λ updates
    """
    
    def __init__(
        self,
        base_sac_agent,  # Standard SACAgent
        wdro_config: WDROConfig,
        scenario_dim: int
    ):
        super().__init__()
        self.base_agent = base_sac_agent
        self.wdro_config = wdro_config
        
        # Value network for adversary
        self.value_network = ValueNetwork(
            state_dim=base_sac_agent.state_dim,
            hidden_dims=[256, 256]
        )
        
        # WDRO adversary
        self.adversary = WDROAdversary(
            scenario_dim=scenario_dim,
            value_network=self.value_network,
            config=wdro_config,
            device=base_sac_agent.device
        )
        
        # Value network optimizer
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=3e-4
        )
    
    def compute_robust_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        scenarios: torch.Tensor,  # Scenario features for WDRO
        **kwargs
    ) -> torch.Tensor:
        """Compute critic loss with robust targets."""

        self.adversary.mag_metric.update_statistics(scenarios)

        # Create next_state function for adversary
        def next_state_fn(xi):
            # In practice, this would modify next_states based on scenario
            # For now, use identity (simplified)
            return next_states
        
        # Compute robust target
        robust_target = self.adversary.compute_robust_target(
            rewards=rewards,
            xi_hat=scenarios,
            next_state_fn=next_state_fn,
            gamma=self.base_agent.gamma
        )
        
        # Standard critic loss with robust target
        if actions.dim() == 3:
            per_vehicle_actions = actions[:, :, 0].long()
        else:
            per_vehicle_actions = actions.long()
        q1, q2 = self.base_agent.call_critic(self.base_agent.critic, states, per_vehicle_actions)
        
        critic_loss = F.mse_loss(q1, robust_target) + F.mse_loss(q2, robust_target)
        
        return critic_loss
    
    def update_value_network(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ):
        """Update value network to match soft value."""
        # Target: r + γ * V(s') (simplified, should use entropy bonus)
        with torch.no_grad():
            next_value = self.value_network(next_states)
            target = rewards.unsqueeze(-1) + self.base_agent.gamma * (1 - dones.unsqueeze(-1).float()) * next_value
        
        value_pred = self.value_network(states)
        value_loss = F.mse_loss(value_pred, target)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return value_loss.item()
    
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """Full update step including WDRO components."""
        # Standard SAC update
        metrics = self.base_agent.update(*args, **kwargs)
        
        # Update λ for primal-dual
        self.adversary.update_lambda()
        
        metrics['lambda'] = self.adversary.lambda_.item()
        metrics['rho_hat'] = self.adversary.running_rho_hat
        
        return metrics
