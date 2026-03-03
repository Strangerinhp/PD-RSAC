"""
Curriculum Learning Scheduler.

Gradually adjusts training parameters (like max_pickup_distance) over time
to help the agent learn from easier to harder scenarios.
"""

import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch


@dataclass
class CurriculumState:
    """Current state of curriculum learning."""
    current_episode: int
    total_episodes: int
    progress: float  # 0.0 to 1.0
    pickup_distance: float
    curriculum_active: bool


class CurriculumScheduler:
    """
    Manages curriculum learning schedules.
    
    Currently supports:
    - pickup_distance: Anneal from large (easy) to small (hard) pickup radius
    
    Schedules:
    - linear: Linear interpolation
    - cosine: Smooth cosine annealing (slower at ends, faster in middle)
    - exponential: Faster decay initially
    """
    
    def __init__(
        self,
        enabled: bool = True,
        pickup_start: float = 10.0,
        pickup_end: float = 1.0,
        schedule: Literal["linear", "cosine", "exponential"] = "cosine",
        end_fraction: float = 0.7,
        total_episodes: int = 5000
    ):
        """
        Args:
            enabled: Whether curriculum is active
            pickup_start: Starting max pickup distance (km) - easier
            pickup_end: Ending max pickup distance (km) - harder
            schedule: Annealing schedule type
            end_fraction: Fraction of training when curriculum completes (e.g., 0.7 = 70%)
            total_episodes: Total training episodes
        """
        self.enabled = enabled
        self.pickup_start = pickup_start
        self.pickup_end = pickup_end
        self.schedule = schedule
        self.end_fraction = end_fraction
        self.total_episodes = total_episodes
        
        # Compute curriculum end episode
        self.curriculum_end_episode = int(total_episodes * end_fraction)
        
        # Track current values
        self._current_pickup_distance = pickup_start
        self._current_episode = 0
        
    def get_pickup_distance(self, episode: int) -> float:
        """
        Get the max pickup distance for a given episode.
        
        Args:
            episode: Current episode number (0-indexed)
            
        Returns:
            Max pickup distance in km
        """
        if not self.enabled:
            return self.pickup_start
        
        # After curriculum ends, use final value
        if episode >= self.curriculum_end_episode:
            return self.pickup_end
        
        # Compute progress within curriculum phase [0, 1]
        progress = episode / self.curriculum_end_episode
        
        # Apply schedule
        if self.schedule == "linear":
            factor = progress
        elif self.schedule == "cosine":
            # Cosine annealing: slower at start and end
            factor = 0.5 * (1 - math.cos(math.pi * progress))
        elif self.schedule == "exponential":
            # Exponential decay
            factor = 1 - math.exp(-3 * progress)
        else:
            factor = progress
        
        # Interpolate between start and end
        pickup_distance = self.pickup_start + factor * (self.pickup_end - self.pickup_start)
        
        self._current_pickup_distance = pickup_distance
        self._current_episode = episode
        
        return pickup_distance
    
    def get_state(self, episode: int) -> CurriculumState:
        """Get full curriculum state for logging."""
        pickup = self.get_pickup_distance(episode)
        progress = min(1.0, episode / self.curriculum_end_episode) if self.enabled else 1.0
        
        return CurriculumState(
            current_episode=episode,
            total_episodes=self.total_episodes,
            progress=progress,
            pickup_distance=pickup,
            curriculum_active=self.enabled and episode < self.curriculum_end_episode
        )
    
    def step(self, episode: int) -> float:
        """
        Convenience method to step curriculum and return pickup distance.
        
        Same as get_pickup_distance but more intuitive naming.
        """
        return self.get_pickup_distance(episode)
    
    @classmethod
    def from_config(cls, config, total_episodes: int) -> 'CurriculumScheduler':
        """
        Create scheduler from config object.
        
        Args:
            config: Config with curriculum settings
            total_episodes: Total training episodes
            
        Returns:
            CurriculumScheduler instance
        """
        curriculum_cfg = config.curriculum
        
        return cls(
            enabled=curriculum_cfg.enabled,
            pickup_start=curriculum_cfg.pickup_distance.start,
            pickup_end=curriculum_cfg.pickup_distance.end,
            schedule=curriculum_cfg.pickup_distance.schedule,
            end_fraction=curriculum_cfg.curriculum_end_fraction,
            total_episodes=total_episodes
        )


def apply_curriculum_to_environment(
    env,
    scheduler: CurriculumScheduler,
    episode: int,
    verbose: bool = False
) -> float:
    """
    Apply curriculum settings to environment.
    
    Args:
        env: GPUEnvironment, GPUEnvironmentV2, or BatchedGPUEnvironment
        scheduler: CurriculumScheduler
        episode: Current episode
        verbose: Print updates
        
    Returns:
        Current pickup distance
    """
    pickup_distance = scheduler.get_pickup_distance(episode)
    
    # Update environment's max_pickup_distance
    env.max_pickup_distance = pickup_distance
    
    # Also update TripAssigner if it exists
    if hasattr(env, '_trip_assigner') and env._trip_assigner is not None:
        env._trip_assigner.max_pickup_distance = pickup_distance
    
    # Also update ActionProcessor if it exists (for V2)
    if hasattr(env, '_action_processor') and env._action_processor is not None:
        env._action_processor.max_pickup_distance = pickup_distance
    
    if verbose:
        state = scheduler.get_state(episode)
        if state.curriculum_active:
            print(f"[Curriculum] Episode {episode}: pickup_distance = {pickup_distance:.2f} km "
                  f"(progress: {state.progress:.1%})")
    
    return pickup_distance
