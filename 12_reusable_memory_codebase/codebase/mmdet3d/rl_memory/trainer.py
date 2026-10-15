"""
High-level RL training loop for memory allocation policies.

This module intentionally provides a lightweight, readable stub for PPO/A2C-like
training over incremental stages. The actual detector training and evaluation
are delegated to the proxy API defined in ``env.run_stage_with_allocation``.
"""

from typing import Any, Dict, List, Sequence

import torch

from mmdet3d.utils import get_root_logger

from .env import run_stage_with_allocation
from .policy import MemoryAllocPolicy
from .structures import PerClassMetrics, SceneDescriptor


class PPOAgent:
    """Minimal PPO-style agent stub.

    This class stores trajectory data and exposes an ``update`` method that can
    be extended into a full PPO/A2C implementation. By default, the update
    performs no gradient steps; it only aggregates reward statistics. This
    keeps the default behavior safe and fast while still providing a clear
    place to plug in policy-gradient logic.
    """

    def __init__(self, policy: MemoryAllocPolicy, rl_cfg: Any):
        self.policy = policy
        self.rl_cfg = rl_cfg

        lr = float(getattr(rl_cfg, "lr", 3e-4))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.trajectories: List[Dict[str, Any]] = []

    def clear_buffer(self) -> None:
        self.trajectories.clear()

    def store_step(self, step_info: Dict[str, Any]) -> None:
        self.trajectories.append(step_info)

    def update(self) -> Dict[str, float]:
        """Run a placeholder PPO/A2C update on the stored trajectories.

        Default behavior:
          - Computes average reward over the collected trajectory.
          - Does not modify the policy parameters.

        This is intended as a safe stub; users can replace the body of this
        method with a full PPO/A2C implementation tailored to their needs.

        Returns:
            Dict with simple statistics (e.g., avg_reward).
        """
        if not self.trajectories:
            return {"avg_reward": 0.0}

        rewards = [float(t["reward"]) for t in self.trajectories]
        avg_reward = sum(rewards) / max(len(rewards), 1)

        # Placeholder: no policy update is performed by default
        stats = {"avg_reward": avg_reward}

        # Clear buffer after update
        self.clear_buffer()
        return stats


def train_rl_policy(
    policy: MemoryAllocPolicy,
    detector_init_ckpt: str,
    staged_datasets: Sequence[Any],
    initial_metrics: Dict[int, PerClassMetrics],
    scene_pool: List[SceneDescriptor],
    max_scenes: int,
    total_object_slots: int,
    rl_cfg: Any,
) -> None:
    """Train an RL policy over multiple episodes and stages.

    Pseudo-code:
        for episode in range(rl_cfg.num_episodes):
            reset detector to detector_init_ckpt
            metrics = initial_metrics
            for stage_data in staged_datasets:
                reward, new_metrics, log_info = run_stage_with_allocation(...)
                store (state, logits, reward, etc.) in RL buffer
                metrics = new_metrics
            after all stages in episode:
                run PPO/A2C update on policy using collected trajectory

    The detector reset and initialization behavior should be implemented inside
    the proxy training function referenced by ``rl_cfg.proxy_train_cfg``.

    Args:
        policy: Memory allocation policy to train.
        detector_init_ckpt: Path to the initial detector checkpoint.
        staged_datasets: Sequence of per-stage data objects.
        initial_metrics: Per-class metrics for the starting stage (k = 0).
        scene_pool: Candidate scenes for replay.
        max_scenes: Maximum number of scenes in the memory bank per stage.
        total_object_slots: Global object-level memory budget.
        rl_cfg: RL configuration object. Expected attributes include:
            - num_episodes: Number of RL episodes to run.
            - proxy_train_cfg: Training config passed through to
              ``run_stage_with_allocation`` (must expose proxy_train_fn).
            - lr, gamma, etc. (optional, for advanced implementations).
    """
    num_episodes = int(getattr(rl_cfg, "num_episodes", 1))
    proxy_train_cfg = getattr(rl_cfg, "proxy_train_cfg", None)

    agent = PPOAgent(policy, rl_cfg)

    logger = get_root_logger(log_level="INFO")

    for episode_idx in range(num_episodes):
        # Reset per-episode state
        metrics = initial_metrics
        detector_ckpt_path = detector_init_ckpt

        # Propagate episode index into proxy training config if present
        if proxy_train_cfg is not None:
            try:
                setattr(proxy_train_cfg, "episode_idx", episode_idx)
            except Exception:
                # Non-fatal; for configs that do not allow attribute assignment
                pass

        logger.info("=== RL episode %d / %d ===", episode_idx + 1, num_episodes)

        for stage_data in staged_datasets:
            reward, new_metrics, log_info = run_stage_with_allocation(
                detector_ckpt_path=detector_ckpt_path,
                stage_data=stage_data,
                metrics_k=metrics,
                policy=policy,
                scene_pool=scene_pool,
                max_scenes=max_scenes,
                total_object_slots=total_object_slots,
                train_cfg=proxy_train_cfg,
            )

            # Store step information for RL update
            agent.store_step(
                {
                    "reward": reward,
                    "metrics_before": metrics,
                    "metrics_after": new_metrics,
                    "log_info": log_info,
                }
            )

            metrics = new_metrics

        # End of episode: run PPO/A2C-style update (stub by default)
        stats = agent.update()
        logger.info(
            "RL episode %d summary: avg_reward=%.4f",
            episode_idx + 1,
            float(stats.get("avg_reward", 0.0)),
        )
