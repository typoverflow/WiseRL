import collections
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import gym
import imageio
import numpy as np
import torch

from wiserl.env.venvs import SubprocVectorEnv
from wiserl.algorithm.base import Algorithm

MAX_METRICS = {"success", "is_success", "completions"}
LAST_METRICS = {"goal_distance"}
MEAN_METRICS = {"discount"}
EXCLUDE_METRICS = {"TimeLimit.truncated"}


class EvalMetricTracker(object):
    """
    A simple class to make keeping track of eval metrics easy.
    Usage:
        Call reset before each episode starts
        Call step after each environment step
        call export to get the final metrics
    """

    def __init__(self, num_episodes: int = 1):
        self.metrics = collections.defaultdict(list)
        self.num_episodes = num_episodes
        self.ep_length = [0 for _ in range(num_episodes)]
        self.ep_reward = [0 for _ in range(num_episodes)]
        self.ep_metrics = [collections.defaultdict(list) for _ in range(num_episodes)]

    def reset(self) -> None:
        for index in range(self.num_episodes):
            if self.ep_length[index] > 0:
                # Add the episode to overall metrics
                self.metrics["reward"].append(self.ep_reward[index])
                self.metrics["length"].append(self.ep_length[index])
                for k, v in self.ep_metrics[index].items():
                    if k in MAX_METRICS:
                        self.metrics[k].append(np.max(v))
                    elif k in LAST_METRICS:  # Append the last value
                        self.metrics[k].append(v[-1])
                    elif k in MEAN_METRICS:
                        self.metrics[k].append(np.mean(v))
                    else:
                        self.metrics[k].append(np.sum(v))

                self.ep_length[index] = 0
                self.ep_reward[index] = 0
                self.ep_metrics[index] = collections.defaultdict(list)

    def step(self, reward: float, info: Dict, index=0) -> None:
        self.ep_length[index] += 1
        self.ep_reward[index] += reward
        for k, v in info.items():
            if (isinstance(v, float) or np.isscalar(v)) and k not in EXCLUDE_METRICS:
                self.ep_metrics[index][k].append(v)

    def add(self, k: str, v: Any):
        self.metrics[k].append(v)

    def export(self) -> Dict:
        # We have remaining episodes to log, make sure to get it.
        self.reset()
        metrics = {k: np.mean(v) for k, v in self.metrics.items()}
        metrics["reward_std"] = np.std(self.metrics["reward"])
        return metrics


@torch.no_grad()
def eval_offline(
    env: gym.Env,
    env_fn: Callable,
    algorithm: Algorithm,
    num_ep: int = 10,
    num_proc: Optional[int] = None,
    seed: int = 0,
    terminate_on_success: bool = False,
    deterministic: bool = True,
):
    if num_proc is None:
        metric_tracker = EvalMetricTracker()
        for i_ep in range(num_ep):
            ep_length, ep_reward = 0, 0
            obs, done = env.reset(), False
            metric_tracker.reset()
            while not done:
                batch = dict(obs=obs)
                batch = algorithm.format_batch(batch)
                action = algorithm.predict(batch, deterministic=deterministic)
                next_obs, reward, done, info = env.step(action)
                metric_tracker.step(reward, info)
                ep_reward += reward
                ep_length += 1
                if terminate_on_success and (
                    info.get("success", False) or info.get("is_success", False)
                ):
                    done = True
                obs = next_obs
            if hasattr(env, "get_normalized_score"):
                metric_tracker.add("score", env.get_normalized_score(ep_reward))
        return metric_tracker.export()
    else:
        assert num_ep % num_proc == 0, "num_ep must be divisible by num_proc"
        metric_tracker = EvalMetricTracker(num_episodes=num_proc)
        envs = SubprocVectorEnv([env_fn for _ in range(num_proc)])
        envs.seed(seed)
        for i in range(0, num_ep, num_proc):
            ep_lengths, ep_rewards = [0] * num_proc, [0] * num_proc
            obss, dones = envs.reset(), [False] * num_proc
            metric_tracker.reset()
            while not np.all(dones):
                batch = dict(obs=obss)
                batch = algorithm.format_batch(batch)
                actions = algorithm.predict(batch, deterministic=deterministic)
                next_obss, rewards, terminals, infos = envs.step(actions)
                for j, (reward, info) in enumerate(zip(rewards, infos)):
                    if not dones[j]:
                        metric_tracker.step(reward, info, index=j)
                        ep_rewards[j] += reward
                        ep_lengths[j] += 1
                for j, terminal in enumerate(terminals):
                    if terminal and not dones[j]:
                        dones[j] = True
                if terminate_on_success:
                    for j, info in enumerate(infos):
                        if info.get("success", False) or info.get("is_success", False):
                            dones[j] = True
                obss = next_obss
            if hasattr(env, "get_normalized_score"):
                for j in range(num_proc):
                    metric_tracker.add("score", env.get_normalized_score(ep_rewards[j]))
        envs.close()
        return metric_tracker.export()
