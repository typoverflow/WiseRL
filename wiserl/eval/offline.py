import collections
import os
from typing import Any, Dict, List, Optional, Sequence

import gym
import imageio
import numpy as np
import torch

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

    def __init__(self):
        self.metrics = collections.defaultdict(list)
        self.ep_length = 0
        self.ep_reward = 0
        self.ep_metrics = collections.defaultdict(list)

    def reset(self) -> None:
        if self.ep_length > 0:
            # Add the episode to overall metrics
            self.metrics["reward"].append(self.ep_reward)
            self.metrics["length"].append(self.ep_length)
            for k, v in self.ep_metrics.items():
                if k in MAX_METRICS:
                    self.metrics[k].append(np.max(v))
                elif k in LAST_METRICS:  # Append the last value
                    self.metrics[k].append(v[-1])
                elif k in MEAN_METRICS:
                    self.metrics[k].append(np.mean(v))
                else:
                    self.metrics[k].append(np.sum(v))

            self.ep_length = 0
            self.ep_reward = 0
            self.ep_metrics = collections.defaultdict(list)

    def step(self, reward: float, info: Dict) -> None:
        self.ep_length += 1
        self.ep_reward += reward
        for k, v in info.items():
            if (isinstance(v, float) or np.isscalar(v)) and k not in EXCLUDE_METRICS:
                self.ep_metrics[k].append(v)

    def add(self, k: str, v: Any):
        self.metrics[k].append(v)

    def export(self) -> Dict:
        if self.ep_length > 0:
            # We have one remaining episode to log, make sure to get it.
            self.reset()
        metrics = {k: np.mean(v) for k, v in self.metrics.items()}
        metrics["reward_std"] = np.std(self.metrics["reward"])
        return metrics


@torch.no_grad()
def eval_offline(
    env: gym.Env,
    algorithm: Algorithm,
    num_ep: int=10,
    terminate_on_success: bool=False,
    deterministic: bool=True
):
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
            if terminate_on_success and (info.get("success", False) or info.get("is_success", False)):
                done = True
            obs = next_obs
        if hasattr(env, "get_normalized_score"):
            metric_tracker.add("score", env.get_normalized_score(ep_reward))
    return metric_tracker.export()

@torch.no_grad()
def eval_offline_with_history_input(
    env: gym.Env,
    algorithm: Algorithm,
    num_ep: int=10,
    terminate_on_success: bool=False,
    deterministic: bool=True
):
    metric_tracker = EvalMetricTracker()
    for i_ep in range(num_ep):
        ep_length, ep_reward = 0, 0
        obs, done = env.reset(), False
        history = {
            'obs': np.expand_dims(obs,0),
            'action':np.zeros([1,algorithm.action_dim]),
            'timestep':np.array([0], dtype=np.int32),
        }
        metric_tracker.reset()
        while not done:
            action = algorithm.predict(history.copy(), deterministic=deterministic)
            next_obs, reward, done, info = env.step(action)
            metric_tracker.step(reward, info)
            ep_reward += reward
            ep_length += 1
            
            if terminate_on_success and (info.get("success", False) or info.get("is_success", False)):
                done = True
            
            history['obs'] = np.concatenate((history['obs'],np.expand_dims(next_obs,0)), axis=0)
            history['action'] = np.concatenate((history['action'][:-1],np.expand_dims(action,0), history['action'][-1:]), axis=0)
            history['timestep'] = np.concatenate((history['timestep'],np.array([ep_length],dtype=np.int32)), axis=0)
            
        if hasattr(env, "get_normalized_score"):
            metric_tracker.add("score", env.get_normalized_score(ep_reward))
    return metric_tracker.export()
