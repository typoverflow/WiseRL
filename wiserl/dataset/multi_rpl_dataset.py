import os
import pickle
from typing import Dict, Optional, Union

import d4rl
import gym
import numpy as np
import torch

from wiserl.utils import utils

prefix = "datasets/rpl"

class MultiRPLComparisonDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space,
        action_space,
        env: str,
        num_tasks: int = 4,
        task_name: str = '1.0_0.5_0.1_5.0',
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        label_key: str="rl_sum",
        odrl: bool = False,
        variant: str = "gravity-50",
        eval: bool = False,
        replay: bool = False,
    ):
        super().__init__()

        self.env_name = env
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length
        self.label_key = label_key + '_label'
        self.variant = variant
        self.eval = eval
        train_or_eval = "eval" if eval else "train"
        #mid_name = f"collect_odrl/{self.env_name}" if odrl else f"{self.env_name}/{variant}"
        self.tasks = task_name.split('_')
        task_prefix = '-'.join(env.split('-')[:-1])
        #assert len(self.tasks) == num_tasks
        num_tasks = len(self.tasks)

        for i in range(num_tasks):
            mid_name = f"collect_odrl/{task_prefix+'-'+self.tasks[i]}" if odrl else f"{self.env_name}/{task_prefix+variant}"
            if replay:
                path = f"{prefix}/{mid_name}/replay_preference_{train_or_eval}_data.npz"
            else:
                path = f"{prefix}/{mid_name}/preference_{train_or_eval}_data.npz"        
            with open(path, "rb") as f:
                data = np.load(f)
                data = utils.nest_dict(data)
                if capacity is not None:
                    data = utils.get_from_batch(data, 0, capacity)
            data = utils.remove_float64(data)
            lim = 1 - 1e-8
            data["action_1"] = np.clip(data["action_1"], a_min=-lim, a_max=lim)
            data["action_2"] = np.clip(data["action_2"], a_min=-lim, a_max=lim)
            data["task_id"] = np.full_like(data["reward_1"], i)
            if i == 0:
                self.data = data
            else:
                for key, value in data.items():
                    self.data[key] = np.concatenate((self.data[key], value), axis=0)
        self.data_size, self.data_segment_length = self.data["action_1"].shape[:2]

    def __len__(self):
        return self.data_size

    def sample_idx(self, idx):
        idx = np.squeeze(idx)
        is_batch = len(idx.shape) > 0
        if self.segment_length is not None:
            start_idx = np.random.randint(self.data_segment_length - self.segment_length)
            end_idx = start_idx + self.segment_length
        else:
            start_idx, end_idx = 0, self.data_segment_length
        batch = {
            "obs_1": self.data["obs_1"][idx, start_idx:end_idx],
            "obs_2": self.data["obs_2"][idx, start_idx:end_idx],
            "next_obs_1": self.data["next_obs_1"][idx, start_idx:end_idx],
            "next_obs_2": self.data["next_obs_2"][idx, start_idx:end_idx],
            "action_1": self.data["action_1"][idx, start_idx:end_idx],
            "action_2": self.data["action_2"][idx, start_idx:end_idx],
            "label": self.data[self.label_key][idx][:, None],
            "reward_1": self.data["reward_1"][idx, start_idx:end_idx],
            "reward_2": self.data["reward_2"][idx, start_idx:end_idx],
            "task_id": self.data["task_id"][idx, start_idx:end_idx],
            "terminal_1": np.zeros([len(idx), end_idx-start_idx, 1], dtype=np.float32) \
                if is_batch else np.zeros([end_idx-start_idx, 1], dtype=np.float32),
            "terminal_2": np.zeros([len(idx), end_idx-start_idx, 1], dtype=np.float32) \
                if is_batch else np.zeros([end_idx-start_idx, 1], dtype=np.float32)
        }
        return batch

    def __iter__(self):
        while True:
            idxs = np.random.randint(0, len(self), size=self.batch_size)
            yield self.sample_idx(idxs)

    def create_sequential_iter(self):
        start, end = 0, min(self.batch_size, self.data_size)
        while start < self.data_size:
            idxs = list(range(start, min(end, self.data_size)))
            yield self.sample_idx(idxs)
            start += self.batch_size
            end += self.batch_size
