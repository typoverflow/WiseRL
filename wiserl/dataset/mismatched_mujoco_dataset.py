from typing import Any, Optional

import gym
import numpy as np
import torch

from wiserl.utils.functional import discounted_cum_sum

PREFIX = "datasets/variant-world"
DATASET_PATH = {
    "HalfCheetah-gravity0.8": f"{PREFIX}/gravity-08/HalfCheetah-v3/seed0/data.npz",
    "HalfCheetah-gravity1.2": f"{PREFIX}/gravity-12/HalfCheetah-v3/seed0/data.npz",
    "Walker2d-gravity0.8": f"{PREFIX}/gravity-08/Walker2d-v3/seed0/data.npz",
    "Walker2d-gravity1.2": f"{PREFIX}/gravity-12/Walker2d-v3/seed0/data.npz",
}


class MismatchedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        env: str,
        capacity: Optional[int] = None,
        mode: str = "transition",
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        assert mode in {"transition", "trajectory"}, "Supported mode for VariantWorldDataset: \{transition, trajectory\}."

        self.env_name = env
        self.capacity = capacity
        self.mode = mode
        self.segment_length = segment_length
        self.batch_size = 1 if batch_size is None else batch_size

        self.load_dataset()

    def __len__(self):
        return self.data_size

    def load_dataset(self):
        with np.load(f"{DATASET_PATH[self.env_name]}") as data_:
            loaded_data = {k: data_[k] for k in data_.files}

        if self.mode == "transition":
            indices = np.squeeze(loaded_data["masks"], axis=-1).astype(np.bool_)
            data = {
                k: v[indices] for k, v in loaded_data.items() if k in {
                    "obs", "action", "next_obs", "reward", "terminal"
                }
            }
            self.data_size = int(loaded_data["masks"].sum())
            if self.capacity is not None:
                if self.capacity > self.data_size:
                    print(f"[Warning]: capacity {self.capacity} exceeds dataset size {self.data_size}")
                self.data_size = min(self.data_size, self.capacity)
                data = {
                    k: v[:self.data_size] for k, v in data.items()
                }
            self.data = data
        elif self.mode == "trajectory":
            self.data_size = loaded_data["observations"].shape[0]
            if self.capacity is not None:
                if self.capacity > self.data_size:
                    print(f"[Warning]: capacity {self.capacity} exceeds dataset size {self.data_size}")
                self.data_size = min(self.data_size, self.capacity)
            data = {
                k: v[:self.data_size] for k, v in loaded_data.items()
            }
            self.traj_len = data["masks"].sum((-2, -1)).astype(np.int32)
            self.data = data

            if self.segment_length is None:
                self.max_len = self.traj_len.max()
                self.segment_length = self.max_len
                self.sample_full = True
            else:
                self.max_len = self.traj_len.max()
                self.sample_prob = self.traj_len / self.traj_len.sum()
                self.sample_full = False
        del loaded_data


class MismatchedOfflineDataset(MismatchedDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        env: str,
        capacity: Optional[int] = None,
        mode: str = "transition",
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        super().__init__(observation_space, action_space, env, capacity, mode, segment_length, batch_size)

    def __iter__(self):
        while True:
            if self.mode == "transition" or (self.mode == "trajectory" and self.sample_full):
                idxs = np.random.randint(0, self.data_size, size=self.batch_size)
                idxs = np.squeeze(idxs)
                yield {
                    "obs": self.data["obs"][idxs],
                    "action": self.data["action"][idxs],
                    "next_obs": self.data["next_obs"][idxs],
                    "reward": self.data["reward"][idxs],
                    "terminal": self.data["terminal"][idxs],
                    "mask": self.data["mask"][idxs]
                }
            else:
                sample = []
                for _ in range(self.batch_size):
                    traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                    start_idx = np.random.choice(self.traj_len[traj_idx])
                    s = {k: v[traj_idx, start_idx:start_idx+self.segment_length] for k, v in self.data.items()}
                    s["timestep"] = np.arange(start_idx, start_idx+self.segment_length)
                    sample.append(s)
                if len(sample) == 1: yield sample[0]
                else:
                    yield {
                        "obs": np.stack([s["obs"] for s in sample], axis=0),
                        "action": np.stack([s["action"] for s in sample], axis=0),
                        "next_obs": np.stack([s["next_obs"] for s in sample], axis=0),
                        "reward": np.stack([s["reward"] for s in sample], axis=0),
                        "terminal": np.stack([s["terminal"] for s in sample], axis=0),
                        "return": np.stack([s["return"] for s in sample], axis=0),
                        "mask": np.stack([s["mask"] for s in sample], axis=0),
                        "timestep": np.stack([s["timestep"] for s in sample], axis=0),
                    }


class MismatchedComparisonDataset(MismatchedDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        env: str,
        capacity: Optional[int] = None,
        mode: str = "trajectory",
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None
    ):
        assert mode == "trajectory", "MismatchedComparisonDataset should be replayed in trajectories."
        super().__init__(observation_space, action_space, env, capacity, mode, segment_length, batch_size)
        self.min_len = self.traj_len.min()
        if segment_length is not None and self.min_len < segment_length:
            print("[Warning]: desired segment length exceeds dataset minimum length.")


    def __iter__(self):
        while True:
            if self.sample_full:
                raise NotImplementedError
            else:
                idx1 = np.random.randint(0, self.data_size, size=self.batch_size)
                idx2 = np.random.randint(0, self.data_size, size=self.batch_size)
                sample = []
                for tidx1, tidx2 in zip(idx1, idx2):
                    sidx1 = np.random.randint(0, self.traj_len[tidx1]-self.segment_length)
                    sidx2 = np.random.randint(0, self.traj_len[tidx2]-self.segment_length)
                    p1 = self.data["q_values"][tidx1, sidx1:sidx1+self.segment_length].sum() - self.data["v_values"][tidx1, sidx1:sidx1+self.segment_length].sum()
                    p2 = self.data["q_values"][tidx2, sidx2:sidx2+self.segment_length].sum() - self.data["v_values"][tidx2, sidx2:sidx2+self.segment_length].sum()
                    label = (1.0 * (p1 < p2) + 0.5 * (p1 == p2)).astype(np.float32)
                    sample.append({
                        "obs_1": self.data["obs"][tidx1, sidx1:sidx1+self.segment_length],
                        "obs_2": self.data["obs"][tidx2, sidx2:sidx2+self.segment_length],
                        "action_1": self.data["action"][tidx1, sidx1:sidx1+self.segment_length],
                        "action_2": self.data["action"][tidx2, sidx2:sidx2+self.segment_length],
                        "reward_1": self.data["reward"][tidx1, sidx1:sidx1+self.segment_length],
                        "reward_2": self.data["reward"][tidx2, sidx2:sidx2+self.segment_length],
                        "terminal_1": self.data["terminal"][tidx1, sidx1:sidx1+self.segment_length],
                        "terminal_2": self.data["terminal"][tidx2, sidx2:sidx2+self.segment_length],
                        "label": label
                    })
                if len(sample == 1): yield sample[0]
                else:
                    yield {
                        k: np.stack([s[k] for s in sample], axis=0) for k in sample[0].keys()
                    }
