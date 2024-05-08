from typing import Any, Dict, Optional

import d4rl
import gym
import numpy as np
import torch

import wiserl.utils.utils as utils
from wiserl.env.cliffwalking_env import get_cliff_dataset
from wiserl.utils.functional import discounted_cum_sum


def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0, direction="right"
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    if direction == "right":
        npad[axis] = (0, pad_size)
    else:
        npad[axis] = (pad_size, 0)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


class CliffWalkingOfflineDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        mode: str="transition",
        padding_mode: str="right",
        config: Dict[str, int] = {"random": 10000},
        path: Optional[str] = None
    ):
        super().__init__()
        assert mode in {"transition", "trajectory"}, "Supported mode for D4RLOfflineDataset: {transition, trajectory}."
        assert padding_mode in {"left", "right", "none", "Supported padding mode for D4RLOfflineDataset: {left, right, none}."}

        self.mode = mode
        self.padding_mode = padding_mode
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length
        self.capacity = capacity
        self.config = config
        self.path = path

        self.load_dataset()

    def __len__(self):
        return self.data_size

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
            else :
                sample = []
                for _ in range(self.batch_size):
                    if self.padding_mode == "right":
                        traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                        start_idx = np.random.choice(self.traj_len[traj_idx])
                        s = {
                            k: pad_along_axis(v[traj_idx, start_idx:min(start_idx+self.segment_length, self.traj_len[traj_idx])], pad_to=self.segment_length) for k, v in self.data.items()
                        }
                        s["timestep"] = np.arange(start_idx, start_idx+self.segment_length)
                    elif self.padding_mode == "left":
                        traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                        end_idx = np.random.choice(self.traj_len[traj_idx])+1
                        s = {
                            k: pad_along_axis(v[traj_idx, max(0, end_idx-self.segment_length):end_idx], pad_to=self.segment_length, direction="left") for k, v in self.data.items()
                        }
                        s["timestep"] = np.maximum(np.arange(self.segment_length)+1-self.segment_length+end_idx, 0)
                    elif self.padding_mode == "none":
                        traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                        while self.traj_len[traj_idx] < self.segment_length:
                            traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                        start_idx = np.random.choice(self.traj_len[traj_idx]-self.segment_length+1)
                        s = {
                            k: v[traj_idx, start_idx:start_idx+self.segment_length] for k, v in self.data.items()
                        }
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

    def load_dataset(self):
        if self.path is not None:
            print(f"[CliffWalkingOfflineDataset]: Loading dataset from {self.path} ...")
            dataset = np.load(self.path)
        else:
            sep_dataset = get_cliff_dataset(self.config)
            dataset = None
            for k, v in sep_dataset:
                if dataset is None:
                    dataset = v
                else:
                    dataset = {kk: np.concatenate([dataset[kk], v[kk]], dim=0) for kk in dataset.keys()}
        data = {
            "obs": dataset["obs"],
            "action": dataset["action"],
            "next_obs": dataset["next_obs"],
            "reward": dataset["reward"][..., None],
            "terminal": dataset["terminal"][..., None],
            "mask": np.ones_like(dataset["reward"])[..., None],
            "end": np.zeros_like(dataset["reward"], dtype=np.bool_)[..., None]
        }
        end_idx = np.cumsum(dataset["traj_len"])
        data["end"][end_idx-1] = True

        if self.mode == "transition":
            self.data_size = len(data["obs"])
            if self.capacity is not None:
                if self.capacity > self.data_size:
                    print(f"[Warning]: capacity {self.capacity} exceeds dataset size {self.data_size}")
                self.data_size = min(self.data_size, self.capacity)
                data = {
                    k: v[:self.data_size] for k, v in data.items()
                }
            self.data = data
        elif self.mode == "trajectory":
            traj, traj_len = [], []
            traj_start = 0
            for i in range(len(data["reward"])):
                if data["end"][i]:
                    episode_data = {
                        k: v[traj_start:i+1] for k, v in data.items()
                    }
                    episode_data["return"] = discounted_cum_sum(episode_data["reward"], 1.0)
                    traj.append(episode_data)
                    traj_len.append(i+1-traj_start)
                    traj_start = i+1
            self.traj_len = np.asarray(traj_len)
            self.data_size = len(self.traj_len)
            if self.capacity is not None:
                if self.capacity > self.data_size:
                    print(f"[Warning]: capacity {self.capacity} exceeds dataset size {self.data_size}")
                self.data_size = min(self.data_size, self.capacity)
                traj = traj[:self.data_size]
                self.traj_len = self.traj_len[:self.data_size]

            if self.segment_length is None:
                self.max_len = self.traj_len.max()
                self.segment_length = self.max_len
                self.sample_full = True
            else:
                self.max_len = self.traj_len.max()
                self.sample_prob = self.traj_len / self.traj_len.sum()
                self.sample_full = False

            for i_traj in range(self.data_size):
                for _key, _value in traj[i_traj].items():
                    traj[i_traj][_key] = pad_along_axis(_value, pad_to=self.max_len)
            self.data = {
                "obs": np.asarray([t["obs"] for t in traj]),
                "action": np.asarray([t["action"] for t in traj]),
                "next_obs": np.asarray([t["next_obs"] for t in traj]),
                "reward": np.asarray([t["reward"] for t in traj]),
                "terminal": np.asarray([t["terminal"] for t in traj]),
                "return": np.asarray([t["return"] for t in traj]),
                "mask": np.asarray([t["mask"] for t in traj]),
            }
        del env

    @torch.no_grad()
    def relabel_reward(self, agent):
        assert hasattr(agent, "select_reward"), f"Agent {agent} must support relabel_reward!"
        bs = 256
        for i_batch in range((self.data_size-1) // bs + 1):
            idx = np.arange(i_batch*bs, min((i_batch+1)*bs, self.data_size))
            batch = {
                "obs": self.data["obs"][idx],
                "action": self.data["action"][idx],
                "next_obs": self.data["next_obs"][idx],
                "mask": self.data["mask"][idx]
            }
            batch = agent.format_batch(batch)
            reward = agent.select_reward(batch).detach().cpu().numpy()
            reward = reward * self.data["mask"][idx]
            self.data["reward"][idx] = reward

        if self.mode == "trajectory":
            # CHECK: may be bug. the max and min returns are not consistent with those computed in transition mode
            return_ = self.data["reward"].copy()
            for t in reversed(range(return_.shape[1]-1)):
                return_[t] += return_[t+1]
            self.data["return"] = return_
            # normalization
            max_return = max(abs(return_[:, 0].max()), abs(return_[:, 0].min()), return_[:, 0].max()-return_[:, 0].min(), 1.0)
            norm = 1000 / max_return
            self.data["reward"] *= norm
            self.data["return"] *= norm
            print(f"[D4RLOfflineDataset]: return range: [{return_[:,0].min()}, {return_[:, 0].max()}], multiplying norm factor {norm}.")
        elif self.mode == "transition":
            ep_reward_ = []
            episode_reward = 0
            N = self.data["reward"].shape[0]
            for i in range(N):
                episode_reward += self.data["reward"][i]
                if self.data["end"][i]:
                    ep_reward_.append(episode_reward)
                    episode_reward = 0
            max_return = max(abs(min(ep_reward_)).item(), abs(max(ep_reward_)).item(), (max(ep_reward_)-min(ep_reward_)).item(), 1.0)
            norm = 1000 / max_return
            self.data["reward"] *= norm
            print(f"[D4RLOfflineDataset]: return range: [{min(ep_reward_)}, {max(ep_reward_)}], multiplying norm factor {norm}.")


class CliffWalkingComparisonDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space,
        action_space,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        path: Optional[str] = None
    ):
        super().__init__()
        self.batch_size = 1 if batch_size is None else batch_size

        assert path is not None
        dataset = np.load(path)
        dataset = {k: np.asarray(v) for k, v in dataset.items()}
        if capacity is not None:
            dataset = utils.get_from_batch(dataset, 0, capacity)
        dataset = utils.remove_float64(dataset)

        self.data = dataset
        self.data_size, self.data_segment_length = dataset["obs_1"].shape[0:2]

    def __len__(self):
        return self.data_size

    def sample_idx(self, idx):
        idx = np.squeeze(idx)
        batch = {
            "obs_1": self.data["obs_1"][idx],
            "obs_2": self.data["obs_2"][idx],
            "action_1": self.data["action_1"][idx],
            "action_2": self.data["action_2"][idx],
            "label": self.data["label"][idx][:, None],
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
