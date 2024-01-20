from typing import Any, Optional

import d4rl
import gym
import numpy as np
import torch

from wiserl.utils.functional import discounted_cum_sum


def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)

class D4RLOfflineDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        env: str,
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        mode: str="transition",
        reward_scale: Optional[float] = None,
        reward_shift: Optional[float] = None,
        reward_normalize: bool = False,
    ):
        super().__init__()
        assert mode in {"transition", "trajectory"}, "Supported mode for D4RLOfflineDataset: \{transition, trajectory\}."
        assert reward_scale is None == reward_shift is None, "reward_scale and reward_shift should be set simultaneously."
        assert not reward_normalize or reward_shift is None, "reward scale & shift and reward normalize can not be set simultaneously."

        self.env_name = env
        self.mode = mode
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length
        self.capacity = capacity

        self.reward_scale = reward_scale
        self.reward_shift = reward_shift
        self.reward_normalize = reward_normalize

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

    def load_dataset(self):
        env = gym.make(self.env_name)
        dataset = env.get_dataset()

        N = dataset["rewards"].shape[0]
        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        terminal_ = []
        end_ = []
        ep_reward_ = []

        use_timeouts = "timeouts" in dataset
        episode_step = 0
        episode_reward = 0
        for i in range(N-1):
            obs = dataset["observations"][i].astype(np.float32)
            next_obs = dataset["observations"][i+1].astype(np.float32)
            action = dataset["actions"][i].astype(np.float32)
            reward = dataset["rewards"][i].astype(np.float32)
            terminal = bool(dataset["terminals"][i])
            end = False
            episode_step += 1
            episode_reward += reward
            if use_timeouts:
                final_timestep = dataset["timeouts"][i]
            else:
                final_timestep = (episode_step == env._max_episode_steps)
            if final_timestep:
                if not terminal:
                    end_[-1] = True
                    ep_reward_.append(episode_reward - reward)
                    episode_step = 0
                    episode_reward = 0
                    continue
            if final_timestep or terminal:
                end = True
                ep_reward_.append(episode_reward)
                episode_step = 0
                episode_reward = 0
            obs_.append(obs)
            next_obs_.append(next_obs)
            action_.append(action)
            reward_.append(reward)
            terminal_.append(terminal)
            end_.append(end)
        end_[-1] = True

        data = {
            "obs": np.asarray(obs_),
            "action": np.asarray(action_),
            "next_obs": np.asarray(next_obs_),
            "reward": np.asarray(reward_)[..., None],
            "terminal": np.asarray(terminal_)[..., None],
            "mask": np.ones([len(obs_), 1], dtype=np.float32)
        }

        if self.reward_normalize:
            min_, max_ = min(ep_reward_), max(ep_reward_)
            data["reward"] = data["reward"] * env._max_episode_steps / (max_-min_)
        if self.reward_shift:
            data["reward"] = data["reward"] * self.reward_scale + self.reward_shift

        if self.mode == "transition":
            self.data_size = len(obs_)
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
            for i in range(len(reward_)):
                if end_[i]:
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
                self.max_len = self.traj_len.max() + self.segment_length - 1
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
