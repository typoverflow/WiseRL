from typing import Any, Optional

import d4rl
import gym
import numpy as np
import torch

from wiserl.utils import utils
from wiserl.utils.functional import discounted_cum_sum

prefix = "datasets/ipl/metaworld/"
DATASET_PATH = {
    "assembly-v2": f"{prefix}/assembly_10000.npz",
    "button-press-v2": f"{prefix}/button-press_10000.npz",
    "door-close-v2": f"{prefix}/door-close_10000.npz",
    "drawer-open-v2": f"{prefix}/drawer-open_10000.npz",
    "plate-slide-v2": f"{prefix}/plate-slide_10000.npz",
    "sweep-into-v2": f"{prefix}/sweep-into_10000.npz",
}


class MetaworldComparisonDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space,
        action_space,
        env: str,
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
    ):
        super().__init__()
        assert env in DATASET_PATH.keys(), f"Env {env} not registered for PT dataset"

        self.env_name = env
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length
        path = DATASET_PATH[self.env_name]
        with open(path, "rb") as f:
            data = np.load(f)
            data = utils.nest_dict(data)
            if capacity is not None:
                data = utils.get_from_batch(data, 0, capacity)
        data = utils.remove_float64(data)
        lim = 1 - 1e-8
        data["action_1"] = np.clip(data["action_1"], a_min=-lim, a_max=lim)
        data["action_2"] = np.clip(data["action_2"], a_min=-lim, a_max=lim)

        self.data = data
        self.data_size, self.data_segment_length = data["action_1"].shape[:2]

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
            "action_1": self.data["action_1"][idx, start_idx:end_idx],
            "action_2": self.data["action_2"][idx, start_idx:end_idx],
            "label": self.data["label"][idx][:, None],
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


class MetaworldOfflineDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        env: str,
        # segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        mode: str = "transition"
    ):
        super().__init__()
        assert mode in {"transition", "trajectory"}

        self.env_name = env
        self.batch_size = 1 if batch_size is None else batch_size
        # self.segment_length = segment_length
        self.capacity = capacity

        self.load_dataset()

    def __len__(self):
        return self.data_size

    def __iter__(self):
        while True:
            idxs = np.random.randint(0, self.data_size, self.batch_size)
            idxs = np.squeeze(idxs)
            traj_len = self.data["obs"][0].shape[0]
            mask = np.ones([self.batch_size, traj_len, 1], dtype=np.float32)
            timestep = np.stack([np.arange(traj_len) for _ in idxs], axis=0)
            yield {
                "obs": self.data["obs"][idxs],
                "next_obs": self.data["next_obs"][idxs],
                "action": self.data["action"][idxs],
                "next_action": self.data["next_action"][idxs],
                "reward": self.data["reward"][idxs],
                "terminal": self.data["terminal"][idxs],
                "mask": mask,
                "timestep": timestep,
            }

    def load_dataset(self):
        path = DATASET_PATH.get(self.env_name, None)
        with open(path, "rb") as f:
            data = np.load(f)
            data = utils.nest_dict(data)
            if self.capacity is not None:
                data = utils.get_from_batch(data, 0, self.capacity)
        data = utils.remove_float64(data)
        lim = 1 - 1e-8
        data["action_1"] = np.clip(data["action_1"], a_min=-lim, a_max=lim)
        data["action_2"] = np.clip(data["action_2"], a_min=-lim, a_max=lim)
        N, L = data["obs_1"].shape[:2]

        data = {
            "obs": np.stack([data["obs_1"], data["obs_2"]], axis=0).reshape(2*N, L, -1)[:, :-1],
            "next_obs": np.stack([data["obs_1"], data["obs_2"]], axis=0).reshape(2*N, L, -1)[:, 1:],
            "action": np.stack([data["action_1"], data["action_2"]], axis=0).reshape(2*N, L, -1)[:, :-1],
            "next_action": np.stack([data["action_1"], data["action_2"]], axis=0).reshape(2*N, L, -1)[:, 1:],
        }
        data["terminal"] = np.zeros([2*N, L-1, 1], dtype=np.bool_)
        data["reward"] = np.zeros([2*N, L-1, 1], dtype=np.float32)
        data["mask"] = np.ones([2*N, L-1, 1], dtype=np.float32)

        self.traj_len = np.asarray([o.shape[0] for o in data["obs"]])
        self.data_size = len(self.traj_len)
        if self.capacity is not None:
            if self.capacity > self.data_size:
                print(f"[Warning]: capacity {self.capacity} exceeds dataset size {self.data_size}")
            self.data_size = min(self.data_size, self.capacity)
            data = {
                k: data[k][:self.data_size] for k in data
            }
            self.traj_len = self.traj_len[:self.data_size]
        self.data = data

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
                "next_action": self.data["next_action"][idx],
                "mask": self.data["mask"][idx]
            }
            batch = agent.format_batch(batch)
            reward = agent.select_reward(batch).detach().cpu().numpy()
            reward = reward * self.data["mask"][idx]
            self.data["reward"][idx] = reward

        return_ = self.data["reward"].sum(1)
        max_return = max(
            abs(return_.max()),
            abs(return_.min()),
            return_.max() - return_.min(),
            1.0
        )
        norm = 500. / max_return
        self.data["reward"] *= norm
        print(f"[MetaworldOfflineDataset]: return range: [{return_.min()}, {return_.max()}], multiplying norm factor {norm}.")

if __name__ == "__main__":
    dataset = MetaworldOfflineDataset(
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(11,)),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(3,)),
        env="assembly",
        # segment_length=100,
        batch_size=256,
        capacity=10000,
        # mode="transition"
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=None)
    iter = iter(loader)
    next(iter)
    pass
