import os
from typing import Any, Optional

import gym
import h5py
import numpy as np
import robomimic
import torch


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

prefix = "~/.robomimic/datasets"
DATASET_PATH = {
    "Can-mh": f"{prefix}/can/mh/low_dim.hdf5",
    "Can-ph": f"{prefix}/can/ph/low_dim.hdf5",
    "Lift-mh": f"{prefix}/lift/mh/low_dim.hdf5",
    "Lift-ph": f"{prefix}/lift/ph/low_dim.hdf5",
    "Square-mh": f"{prefix}/square/mh/low_dim.hdf5",
    "Square-ph": f"{prefix}/square/ph/low_dim.hdf5",
}


class RobomimicDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        env: str,
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        mode: str = "transition",
        padding_mode: str = "right",
    ):
        super().__init__()
        assert mode in {"transition", "trajectory"}, "Supported mode for D4RLOfflineDataset: {transition, trajectory}."
        assert padding_mode in {"left", "all", "none", "Supported padding mode for RobomimicDataset: {left, right, none}."}
        self.env_name = env
        self.mode = mode
        self.padding_mode = padding_mode
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length
        self.capacity = capacity

        self.load_dataset()

    def __len__(self):
        return self.data_size

    def load_dataset(self):
        keys = [
            "object",
            "robot0_joint_pos",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        ]
        path = DATASET_PATH[self.env_name]
        path = os.path.expanduser(path)
        file = h5py.File(path, "r")

        obs_ = []
        action_ = []
        next_obs_ = []
        reward_ = []
        terminal_ = []
        ep_reward_ = []
        traj_len_ = []
        demos = list(file["data"].keys())
        for i, demo in enumerate(demos):
            obs = np.concatenate([
                file["data"][demo]["obs"][k] for k in keys
            ], axis=1)
            next_obs = np.concatenate([
                file["data"][demo]["next_obs"][k] for k in keys
            ], axis=1)
            obs = np.asarray(obs, dtype=np.float32)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            action = np.asarray(file["data"][demo]["actions"], dtype=np.float32)
            reward = np.asarray(file["data"][demo]["rewards"], dtype=np.float32)
            terminal = np.asarray(file["data"][demo]["dones"], dtype=np.float32)
            terminal[-1] = 1.0
            assert len(obs) == len(next_obs) == len(action) == len(reward) == len(terminal)
            obs_.append(obs)
            action_.append(action)
            next_obs_.append(next_obs)
            reward_.append(reward)
            terminal_.append(terminal)
            ep_reward_.append(reward.sum())
            traj_len_.append(len(reward))

        if self.mode == "transition":
            self.data_size = sum([len(t) for t in obs_])
            self.data = {
                "obs": np.concatenate([t for t in obs_]),
                "action": np.concatenate([t for t in action_]),
                "next_obs": np.concatenate([t for t in next_obs_]),
                "reward": np.concatenate([t for t in reward_]),
                "terminal": np.concatenate([t for t in terminal_]),
                "mask": np.ones([self.data_size, 1], dtype=np.float32)
            }
            for k in self.data:
                assert len(self.data[k]) == self.data_size
            if self.capacity is not None:
                if self.capacity > self.data_size:
                    print(f"[Warning]: capacity {self.capacity} exceeds dataset size {self.data_size}.")
                self.data_size = min(self.data_size, self.capacity)
                self.data = {
                    k: v[:self.data_size] for k, v in self.data.items()
                }
        elif self.mode == "trajectory":
            trajs = {
                "obs": obs_,
                "action": action_,
                "next_obs": next_obs_,
                "reward": reward_,
                "terminal": terminal_
            }
            self.data_size = len(obs_)
            if self.capacity is not None:
                if self.capacity > self.data_size:
                    print(f"[Warning]: capacity {self.capacity} exceeds dataset size {self.data_size}.")
                self.data_size = min(self.data_size, self.capacity)
            self.traj_len = np.asarray(traj_len_[:self.data_size], dtype=np.int32)
            trajs = {k: v[:self.data_size] for k, v in trajs.items()}

            if self.segment_length is None:
                self.max_len = self.traj_len.max()
                self.segment_length = self.max_len
                self.sample_full = True
            else:
                self.max_len = self.traj_len.max()
                self.sample_prob = self.traj_len / self.traj_len.sum()
                self.sample_full = False

            for k, v in trajs.items():
                for i_traj in range(self.data_size):
                    trajs[k][i_traj] = pad_along_axis(v, pad_to=self.max_len)
            self.data = {
                "obs": np.stack(trajs["obs"], axis=0),
                "action": np.stack(trajs["action"], axis=0),
                "next_obs": np.stack(trajs["next_obs"], axis=0),
                "reward": np.stack(trajs["reward"], axis=0),
                "terminal": np.stack(trajs["terminal"], axis=0)
            }

        file.close()


if __name__ == "__main__":
    from wiserl.env.robomimic_env import RobomimicEnv
    env = RobomimicEnv("Can-ph")

    dataset = RobomimicDataset(
        env.observation_space,
        env.action_space,
        "Can-ph"
    )
