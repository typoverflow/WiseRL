import os
from typing import Any, Optional

import gym
import h5py
import numpy as np
import robomimic
import torch

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
        assert padding_mode in {"left", "right", "none", "Supported padding mode for RobomimicDataset: {left, right, none}."}
        self.env_name = env
        self.mode = mode
        self.padding_mode = padding_mode
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length
        self.capacity = capacity

        self.load_dataset()

    def __len__(self):
        return self.data_size

    def __iter__(self):
        while True:
            if self.mode == "transition" or (self.mode == "trajectory" and self.sample_full):
                idxs = np.random.randint(0, self.data_size, size=self.batch_size)
                idxs = np.squeeze(idxs)
                yield {
                    k: self.data[k][idxs] for k in self.data
                }
            else:
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
                        k: np.stack([s[k] for s in sample], axis=0) for k in sample[0].keys()
                    }

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
        end_ = []
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
            end = np.zeros_like(terminal, dtype=np.bool_)
            end[-1] = True
            assert len(obs) == len(next_obs) == len(action) == len(reward) == len(terminal) == len(end)
            obs_.append(obs)
            action_.append(action)
            next_obs_.append(next_obs)
            reward_.append(reward)
            terminal_.append(terminal)
            end_.append(end)
            ep_reward_.append(reward.sum())
            traj_len_.append(len(reward))

        if self.mode == "transition":
            self.data_size = sum([len(t) for t in obs_])
            self.data = {
                "obs": np.concatenate([t for t in obs_], axis=0),
                "action": np.concatenate([t for t in action_], axis=0),
                "next_obs": np.concatenate([t for t in next_obs_], axis=0),
                "reward": np.concatenate([t for t in reward_], axis=0)[..., None],
                "terminal": np.concatenate([t for t in terminal_], axis=0)[..., None],
                "end": np.concatenate([t for t in end_], axis=0)[..., None],
                "mask": np.ones([self.data_size, ], dtype=np.float32)[..., None],
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
                "terminal": terminal_,
            }
            trajs["return"] = [discounted_cum_sum(t, 1.0) for t in reward_]
            trajs["mask"] = [np.ones([len(t), ], dtype=np.float32) for t in reward_]
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
                    trajs[k][i_traj] = pad_along_axis(trajs[k][i_traj], pad_to=self.max_len)
            self.data = {
                "obs": np.stack(trajs["obs"], axis=0),
                "action": np.stack(trajs["action"], axis=0),
                "next_obs": np.stack(trajs["next_obs"], axis=0),
                "reward": np.stack(trajs["reward"], axis=0)[..., None],
                "terminal": np.stack(trajs["terminal"], axis=0)[..., None],
                "return": np.stack(trajs["return"], axis=0)[..., None],
                "mask": np.stack(trajs["mask"], axis=0)[..., None],
            }

        file.close()

    @torch.no_grad()
    def relabel_reward(self, agent):
        # CHECK: may be don't use reward normalization since sparse reward
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
            prev_return_min, prev_return_max = return_[:, 0].min(), return_[:, 0].max()
            max_return = max(abs(return_[:, 0].max()), abs(return_[:, 0].min()), return_[:, 0].max()-return_[:, 0].min(), 1.0)
            norm = 1000. / max_return
            self.data["reward"] *= norm
            self.data["return"] *= norm
            print(f"[D4RLOfflineDataset]: return range: [{prev_return_min}, {prev_return_max}], multiplying norm factor {norm}.")
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
            print(f"[RobomimicDataset]: return range: [{min(ep_reward_)}, {max(ep_reward_)}], multiplying norm factor {norm}.")


if __name__ == "__main__":
    from wiserl.env.robomimic_env import RobomimicEnv
    env = RobomimicEnv("Can-ph")

    dataset = RobomimicDataset(
        env.observation_space,
        env.action_space,
        "Can-mh",
        capacity=100,
        mode="trajectory",
        segment_length=64
    )
    dataset_iter = torch.utils.data.DataLoader(dataset)
    dataset_iter = iter(dataset_iter)
    sample = next(dataset_iter)
