from typing import Any, Optional

import d4rl
import gym
import numpy as np
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
        padding_mode: str="right",
        reward_scale: Optional[float] = None,
        reward_shift: Optional[float] = None,
        reward_normalize: bool = False,
    ):
        super().__init__()
        assert mode in {"transition", "trajectory"}, "Supported mode for D4RLOfflineDataset: {transition, trajectory}."
        assert padding_mode in {"left", "right", "none", "Supported padding mode for D4RLOfflineDataset: {left, right, none}."}
        assert reward_scale is None == reward_shift is None, "reward_scale and reward_shift should be set simultaneously."
        assert not reward_normalize or reward_shift is None, "reward scale & shift and reward normalize can not be set simultaneously."

        self.env_name = env
        self.mode = mode
        self.padding_mode = padding_mode
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
            "mask": np.ones([len(obs_), 1], dtype=np.float32),
            "end": np.asarray(end_)[..., None],
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
        bs = 64
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
                return_[:, t] += return_[:, t+1]
            self.data["return"] = return_
            if "antmaze" in self.env_name:
                # normalization on antmaze may look weird, we borrow it from preference transformer
                # https://github.com/csmile-1006/PreferenceTransformer/blob/f71647bb075c8287e2f26aded78aa8f1ac176eb5/train_offline.py#L78 and
                # https://github.com/csmile-1006/PreferenceTransformer/blob/f71647bb075c8287e2f26aded78aa8f1ac176eb5/train_offline.py#L119
                min_return, max_return = self.data["return"][:, 0].min(), self.data["return"][:, 0].max()
                norm = 1000. / (max_return - min_return)
                self.data["reward"] *= norm
                self.data["reward"] -= 1.0
                self.data["reward"] *= self.data["mask"]

                # for i in range(self.data["reward"].shape[0]):
                #     self.data["reward"][i] -= (1. + self.data["return"][i, 0] / self.traj_len[i]) * self.data["mask"][i]
                return_ = self.data["reward"].copy()
                for t in reversed(range(return_.shape[1]-1)):
                    return_[:, t] += return_[:, t+1]
                self.data["return"] = return_
            else:
                prev_return_min, prev_return_max = return_[:, 0].min(), return_[:, 0].max()
                max_return = max(abs(return_[:, 0].max()), abs(return_[:, 0].min()), return_[:, 0].max()-return_[:, 0].min(), 1.0)
                norm = 1000. / max_return
                self.data["reward"] *= norm
                self.data["return"] *= norm
                print(f"[D4RLOfflineDataset]: return range: [{prev_return_min}, {prev_return_max}], multiplying norm factor {norm}.")
        elif self.mode == "transition":
            ep_reward_ = []
            ep_length_ = []
            episode_reward = 0
            episode_length = 0
            N = self.data["reward"].shape[0]
            for i in range(N):
                episode_reward += self.data["reward"][i]
                episode_length += 1
                if self.data["end"][i]:
                    ep_reward_.append(episode_reward)
                    ep_length_.append(episode_length)
                    episode_reward = episode_length = 0

            if "antmaze" in self.env_name:
                # normalization on antmaze may look weird, we borrow it from preference transformer
                # https://github.com/csmile-1006/PreferenceTransformer/blob/f71647bb075c8287e2f26aded78aa8f1ac176eb5/train_offline.py#L78 and
                # https://github.com/csmile-1006/PreferenceTransformer/blob/f71647bb075c8287e2f26aded78aa8f1ac176eb5/train_offline.py#L119
                min_return, max_return  = min(ep_reward_), max(ep_reward_)
                # idx = 0
                print("max return: ", max_return, " min_return: ", min_return)
                # for ep_len in ep_length_:
                #     for l in range(ep_len):
                #         self.data["reward"][idx] -= max_return / ep_len
                #         idx += 1
                # assert idx == N
                norm = 1000 / (max_return - min_return)
                # self.data["reward"] *= norm
                self.data["reward"] -= 1.0
                self.data["reward"] *= self.data["mask"]
                print(f"[D4RLOfflineDataset]: return range: [{min_return}, {max_return}], multiplying norm factor {norm}.")
            else:
                max_return = max(abs(min(ep_reward_)).item(), abs(max(ep_reward_)).item(), (max(ep_reward_)-min(ep_reward_)).item(), 1.0)
                norm = 1000 / max_return
                self.data["reward"] *= norm
                print(f"[D4RLOfflineDataset]: return range: [{min(ep_reward_)}, {max(ep_reward_)}], multiplying norm factor {norm}.")
