from typing import Any, Dict, Optional

import d4rl
import gym
import numpy as np
import torch

import wiserl.utils.utils as utils


class FourRoomsComparisonDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        observation_space: gym.Space, 
        action_space: gym.Space, 
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        goal_state=(1, 11), 
        path: Optional[str] = None, 
        mode: str = "sparse", 
        noisy_label: bool = False
    ):
        super().__init__()
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length
        self.capacity = capacity
        self.path = path
        self._goal_x = goal_state[0]
        self._goal_y = goal_state[1]
        self.mode = mode
        self.noisy_label = noisy_label
        if mode == "sparse":
            self.capacity = 2 * capacity
        
        from wiserl.env.gridworld_env import OptimalAgent
        self.optimal_agent = OptimalAgent()
        self.load_dataset()

    def __len__(self):
        return self.data_size
    
    def distance_to_goal(self, state):
        return np.abs(state[..., 0] - self._goal_x) + np.abs(state[..., 1] - self._goal_y)

    def load_dataset(self):
        if self.path is not None:
            print(f"[FourRoomsComparisonDataset]: Loading dataset from {self.path} ...")
            dataset = np.load(self.path)
        else:
            assert False
        data = {
            "obs": np.asarray(dataset["obs"], dtype=np.float32),
            "action": np.asarray(dataset["action"], dtype=np.float32),
            "next_obs": np.asarray(dataset["next_obs"], dtype=np.float32),
        }
        self.data_size = len(data["obs"])
        self.traj_len = np.ones([self.data_size, ], dtype=np.int32) * data["obs"].shape[1]
        if self.capacity is not None:
            if self.capacity > self.data_size:
                print(f"[Warning]: capacity {self.capacity} exceeds dataset size {self.data_size}")
            data = {k: v[:self.capacity] for k, v in data.items()}
            self.data_size = min(self.data_size, self.capacity)
            self.traj_len = self.traj_len[:self.data_size]
        self.data = data
        self.data_segment_length = self.traj_len[0]
        
    def sample_idx(self, traj_idx, noisy_label=False):
        if self.mode == "sparse":
            traj_idx1 = traj_idx
            traj_idx2 = traj_idx + self.data_size // 2
        else:
            traj_idx1 = traj_idx
            traj_idx2 = np.random.randint(self.data_size, size=[self.batch_size, ])
        if self.segment_length is not None:
            start_1 = np.random.randint(0, self.traj_len[traj_idx1] - self.segment_length)
            start_2 = np.random.randint(0, self.traj_len[traj_idx2] - self.segment_length)
            end_1 = int(start_1 + self.segment_length)
            end_2 = int(start_2 + self.segment_length)
        else:
            start_1 = start_2 = 0
            end_1 = end_2 = self.data_segment_length
        traj_idx1, traj_idx2 = np.squeeze(traj_idx1), np.squeeze(traj_idx2)
        batch = {
            "obs_1": self.data["obs"][traj_idx1, start_1:end_1], 
            "action_1": self.data["action"][traj_idx1, start_1:end_1],
            "next_obs_1": self.data["next_obs"][traj_idx1, start_1:end_1],
            "obs_2": self.data["obs"][traj_idx2, start_2:end_2],
            "action_2": self.data["action"][traj_idx2, start_2:end_2],
            "next_obs_2": self.data["next_obs"][traj_idx2, start_2:end_2],
        }
        # init_distance_1 = self.distance_to_goal(batch["obs_1"][..., 0, :])
        # final_distance_1 = self.distance_to_goal(batch["next_obs_1"][..., -1, :])
        # init_distance_2 = self.distance_to_goal(batch["obs_2"][..., 0, :])
        # final_distance_2 = self.distance_to_goal(batch["next_obs_2"][..., -1, :])
        # return_1 = init_distance_1 - final_distance_1
        # return_2 = init_distance_2 - final_distance_2
        return_1 = self.optimal_agent.evaluate(batch["obs_1"], batch["action_1"])
        return_2 = self.optimal_agent.evaluate(batch["obs_2"], batch["action_2"])
        return_1, return_2 = return_1.sum(1), return_2.sum(1)
        if noisy_label:
            label = 1 / (1 + np.exp(0.5*(return_1 - return_2)))
            batch["label"] = label.astype(np.float32)[:, None]
        else:
            hard_label = 1.0 * (return_1 < return_2)
            soft_label = 0.5 * (return_1 == return_2)
            batch["label"] = (hard_label + soft_label).astype(np.float32)[:, None]
        return batch

    def __iter__(self):
        while True:
            if self.mode == "sparse":
                traj_idx1 = np.random.randint(self.data_size // 2, size=[self.batch_size, ])
            else:
                traj_idx1 = np.random.randint(self.data_size, size=[self.batch_size, ])
            yield self.sample_idx(traj_idx1, noisy_label=self.noisy_label)

    def create_sequential_iter(self):
        assert self.mode == "sparse"
        start, end = 0, min(self.batch_size, self.data_size // 2)
        while start < self.data_size//2:
            idxs = np.asarray(range(start, min(end, self.data_size//2)), dtype=np.int32)
            yield self.sample_idx(idxs, noisy_label=False)
            start += self.batch_size
            end += self.batch_size
            


class FourRoomsOfflineDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        observation_space: gym.Space,
        action_space: gym.Space,
        batch_size: Optional[int] = None, 
        segment_length: Optional[int] = None, 
        capacity: Optional[int] = None,
        path: Optional[str] = None
    ):
        super().__init__()
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length
        self.capacity = capacity
        self.path = path
        self.load_dataset()
        
    def load_dataset(self):
        if self.path is not None:
            print(f"[FourRoomsOfflineDataset]: Loading dataset from {self.path} ...")
            dataset = np.load(self.path)
        else:
            assert False
        data = {
            "obs": np.asarray(dataset["obs"], dtype=np.float32),
            "action": np.asarray(dataset["action"], dtype=np.float32),
            "next_obs": np.asarray(dataset["next_obs"], dtype=np.float32),
        }
        self.data_size = len(data["obs"])
        self.traj_len = np.ones([self.data_size, ], dtype=np.int32) * data["obs"].shape[1]
        if self.capacity is not None:
            if self.capacity > self.data_size:
                print(f"[Warning]: capacity {self.capacity} exceeds dataset size {self.data_size}")
            data = {k: v[:self.capacity] for k, v in data.items()}
            self.data_size = min(self.data_size, self.capacity)
            self.traj_len = self.traj_len[:self.data_size]
        
        data_shape = data["obs"].shape[:-1]
        data["reward"] = np.zeros([*data_shape, 1], dtype=np.float32)
        data["terminal"] = np.zeros([*data_shape, 1], dtype=np.float32)
        data["mask"] = np.ones([*data_shape, 1], dtype=np.float32)

        self.data = data
        self.data_segment_length = self.traj_len[0]

    def __len__(self):
        return self.data_size

    def __iter__(self):
        while True:
            sample = []
            for _ in range(self.batch_size):
                traj_idx = np.random.choice(self.data_size)
                start_idx = np.random.choice(self.traj_len[traj_idx] - self.segment_length+1)
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
                    "mask": np.stack([s["mask"] for s in sample], axis=0),
                    "timestep": np.stack([s["timestep"] for s in sample], axis=0),
                }

    @torch.no_grad()
    def relabel_reward(self, agent):
        assert hasattr(agent, "select_reward"), f"Agent {agent} must support relable_reward!"
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
            
        return_ = self.data["reward"].copy()
        for t in reversed(range(return_.shape[1]-1)):
            return_[t] += return_[t+1]
        self.data["return"] = return_
        

