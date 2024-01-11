import os
import pickle
from typing import Dict, Optional, Union

import d4rl
import gym
import numpy as np
import torch

from wiserl.utils import utils

DATASET_PATH={
    task: f"datasets/pt/{task}" for task in [
        "antmaze-large-diverse-v2",
        "antmaze-large-play-v2",
        "antmaze-medium-diverse-v2",
        "antmaze-medium-play-v2",
        "hammer-cloned-v1",
        "hammer-human-v1",
        "pen-cloned-v1",
        "pen-human-v1",
        "walker2d-medium-expert-v2",
        "walker2d-medium-replay-v2",
        "hopper-medium-expert-v2",
        "hopper-medium-replay-v2",
        "Square-mh",
        "Square-ph",
        "Lift-mh",
        "Lift-ph",
        "Can-mh",
        "Can-ph"
    ]
}

class PTComparisonOfflineDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space,
        action_space,
        env: str,
        capacity: Optional[int] = None,
        segment_length: Optional[int] = None,
        label_key: str="human",
    ) -> None:
        super().__init__()
        assert env in DATASET_PATH.keys(), f"Env {env} not registered for PT dataset."
        assert label_key in {"script", "human"}, f"PTComparisonOfflineDataset does not support label_key: {label_key}."

        self.env_name = env
        self.obs_shape = observation_space.shape[0]
        self.action_shape = action_space.shape[0]
        self.capacity = capacity
        self.segment_length = segment_length
        self.label_key = label_key  # range: ["script", "human"]
        self.load_dataset()

    def load_dataset(self):
        base_path = DATASET_PATH.get(self.env_name)
        human_indices_2_file, human_indices_1_file, human_labels_file = sorted(os.listdir(base_path))
        with open(os.path.join(base_path, human_indices_1_file), "rb") as fp:
            human_indices_1 = pickle.load(fp)
        with open(os.path.join(base_path, human_indices_2_file), "rb") as fp:
            human_indices_2 = pickle.load(fp)
        with open(os.path.join(base_path, human_labels_file), "rb") as fp:   # Unpickling
            human_labels = pickle.load(fp)

        if self.env_name.startswith(("hammer", "pen", "walker2d", "hopper")):
            env = gym.make(self.env_name)
            dataset = d4rl.qlearning_dataset(env)
            dataset["terminals"] = dataset["terminals"].astype(np.float32)

        # compute the trajectory idx list
        N = dataset["rewards"].shape[0]
        use_timeouts = False
        if "timeouts" in dataset:
            use_timeouts = True
        episode_step = 0
        start_idx, data_idx = 0, 0
        trj_idx_list = []
        for i in range(N-1):
            if "maze" in self.env_name:
                done_bool = sum(dataset["infos/goal"][i+1] - dataset["infos/goal"][i]) > 0
            else:
                done_bool = bool(dataset["terminals"][i])
            if use_timeouts:
                final_timestep = dataset["timeouts"][i]
            else:
                final_timestep = (episode_step == env._max_episode_steps-1)
            if final_timestep:   # why skip the transition?
                episode_step = 0
                trj_idx_list.append([start_idx, data_idx-1])
                start_idx = data_idx
                continue
            if done_bool:
                episode_step = 0
                trj_idx_list.append([start_idx, data_idx])
                start_idx = data_idx + 1
            episode_step += 1
            data_idx += 1
        trj_idx_list.append([start_idx, data_idx])
        trj_idx_list = np.array(trj_idx_list)
        trj_len_list = trj_idx_list[:, 1] - trj_idx_list[:, 0] + 1
        assert max(trj_len_list) > self.segment_length

        obs_1 = np.zeros([self.capacity, self.segment_length, self.obs_shape], dtype=np.float32)
        obs_2 = np.zeros([self.capacity, self.segment_length, self.obs_shape], dtype=np.float32)
        next_obs_1 = np.zeros([self.capacity, self.segment_length, self.obs_shape], dtype=np.float32)
        next_obs_2 = np.zeros([self.capacity, self.segment_length, self.obs_shape], dtype=np.float32)
        action_1 = np.zeros([self.capacity, self.segment_length, self.action_shape], dtype=np.float32)
        action_2 = np.zeros([self.capacity, self.segment_length, self.action_shape], dtype=np.float32)
        reward_1 = np.zeros([self.capacity, self.segment_length, ], dtype=np.float32)
        reward_2 = np.zeros([self.capacity, self.segment_length, ], dtype=np.float32)
        timestep_1 = np.zeros([self.capacity, self.segment_length, ], dtype=np.int32)
        timestep_2 = np.zeros([self.capacity, self.segment_length, ], dtype=np.int32)

        query_range = np.arange(len(human_labels) - self.capacity, len(human_labels))
        for query_count, i in enumerate(query_range):
            start_idx = human_indices_1[i]  # why not use query count?
            end_idx = start_idx + self.segment_length
            obs_1[query_count] = dataset["observations"][start_idx:end_idx]
            next_obs_1[query_count] = dataset["next_observations"][start_idx:end_idx]
            action_1[query_count] = dataset["actions"][start_idx:end_idx]
            reward_1[query_count] = dataset["rewards"][start_idx:end_idx]
            timestep_1[query_count] = np.arange(1, self.segment_length+1)

            start_idx = human_indices_2[i]  # why not use query count?
            end_idx = start_idx + self.segment_length
            obs_2[query_count] = dataset["observations"][start_idx:end_idx]
            next_obs_2[query_count] = dataset["next_observations"][start_idx:end_idx]
            action_2[query_count] = dataset["actions"][start_idx:end_idx]
            reward_2[query_count] = dataset["rewards"][start_idx:end_idx]
            timestep_2[query_count] = np.arange(1, self.segment_length+1)

        if self.label_key == "script":
            sum_reward_1 = reward_1.sum(axis=1)
            sum_reward_2 = reward_2.sum(axis=1)
            hard_label = 1.0 * (sum_reward_1 < sum_reward_2)
            soft_label = 0.5 * (sum_reward_1 == sum_reward_2)
            label = hard_label + soft_label
        elif self.label_key == "human":
            label = np.zeros([self.capacity, ], dtype=np.float32)
            label[np.array(human_labels) == 0] = 0
            label[np.array(human_labels) == 1] = 1
            label[np.array(human_labels) == -1] = 0.5

        data = {
            "obs_1": obs_1,
            "obs_2": obs_2,
            "next_obs_1": next_obs_1,
            "next_obs_2": next_obs_2,
            "action_1": action_1,
            "action_2": action_2,
            "timestep_1": timestep_1,
            "timestep_2": timestep_2,
            "label": label
        }
        self.data = data


if __name__ == "__main__":
    import d4rl
    import gym

    import wiserl.env
    env = gym.make("hopper-medium-expert-v2")
    dataset = PTComparisonOfflineDataset(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env="hopper-medium-expert-v2",
        capacity=100,
        segment_length=25,
        label_key="human"
    )
