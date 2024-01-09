import io
import math
from typing import Dict, Optional, Union

import gym
import numpy as np
import torch

from src.utils import utils


class PairwiseComparisonOfflineDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space,
        action_space,
        path: Optional[str] = None,
        discount: float = 0.99,
        segment_length: Optional[int] = None,
        # segment_size: int = 20,
        # subsample_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        mode: str="sparse",
        label_key: str="label",
    ):
        super().__init__()
        # assert mode in {"dense", "sparse", "score"}
        self.mode = mode
        self.label_key = label_key
        self.discount = discount
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length

        assert path is not None, "Must provide dataset file."
        with open(path, "rb") as f:
            data = np.load(f)
            data = utils.nest_dict(data)
            assert self.label_key in data, "Key not found, valid keys:" + str(list(data.keys()))

            # If we are dealing with a new format dataset
            dataset_size = data["action"].shape[0]  # The number of segments in the dataset
            if capacity is not None:
                if mode == "sparse":
                    capacity = capacity * 2
                if capacity > dataset_size:
                    raise ValueError("Capacity exceeds the size of dataset!")
                data = utils.get_from_batch(data, 0, capacity)

        # preprocess the data
        data = utils.remove_float64(data)
        lim = 1 - 1e-8
        data["action"] = np.clip(data["action"], a_min=-lim, a_max=lim)
        # data["reward"] = reward_scale * data["reward"] + reward_shift

        # Save the data
        self.data = data
        self.data_size, self.data_segment_length = data["action"].shape[:2]

    def __len__(self):
        if self.mode.startswith("sparse"):
            return self.data_size // 2
        return self.data_size

    def sample_idx(self, idx):
        data_1_idxs = idx
        if self.mode == "sparse":
            data_2_idxs = data_1_idxs + len(self)
        elif self.mode == "dense":
            data_2_idxs = np.random.randint(0, len(self), size=data_1_idxs.shape[0])
        if self.segment_length is not None:
            start_1 = np.random.randint(0, self.data_segment_length - self.segment_length)
            end_1 = start_1 + self.segment_length
            start_2 = np.random.randint(0, self.data_segment_length - self.segment_length)
            end_2 = start_2 + self.segment_length
        else:
            start_1, end_1 = 0, self.data_segment_length
            start_2, end_2 = 0, self.data_segment_length


        batch = {
            "obs_1": self.data["obs"][data_1_idxs, start_1:end_1],
            "obs_2": self.data["obs"][data_2_idxs, start_2:end_2],
            "action_1": self.data["action"][data_1_idxs, start_1:end_1],
            "action_2": self.data["action"][data_2_idxs, start_2:end_2],
            "reward_1": self.data["reward"][data_1_idxs, start_1:end_1],
            "reward_2": self.data["reward"][data_2_idxs, start_2:end_2],
            "terminal_1": self.data["done"][data_1_idxs, start_1:end_1],
            "terminal_2": self.data["done"][data_2_idxs, start_2:end_2]
        }
        hard_label = 1.0 * (self.data[self.label_key][data_1_idxs] < self.data[self.label_key][data_2_idxs])
        soft_label = 0.5 * (self.data[self.label_key][data_1_idxs] == self.data[self.label_key][data_2_idxs])
        batch["label"] = (hard_label + soft_label).astype(np.float32)
        # The discount indicates the discounting factor used for generating the label
        batch["discount_1"] = self.discount * np.ones_like(batch["reward_1"], dtype=np.float32)
        batch["discount_2"] = self.discount * np.ones_like(batch["reward_2"], dtype=np.float32)
        return batch

    def __iter__(self):
        while True:
            idxs = np.random.randint(0, len(self), size=self.batch_size)
            yield self.sample_idx(idxs)
