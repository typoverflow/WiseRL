import io
import math
from typing import Dict, Optional, Union

import gym
import numpy as np
import torch

from wiserl.utils import utils

DATASET_PATH = {
    "mw_bin-picking-v2": "datasets/mw/pref/mw_bin-picking-v2_ep2500_n0.3.npz",
    "mw_button-press-v2": "datasets/mw/pref/mw_button-press-v2_ep2500_n0.3.npz",
    "mw_door-open-v2": "datasets/mw/pref/mw_door-open-v2_ep2500_n0.3.npz",
    "mw_drawer-open-v2": "datasets/mw/pref/mw_drawer-open-v2_ep2500_n0.3.npz",
    "mw_plate-slide-v2": "datasets/mw/pref/mw_plate-slide-v2_ep2500_n0.3.npz",
    "mw_sweep-info-v2": "datasets/mw/pref/mw_sweep-info-v2_ep2500_n0.3.npz",
    "mw_bin-picking-image-v2": "datasets/mw/pref_image/mw_bin-picking-v2_ep2500_n0.3_img64.npz",
    "mw_button-press-image-v2": "datasets/mw/pref_image/mw_button-press-v2_ep2500_n0.3_img64.npz",
    "mw_door-open-image-v2": "datasets/mw/pref_image/mw_door-open-v2_ep2500_n0.3_img64.npz",
    "mw_drawer-open-image-v2": "datasets/mw/pref_image/mw_drawer-open-v2_ep2500_n0.3_img64.npz",
    "mw_plate-slide-image-v2": "datasets/mw/pref_image/mw_plate-slide-v2_ep2500_n0.3_img64.npz",
    "mw_sweep-info-image-v2": "datasets/mw/pref_image/mw_sweep-info-v2_ep2500_n0.3_img64.npz",
}


class MetaworldComparisonOfflineDataset(torch.utils.data.IterableDataset):
    """
    Metaworld dataset, borrowed from CPL.
    label_keys:
        rl_dir: \sum_{t} Q(s_t, a_t) - V(s_t)
        rl_dis_dir: \sum_{t} \gamma^t (Q(s_t, a_t) - V(s_t))
        rl_sum: \sum_{t} [r_t] + V(s_T) - V(s_0)
        rl_dis_sum: \sum_{t} \sum_{t} [\gamma^t r_t] + \gamma^{T-1} V(s_T) - V(s_0)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        env: Optional[str],
        discount: float = 0.99,
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        mode: str="sparse",
        label_key: str="rl_sum",
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
    ):
        super().__init__()
        assert env in DATASET_PATH.keys(), "Env {env} not registered for PT dataset."
        assert label_key in {"rl_dir", "rl_dis_dir", "rl_sum", "rl_dis_sum"}, f"MetaworldComparisonOfflineDataset does not support label_key: {label_key}"
        assert mode in {"sparse", "dense"}, f"MetaworldComparisonOfflineDataset does not support mode: {mode}"

        self.env_name = env
        self.mode = mode
        self.label_key = label_key
        self.discount = discount
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length

        self.reward_scale = reward_scale
        self.reward_shift = reward_shift

        path = DATASET_PATH[self.env_name]
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
        data["reward"] = reward_scale * data["reward"] + reward_shift

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

        data_1_idxs, data_2_idxs = np.squeeze(data_1_idxs), np.squeeze(data_2_idxs)
        batch = {
            "obs_1": self.data["obs"][data_1_idxs, start_1:end_1],
            "obs_2": self.data["obs"][data_2_idxs, start_2:end_2],
            "action_1": self.data["action"][data_1_idxs, start_1:end_1],
            "action_2": self.data["action"][data_2_idxs, start_2:end_2],
            "reward_1": self.data["reward"][data_1_idxs, start_1:end_1, None],
            "reward_2": self.data["reward"][data_2_idxs, start_2:end_2, None],
        }
        hard_label = 1.0 * (self.data[self.label_key][data_1_idxs] < self.data[self.label_key][data_2_idxs])
        soft_label = 0.5 * (self.data[self.label_key][data_1_idxs] == self.data[self.label_key][data_2_idxs])
        batch["label"] = (hard_label + soft_label).astype(np.float32)

        batch["terminal_1"] = np.zeros_like(batch["reward_1"], dtype=np.float32)
        batch["terminal_2"] = np.zeros_like(batch["reward_2"], dtype=np.float32)
        return batch

    def __iter__(self):
        while True:
            idxs = np.random.randint(0, len(self), size=self.batch_size)
            yield self.sample_idx(idxs)
