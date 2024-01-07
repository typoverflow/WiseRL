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
        observation_space: gym.Space,
        action_space: gym.Space,
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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        size = len(self)
        chunk_size = size // num_workers
        my_inds = np.arange(chunk_size * worker_id, chunk_size * (worker_id + 1))
        idxs = np.random.permutation(my_inds)
        for i in range(math.ceil(len(idxs) / self.batch_size)):  # Need to use ceil to get all data points.
            if self.batch_size == 1:
                data_1_idxs = idxs[i]
            else:
                # Might be some overlap here but its probably OK.
                data_1_idxs = idxs[i * self.batch_size : min((i + 1) * self.batch_size, len(self))]

            if self.mode == "sparse":
                data_2_idxs = data_1_idxs + size
            elif self.mode == "dense":
                data_2_idxs = np.random.randint(0, size, size=data_1_idxs.shape)

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
            }
            hard_label = 1.0 * (self.data[self.label_key][data_1_idxs] < self.data[self.label_key][data_2_idxs])
            soft_label = 0.5 * (self.data[self.label_key][data_1_idxs] == self.data[self.label_key][data_2_idxs])
            batch["label"] = (hard_label + soft_label).astype(np.float32)
            # The discount indicates the discounting factor used for generating the label
            batch["discount_1"] = self.discount * np.ones_like(batch["reward_1"], dtype=np.float32)
            batch["discount_2"] = self.discount * np.ones_like(batch["reward_2"], dtype=np.float32)
        yield batch


# class ReplayAndFeedbackBuffer(torch.utils.data.IterableDataset):
#     """
#     Dataset class that combines a replay buffer and a feedback buffer
#     This is used for IH Learn and Offline Learning
#     """

#     def __init__(
#         self,
#         observation_space: gym.Space,
#         action_space: gym.Space,
#         replay_class: Union[str, torch.utils.data.IterableDataset] = EmptyDataset,
#         feedback_class: Union[str, torch.utils.data.IterableDataset] = PairwiseComparisonDataset,
#         replay_kwargs: Dict = {},
#         feedback_kwargs: Dict = {},
#         **kwargs,
#     ):
#         replay_kwargs = replay_kwargs.copy()
#         replay_kwargs.update(kwargs)
#         replay_class = vars(research.datasets)[replay_class] if isinstance(replay_class, str) else replay_class
#         self.replay_buffer = replay_class(observation_space, action_space, **replay_kwargs)
#         feedback_kwargs = feedback_kwargs.copy()
#         feedback_kwargs.update(kwargs)
#         feedback_class = vars(research.datasets)[feedback_class] if isinstance(feedback_class, str) else feedback_class
#         self.feedback_dataset = feedback_class(observation_space, action_space, **feedback_kwargs)

#     def __iter__(self):
#         # Yield one batch of each in a tuple per step.
#         replay_iter = iter(self.replay_buffer)
#         feedback_iter = iter(self.feedback_dataset)
#         current_feedback_size = len(self.feedback_dataset)

#         while True:
#             replay_data = next(replay_iter)  # Replay iter should be infinite
#             if len(self.feedback_dataset) > current_feedback_size:
#                 # Check to see if the size of the feedback dataset has increased
#                 # If so, recreate the iterator to fetch new data.
#                 current_feedback_size = len(self.feedback_dataset)
#                 del feedback_iter
#                 feedback_iter = iter(self.feedback_dataset)

#             feedback_data = next(feedback_iter, None)
#             if feedback_data is None:
#                 # Check once to re-add. If this is the first epoch, we may get None back.
#                 feedback_iter = iter(self.feedback_dataset)
#                 feedback_data = next(feedback_iter, None)

#             yield replay_data, feedback_data
