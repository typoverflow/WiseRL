import itertools
import os
from operator import itemgetter
from typing import Any, Dict, Optional, Sequence, Tuple, Type

import gym
import imageio
import numpy as np
import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.base import Algorithm
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.module.encoder_decoder import MLPEncDec
from wiserl.module.net.attention.gpt2 import GPT2
from wiserl.module.net.mlp import MLP
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target


class MultiSpatialRewardModelLearning(Algorithm):
    def __init__(
        self,
        *args,
        num_preference: int = 2,
        preference_dimension: int = 64,
        **kwargs
    ):
        self.num_preference = num_preference
        self.preference_dimension = preference_dimension
        super().__init__(*args, **kwargs)

    def setup_network(self, network_kwargs) -> None:
        super().setup_network(network_kwargs)
        self.reward_points = []
        for i in range(self.num_preference):
            rm = torch.randn(self.preference_dimension)
            rm = rm / torch.norm(rm)
            self.reward_points.append(rm)

    def setup_optimizers(self, optim_kwargs):
        pass

    def select_action(self, batch, deterministic: bool=True):
        pass

    def select_reward(self, batch, deterministic=False):
        pass

    def select_embedding(self, batch):
        # TODO: replace with true embedding
        embeddings =  torch.randn([batch['obs'].shape[0], 2, self.preference_dimension])
        norms = torch.norm(embeddings, dim=2, keepdim=True)
        unit_embeddings = embeddings / norms
        return unit_embeddings

    def pretrain_step(self, batches, step: int, total_steps: int):
        embeddings = self.select_embedding(batches[0])
        num_data = embeddings.shape[0]

        distances = np.zeros((num_data, self.num_preference))
        for i in range(num_data):
            for j in range(self.num_preference):
                distances[i, j] = torch.dot(embeddings[i][0]-embeddings[i][1], self.reward_points[j])
        assignments = np.argmax(distances, axis=1)

        for j in range(self.num_preference):
            clustered_embeddings = embeddings[assignments == j]
            if clustered_embeddings.shape[0] > 0:
                self.reward_points[j] = torch.mean(clustered_embeddings[:,0]-clustered_embeddings[:,1], dim=0)
                self.reward_points[j] = self.reward_points[j]/torch.norm(self.reward_points[j])
        
    def train_step(self, batches, step: int, total_steps: int):
        pass

    def update_agent(self, obs, action, next_obs, reward, terminal):
        pass

    def update_reward(self, obs_1, obs_2, action_1, action_2, label, extra_obs, extra_action):
        pass

    def update_vae(self, obs: torch.Tensor, action: torch.Tensor, timestep: torch.Tensor, mask: torch.Tensor):
        pass

    def load_pretrain(self, path):
        for attr in ["future_encoder", "future_proj", "decoder", "prior", "reward"]:
            state_dict = torch.load(os.path.join(path, attr+".pt"), map_location=self.device)
            self.network.__getattr__(attr).load_state_dict(state_dict)

    def save_pretrain(self, path):
        os.makedirs(path, exist_ok=True)
        for attr in ["future_encoder", "future_proj", "decoder", "prior", "reward"]:
            state_dict = self.network.__getattr__(attr).state_dict()
            torch.save(state_dict, os.path.join(path, attr+".pt"))
