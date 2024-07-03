import itertools
import os
from operator import itemgetter
from typing import Any, Dict, Optional, Sequence, Tuple, Type

import gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wiserl.module
from wiserl.algorithm.base import Algorithm
from wiserl.algorithm.msrml.embed_him import HIMEmbedding
from wiserl.algorithm.msrml.embed_bt import BTEmbedding
from wiserl.algorithm.msrml.embed_sale import SALEEmbedding
from wiserl.algorithm.msrml.embed_test import TestEmbedding
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.module.encoder_decoder import MLPEncDec
from wiserl.module.net.attention.gpt2 import GPT2
from wiserl.module.net.mlp import MLP
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target
from wiserl.module.net.attention.encoder_transformer import EncoderTransformer
from wiserl.module.net.attention.preference_decision_transformer import PreferenceDecisionTransformer


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
        print(args, kwargs)
        self.args = args
        self.embed_args = kwargs.pop('embed_args')
        self.embed_type = kwargs.pop('embed_type')
        super().__init__(*args, **kwargs)
        self.setup_embeddings()

    def setup_embeddings(self):
        print(self.embed_type)
        embed_dct = {
            'HIM': HIMEmbedding,
            'BT': BTEmbedding,
            'SALE': SALEEmbedding,
            'Test': TestEmbedding
        }
        self.embed_algo = embed_dct[self.embed_type](
            self.observation_space,
            self.action_space,
            self.embed_args['network'],
            self.embed_args['optim'],
            self.embed_args['schedulers'],
            self.embed_args['processor'],
            self.embed_args['checkpoint'],
            **self.embed_args['embed_args'],
            embed_dim=self.preference_dimension,
            device=self.device,
        )

    def setup_network(self, network_kwargs) -> None:
        self.network_dct = {}
        self.reward_points = []
        for i in range(self.num_preference):
            rm = torch.randn(self.preference_dimension).to(self.device)
            rm = F.normalize(rm, dim=-1)
            self.reward_points.append(rm)
        self.network = nn.ModuleDict(self.network_dct)        

    def setup_optimizers(self, optim_kwargs):
        self.optim = {}

    def select_action(self, batch, deterministic: bool=True):
        pass

    def select_reward(self, batch, deterministic=False):
        pass

    def split_batch(self, batch):
        batch_size = batch['obs'].shape[0]

        indices = list(range(batch_size))
        np.random.shuffle(indices)
        batch1_indices = indices[:batch_size//2]
        batch2_indices = indices[batch_size//2:]

        batch1 = {key: batch[key][batch1_indices] for key in batch.keys()}
        batch2 = {key: batch[key][batch2_indices] for key in batch.keys()}
        return batch1, batch2

    def select_embedding(self, batch):
        # TODO: replace with true embedding
        #batch1, batch2 = self.split_batch(batch)
        
        batch1 = {
            'obs': batch['obs_1'],
            'action': batch['action_1']
        }
        batch2 = {
            'obs': batch['obs_2'],
            'action': batch['action_2']
        }
        with torch.no_grad():
            embed1 = self.embed_algo.get_embedding(batch1)
            embed2 = self.embed_algo.get_embedding(batch2)
        embed = torch.stack([embed1, embed2], dim=1)

        new_embed = embed.clone()
        for i in range(batch['obs_1'].shape[0]):
            label = batch['label'][i]
            if label == 1:
                new_embed[i, 0, :] = embed[i, 1, :]
                new_embed[i, 1, :] = embed[i, 0, :]
        
        return new_embed

    def embedding_step(self, batches, step: int, total_steps: int):
        self.embed_algo.train()
        embed_metric = self.embed_algo.train_step(batches, step, total_steps)

    def cluster_step(self, batches):
        self.embed_algo.eval()
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
    
    def cluster_eval(self, batches):
        embeddings = self.select_embedding(batches[0])
        num_data = embeddings.shape[0]
        distances = np.zeros((num_data, self.num_preference))
        for i in range(num_data):
            for j in range(self.num_preference):
                distances[i, j] = torch.dot(embeddings[i][0]-embeddings[i][1], self.reward_points[j])

        count = np.count_nonzero(distances < 0)
        reward_accuracy = count / distances.shape[0]
        print(reward_accuracy)

        return {
            'reward:accuracy': reward_accuracy
        }

        
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
