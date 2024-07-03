import itertools
import os
from operator import itemgetter
from typing import Any, Dict, Optional, Sequence, Tuple, Type

import gym
import imageio
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import wiserl.module
from wiserl.algorithm.base import Algorithm
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.module.encoder_decoder import MLPEncDec
from wiserl.module.net.attention.gpt2 import GPT2
from wiserl.module.net.mlp import MLP
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target
from wiserl.module.net.attention.encoder_transformer import EncoderTransformer
from wiserl.module.net.attention.preference_decision_transformer import PreferenceDecisionTransformer


class BTEmbedding(Algorithm):
    def __init__(
        self,
        *args,
        type='Linear',
        embed_dim=64,
        **kwargs
    ):
        
        self.embed_dim = embed_dim
        super().__init__(*args, **kwargs)
            
    def setup_network(self, network_kwargs):
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        encoder_act = {
                    "identity": nn.Identity(),
                    "sigmoid": nn.Sigmoid(),
                }.get(network_kwargs["encoder"].pop("activation"))
        encoder = vars(wiserl.module)[network_kwargs["encoder"].pop("class")](
            input_dim=self.obs_dim+self.action_dim,
            output_dim=self.embed_dim,
            **network_kwargs["encoder"]
        )

        network_dct = {
            'encoder': nn.Sequential(encoder, encoder_act)
        }
        self.network = nn.ModuleDict(network_dct).to(self.device)
    
    def setup_optimizers(self, optim_kwargs):
        self.optim = {}
        encoder_kwargs = optim_kwargs['encoder']
        self.optim["encoder"] = vars(torch.optim)[encoder_kwargs.pop("class")](self.network.encoder.parameters(), **encoder_kwargs)

    def split_batch(self, batch):
        batch_size = batch['obs'].shape[0]

        indices = list(range(batch_size))
        np.random.shuffle(indices)
        batch1_indices = indices[:batch_size//2]
        batch2_indices = indices[batch_size//2:]

        batch1 = {key: batch[key][batch1_indices] for key in batch.keys()}
        batch2 = {key: batch[key][batch2_indices] for key in batch.keys()}
        return batch1, batch2

    def train_step(self, batches, step: int, total_steps: int):
        batch1, batch2 = self.split_batch(batches[0])
        for k in batch1.keys():
            print(k, batch1[k].shape)
        embed1 = self.network.encoder(torch.cat([batch1['obs'], batch1['action']], dim=-1))
        embed1 = torch.sum(embed1, dim=-2)        
        embed1 = F.normalize(embed1, dim=-1)
        
        embed2 = self.network.encoder(torch.cat([batch2['obs'], batch2['action']], dim=-1))
        embed2 = torch.sum(embed2, dim=-2)        
        embed2 = F.normalize(embed2, dim=-1)
        
        norm = torch.norm(embed1-embed2, p=2, dim=-1)
        loss = torch.mean(-torch.log(1 / (1 + torch.exp(norm ** 2))))

        self.optim["encoder"].zero_grad()
        loss.backward()
        self.optim["encoder"].step()
        return {
            'loss':loss.item(),
        }
        
    
    def select_action(self, batch, *args, **kwargs):
        raise NotImplementedError
    
    def get_embedding(self, batch):
        embed = self.network.encoder(torch.cat([batch['obs'], batch['action']], dim=-1)).squeeze(dim=0)
        return F.normalize(torch.sum(embed, dim=-2), dim=-1)