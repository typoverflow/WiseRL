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


def AvgL1Norm(x, eps=1e-8):
	return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)

class TestEmbedding(Algorithm):
    def __init__(
        self,
        *args,
        bt_ratio=0.5,
        sale_ratio=1,
        embed_dim=64,
        **kwargs
    ):
        
        self.embed_dim = embed_dim
        self.bt_ratio = bt_ratio
        self.sale_ratio = sale_ratio
        super().__init__(*args, **kwargs)
            
    def setup_network(self, network_kwargs):
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        act_dct = {
                    "identity": nn.Identity(),
                    "sigmoid": nn.Sigmoid(),
                    "ELU": nn.ELU(),
                }
        state_encoder_act = act_dct.get(network_kwargs["state_encoder"].pop("activation"))
        joint_encoder_act = act_dct.get(network_kwargs["joint_encoder"].pop("activation"))

        state_encoder = vars(wiserl.module)[network_kwargs["state_encoder"].pop("class")](
            input_dim=self.obs_dim,
            output_dim=self.embed_dim,
            **network_kwargs["state_encoder"]
        )
        joint_encoder = vars(wiserl.module)[network_kwargs["joint_encoder"].pop("class")](
            input_dim=self.embed_dim+self.action_dim,
            output_dim=self.embed_dim,
            **network_kwargs["joint_encoder"]
        )


        network_dct = {
            'state_encoder': nn.Sequential(state_encoder, state_encoder_act),
            'joint_encoder': nn.Sequential(joint_encoder, joint_encoder_act),
        }
        self.network = nn.ModuleDict(network_dct).to(self.device)
    
    def setup_optimizers(self, optim_kwargs):
        self.optim = {}
        state_encoder_kwargs = optim_kwargs['state_encoder']
        joint_encoder_kwargs = optim_kwargs['joint_encoder']
        self.optim["state_encoder"] = vars(torch.optim)[state_encoder_kwargs.pop("class")](self.network.state_encoder.parameters(), **state_encoder_kwargs)
        self.optim["joint_encoder"] = vars(torch.optim)[joint_encoder_kwargs.pop("class")](self.network.joint_encoder.parameters(), **joint_encoder_kwargs)

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
        batch = batches[0]
        #for k in batch.keys():
        #    print(k, batch[k].shape)

        state1_embed = AvgL1Norm(self.network.state_encoder(batch['obs_1']))
        joint1_embed = self.network.joint_encoder(torch.cat([state1_embed[0], batch['action_1']], dim=-1))

        state2_embed = AvgL1Norm(self.network.state_encoder(batch['obs_2']))
        joint2_embed = self.network.joint_encoder(torch.cat([state2_embed[0], batch['action_2']], dim=-1))

        
        nexts1_embed = AvgL1Norm(self.network.state_encoder(batch['obs_1'][:,1:,:])).detach()
        nexts2_embed = AvgL1Norm(self.network.state_encoder(batch['obs_2'][:,1:,:])).detach()
        sale_loss = torch.mean(torch.mean((joint1_embed[:,:,:-1,:]-nexts1_embed)**2) + 
                               torch.mean((joint2_embed[:,:,:-1,:]-nexts2_embed)**2))

        traj1_embed = F.normalize(torch.sum(joint1_embed, dim=-2) , dim=-1)
        traj2_embed = F.normalize(torch.sum(joint2_embed, dim=-2) , dim=-1)
        norm = torch.norm(traj1_embed-traj2_embed, p=2, dim=-1)
        bt_loss = torch.mean(-torch.log(1 / (1 + torch.exp(norm**2))))

        self.optim["state_encoder"].zero_grad()
        self.optim["joint_encoder"].zero_grad()
        (self.sale_ratio * sale_loss+self.bt_ratio * bt_loss).backward()
        self.optim["state_encoder"].step()
        self.optim["joint_encoder"].step()
        #print(sale_loss.item(), bt_loss.item())
        return {
            'sale_loss':sale_loss.item(),
            'bt_loss':bt_loss.item(),
        }
        
    def select_action(self, batch, *args, **kwargs):
        raise NotImplementedError
    
    def get_embedding(self, batch):
        state_embed = AvgL1Norm(self.network.state_encoder(batch['obs']))
        joint_embed = self.network.joint_encoder(torch.cat([state_embed.squeeze(dim=0), batch['action']], dim=-1)).squeeze(dim=0)
        joint_embed = torch.sum(joint_embed, dim=-2)
        return F.normalize(joint_embed, dim=-1)