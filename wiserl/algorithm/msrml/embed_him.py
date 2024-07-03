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


class HIMEmbedding(Algorithm):
    def __init__(
        self,
        *args,
        seq_len=100,
        znorm_loss_ratio=0.1,
        warmup_steps=10000,
        embed_dim=64,
        **kwargs
    ):
        
        self.seq_len = seq_len
        self.znorm_loss_ratio = znorm_loss_ratio
        self.warmup_steps = warmup_steps
        self.embed_dim = embed_dim
        super().__init__(*args, **kwargs)
        self.him_loss = nn.MSELoss(reduction='none')
        self.znorm_loss = nn.MSELoss()
            
    def setup_network(self, network_kwargs):
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        enc_kwargs = network_kwargs["encoder"]
        reconstructor_kwargs = network_kwargs["reconstructor"]
        encoder = EncoderTransformer(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            embed_dim=enc_kwargs["embed_dim"],
            z_dim=self.embed_dim,
            num_layers=enc_kwargs["num_layers"],
            num_heads=enc_kwargs["num_heads"],
            attention_dropout=enc_kwargs["dropout"],
            residual_dropout=enc_kwargs["dropout"],
            embed_dropout=enc_kwargs["dropout"],
         )
        reconstructor = PreferenceDecisionTransformer(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            embed_dim=reconstructor_kwargs["embed_dim"],
            z_dim=self.embed_dim,
            num_layers=reconstructor_kwargs["num_layers"],
            num_heads=reconstructor_kwargs["num_heads"],
            attention_dropout=reconstructor_kwargs["dropout"],
            residual_dropout=reconstructor_kwargs["dropout"],
            embed_dropout=reconstructor_kwargs["dropout"],
            seq_len=self.seq_len
        )

        network_dct = {
            'encoder': encoder.to(self.device),
            'reconstructor': reconstructor.to(self.device)
        }
        self.network = nn.ModuleDict(network_dct).to(self.device)
    
    def setup_optimizers(self, optim_kwargs):
        encoder_optimizer = torch.optim.AdamW(
                self.network.encoder.parameters(),
                lr=optim_kwargs['encoder']['lr'],
                weight_decay=optim_kwargs['encoder']['weight_decay'],
            )
        reconstructor_optimizer = torch.optim.AdamW(
                self.network.reconstructor.parameters(),
                lr=optim_kwargs['reconstructor']['lr'],
                weight_decay=optim_kwargs['reconstructor']['weight_decay'],
            )
        self.optim = {}
        self.optim['encoder']=encoder_optimizer
        self.optim['reconstructor']=reconstructor_optimizer
    
    def setup_schedulers(self, scheduler_kwargs):
        self.schedulers = {
            'policy':torch.optim.lr_scheduler.LambdaLR(
                self.optim['reconstructor'],
                lambda steps: min((steps+1)/self.warmup_steps, 1)
            )
        }
    
    def train_step(self, batches, step: int, total_steps: int):
        for k in batches[0].keys():
            print(k, batches[0][k].shape)
        batch = batches[0]
        batch_size = batch['obs'].shape[0]
        action_target = torch.clone(batch['action'])
        #batch['mask'] = batch['mask'][:,:,0]

        z = self.network.encoder(
            states=batch['obs'],
            actions=batch['action'],
            timesteps=batch['timestep'],
            key_padding_mask=1-batch['mask']
        )
        znorm_loss = self.znorm_loss(torch.norm(z, dim=1), torch.ones(batch_size).to(self.device))

        _, action_preds, _ = self.network.reconstructor(
            states=batch['obs'],
            actions=batch['action'],
            zs = z.expand(self.seq_len, -1, -1).permute(1, 0, 2),
            timesteps=batch['timestep'],
            key_padding_mask=1-batch['mask']
        )
        him_mask = batch['mask'].unsqueeze(-1).repeat([1,1,3])
        him_loss = (self.him_loss(action_preds, action_target) * him_mask).sum() / him_mask.sum()

        self.optim['encoder'].zero_grad()
        self.optim['reconstructor'].zero_grad()
        (
            him_loss
            + self.znorm_loss_ratio * znorm_loss
        ).backward()
        
        nn.utils.clip_grad_norm_(self.network.encoder.parameters(), 0.25)
        nn.utils.clip_grad_norm_(self.network.reconstructor.parameters(), 0.25)
        self.optim['encoder'].step()
        self.optim['reconstructor'].step()
        print(him_loss.item(), znorm_loss.item())
        if step > 0 and step % 1000 == 0:
            self.schedulers['policy'].step()
        return {
            'him_loss':him_loss.item(),
            'znorm_loss':znorm_loss.item(),
        }
    
    def select_action(self, batch, *args, **kwargs):
        raise NotImplementedError
    
    def get_embedding(self, batch):
        embed = self.network.encoder(
            states=batch['obs'],
            actions=batch['action'],
            timesteps=batch['timestep'],
            key_padding_mask=1-batch['mask']
        )
        
        return F.normalize(embed, dim=-1)
