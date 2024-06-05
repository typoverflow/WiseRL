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


class OPPO(Algorithm):
    def __init__(
        self,
        *args,
        z_dim: int = 64,
        seq_len: int = 1024,
        warmup_steps: int = 100,
        pref_loss_ratio: float = 0.1,
        znorm_loss_ratio: float = 0.1,
        **kwargs
    ):
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.warmup_steps = warmup_steps
        super().__init__(*args, **kwargs)
        
        self.him_loss = nn.MSELoss()
        self.pref_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.znorm_loss = nn.MSELoss()

        self.pref_loss_ratio = pref_loss_ratio
        self.znorm_loss_ratio = znorm_loss_ratio


    def setup_network(self, network_kwargs) -> None:
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        enc_kwargs = network_kwargs["encoder"]
        policy_kwargs = network_kwargs["policy"]

        encoder = EncoderTransformer(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            embed_dim=enc_kwargs["embed_dim"],
            z_dim=self.z_dim,
            num_layers=enc_kwargs["num_layers"],
            num_heads=enc_kwargs["num_heads"],
            attention_dropout=enc_kwargs["dropout"],
            residual_dropout=enc_kwargs["dropout"],
            embed_dropout=enc_kwargs["dropout"],
        )
        policy = PreferenceDecisionTransformer(
            state_dim=self.obs_dim,
            action_dim=self.action_dim,
            embed_dim=policy_kwargs["embed_dim"],
            z_dim=self.z_dim,
            num_layers=policy_kwargs["num_layers"],
            num_heads=policy_kwargs["num_heads"],
            attention_dropout=policy_kwargs["dropout"],
            residual_dropout=policy_kwargs["dropout"],
            embed_dropout=policy_kwargs["dropout"],
            seq_len=self.seq_len
        )

        self.z_star = (torch.randn(self.z_dim) * 2).to(self.device)
        self.z_star.requires_grad = True

        self.network = nn.ModuleDict({
            'encoder': encoder.to(self.device),
            'policy': policy.to(self.device)
        })


    def setup_optimizers(self, optim_kwargs):
        policy_optimizer = torch.optim.AdamW(
            self.network.policy.parameters(),
            lr=optim_kwargs['policy']['lr'],
            weight_decay=optim_kwargs['policy']['weight_decay'],
        )
        encoder_optimizer = torch.optim.AdamW(
            self.network.encoder.parameters(),
            lr=optim_kwargs['encoder']['lr'],
            weight_decay=optim_kwargs['encoder']['weight_decay'],
        )
        zstar_optimizer = torch.optim.AdamW(
            [self.z_star],
            lr=optim_kwargs['zstar']['lr'],
            weight_decay=optim_kwargs['zstar']['weight_decay'],
        )
        self.optim={
            'policy':policy_optimizer,
            'encoder':encoder_optimizer,
            'zstar':zstar_optimizer
        }

    def setup_schedulers(self, scheduler_kwargs):
        self.schedulers = {
            'policy':torch.optim.lr_scheduler.LambdaLR(
                self.optim['policy'],
                lambda steps: min((steps+1)/self.warmup_steps, 1)
            )
        }

    def select_action(self, batch, deterministic: bool=True):
        return self.network['policy'].get_action(batch['obs'], batch['action'], self.z_star, batch['timestep'])

    def select_reward(self, batch, deterministic=False):
        pass
    
    # seperate one batch into two batches to simulate batches with preference
    def split_batch(self, batches):
        batch = batches[0]
        batch_size = batch['obs'].shape[0] // 2
        #print(batch['obs'][0])
        # TODO OPPO's rtg
        sample_indices = list(range(2*batch_size))
        random.shuffle(sample_indices)

        batch1 = {}
        batch2 = {}
        for k in batch.keys():
            # TODO random sample seq_len slices
            batch1[k] = batch[k][sample_indices[:batch_size],:self.seq_len].to(self.device)
            batch2[k] = batch[k][sample_indices[batch_size:],:self.seq_len].to(self.device)
        return [batch1, batch2]

    def train_step(self, batches, step: int, total_steps: int):
        print('train', str(step)+':'+str(total_steps))
        batches = self.split_batch(batches)
        metrics = {}

        batch_size = batches[0]['obs'].shape[0]

        action_target_1 = torch.clone(batches[0]['action'])
        action_target_2 = torch.clone(batches[1]['action'])

        margin = 0
        lb = (batches[0]['return_to_go'][:,0,0] - batches[1]['return_to_go'][:,0,0]) > margin
        rb = (batches[1]['return_to_go'][:,0,0] - batches[0]['return_to_go'][:,0,0]) > margin

        z1 = self.network.encoder(
            states=batches[0]['obs'],
            actions=batches[0]['action'],
            timesteps=batches[0]['timestep'],
            key_padding_mask=1-batches[0]['mask']
        )
        z2 = self.network.encoder(
            states=batches[1]['obs'],
            actions=batches[1]['action'],
            timesteps=batches[1]['timestep'],
            key_padding_mask=1-batches[1]['mask']
        )
        znorm_loss = (self.znorm_loss(torch.norm(z1, dim=1), torch.ones(batch_size).to(self.device))
                       + self.znorm_loss(torch.norm(z2, dim=1), torch.ones(batch_size).to(self.device)))
        positive = torch.cat((z1[lb], z2[rb]), 0)
        negative = torch.cat((z2[lb], z1[rb]), 0)
        anchor = self.z_star.expand(positive.shape[0], -1).detach()
        pref_loss = self.pref_loss(anchor, positive, negative)

        z1 = z1.expand(self.seq_len, -1, -1).permute(1, 0, 2)
        z2 = z2.expand(self.seq_len, -1, -1).permute(1, 0, 2)
        _, action_preds_1, _ = self.network.policy(
            states=batches[0]['obs'],
            actions=batches[0]['action'],
            zs = z1,
            timesteps=batches[0]['timestep'],
            key_padding_mask=1-batches[0]['mask']
        )
        _, action_preds_2, _ = self.network.policy(
            states=batches[1]['obs'],
            actions=batches[1]['action'],
            zs = z2,
            timesteps=batches[1]['timestep'],
            key_padding_mask=1-batches[1]['mask']
        )

        action_preds_1 = action_preds_1.reshape(-1, self.action_dim)[batches[0]['mask'].reshape(-1) > 0]
        action_target_1 = action_target_1.reshape(-1, self.action_dim)[batches[0]['mask'].reshape(-1) > 0]
        action_preds_2 = action_preds_2.reshape(-1, self.action_dim)[batches[1]['mask'].reshape(-1) > 0]
        action_target_2 = action_target_2.reshape(-1, self.action_dim)[batches[1]['mask'].reshape(-1) > 0]

        him_loss = self.him_loss(action_preds_1, action_target_1) + self.him_loss(action_preds_2, action_target_2)
        self.optim['encoder'].zero_grad()
        self.optim['policy'].zero_grad()
        (
            him_loss
            + self.pref_loss_ratio * pref_loss
            + self.znorm_loss_ratio * znorm_loss
        ).backward()
        
        nn.utils.clip_grad_norm_(self.network.encoder.parameters(), 0.25)
        nn.utils.clip_grad_norm_(self.network.policy.parameters(), 0.25)
        self.optim['encoder'].step()
        self.optim['policy'].step()

        zstar_loss = self.update_zstar(batches, lb, rb)
        if step > 0 and step % 1000 == 0:
            self.schedulers['policy'].step()

        metrics = {
            'znorm_loss':znorm_loss.detach().cpu().item(),
            'pref_loss':pref_loss.detach().cpu().item(),
            'him_loss':him_loss.detach().cpu().item(),
            'zstar_loss':zstar_loss
        }
        for k in metrics.keys():
            print(k, metrics[k], end=' ')
        print('')
        return metrics

    def update_zstar(self, batches, lb, rb):
        z1 = self.network.encoder(
            states=batches[0]['obs'],
            actions=batches[0]['action'],
            timesteps=batches[0]['timestep'],
            key_padding_mask=1-batches[0]['mask']
        )
        z2 = self.network.encoder(
            states=batches[1]['obs'],
            actions=batches[1]['action'],
            timesteps=batches[1]['timestep'],
            key_padding_mask=1-batches[1]['mask']
        )
        positive = torch.cat((z1[lb], z2[rb]), 0)
        negative = torch.cat((z2[lb], z1[rb]), 0)
        anchor = self.z_star.expand(positive.shape[0], -1)
        pref_loss = self.pref_loss(anchor, positive, negative)

        self.optim['zstar'].zero_grad()
        pref_loss.backward()
        self.optim['zstar'].step()

        return pref_loss.detach().cpu().item()

