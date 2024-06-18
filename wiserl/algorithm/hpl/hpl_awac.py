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
from wiserl.algorithm.hpl.hpl import Decoder, HindsightPreferenceLearning
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.module.encoder_decoder import MLPEncDec
from wiserl.module.net.attention.gpt2 import GPT2
from wiserl.module.net.mlp import MLP
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target


class HindsightPreferenceLearningAWAC(HindsightPreferenceLearning):
    def __init__(
        self,
        *args,
        beta: float = 0.3333,
        max_exp_clip: float = 100.0,
        discount: float = 0.99,
        tau: float = 0.005,
        seq_len: int = 100,
        future_len: int = 50,
        z_dim: int = 64,
        prior_sample: int = 5,
        kl_loss_coef: float = 1.0,
        kl_balance_coef: float = 0.8,
        reg_coef: float = 0.01,
        vae_steps: int = 100000,
        rm_label: bool = True,
        reward_steps: int = 100000,
        stoc_encoding: bool = True,
        discrete: bool = True,
        discrete_group: int = 8,
        **kwargs
    ):
        super().__init__(
            *args,
            expectile=0.0,
            beta=beta,
            max_exp_clip=max_exp_clip,
            discount=discount,
            tau=tau,
            seq_len=seq_len,
            future_len=future_len,
            z_dim=z_dim,
            prior_sample=prior_sample,
            kl_loss_coef=kl_loss_coef,
            kl_balance_coef=kl_balance_coef,
            reg_coef=reg_coef,
            vae_steps=vae_steps,
            rm_label=rm_label,
            reward_steps=reward_steps,
            stoc_encoding=stoc_encoding,
            discrete=discrete,
            discrete_group=discrete_group,
            **kwargs
        )

    def update_agent(self, obs, action, next_obs, reward, terminal):
        # compute the loss for actor
        with torch.no_grad():
            baseline_actions = self.network.actor.sample(obs)[0]
            v = self.network.critic(obs, baseline_actions).mean(0)
            q = self.network.critic(obs, action).mean(0)
            advantage = q-v
        exp_advantage = (advantage / self.beta).exp().clip(max=self.max_exp_clip)
        if isinstance(self.network.actor, DeterministicActor):
            policy_out = torch.sum((self.network.actor.sample(obs)[0] - action)**2, dim=-1, keepdim=True)
        elif isinstance(self.network.actor, GaussianActor):
            policy_out = - self.network.actor.evaluate(obs, action)[0]
        actor_loss = (exp_advantage * policy_out).mean()
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()

        # compute the loss for q
        with torch.no_grad():
            self.target_network.eval()
            next_actions = self.network.actor.sample(next_obs)[0]
            target_q = self.target_network.critic(next_obs, next_actions).min(0)[0]
            target_q = reward + self.discount * (1-terminal.float()) * target_q
        q_pred = self.network.critic(obs, action)
        q_loss = (q_pred - target_q.unsqueeze(0)).pow(2).sum(0).mean()
        self.optim["critic"].zero_grad()
        q_loss.backward()
        self.optim["critic"].step()

        sync_target(self.network.critic, self.target_network.critic, tau=self.tau)

        metrics = {
            "loss/q_loss": q_loss.item(),
            "loss/actor_loss": actor_loss.item(),
            "misc/q_pred": q_pred.mean().item(),
            "misc/advantage": advantage.mean().item(),
            "misc/reward_prior": reward.mean().item()
        }
        return metrics
