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
from wiserl.module.net.attention.positional_encoding import get_pos_encoding
from wiserl.module.net.attention.preference_transformer import PreferenceTransformer
from wiserl.module.net.attention.transformer import TransformerDecoder
from wiserl.module.net.mlp import MLP
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target
from wiserl.utils.optim import LinearWarmupCosineAnnealingLR


class RewardDecoder(TransformerDecoder):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        seq_len: int,
        episode_len: int,
        embed_dropout: Optional[float] = 0.1,
        attention_dropout: Optional[float] = 0.1,
        residual_dropout: Optional[float] = 0.1,
        pos_encoding = "sinusoidal",
        reward_act: str = "identity",
    ) -> None:
        super().__init__(
            input_dim=obs_dim+action_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            causal=False,
            out_ln=True,
            pre_norm=True,
            embed_dropout=embed_dropout,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            pos_encoding=pos_encoding
        )
        self.pos_encoding = get_pos_encoding(pos_encoding, embed_dim, episode_len+seq_len)
        self.z_embed = nn.Linear(z_dim, embed_dim)
        self.input_embed = nn.Linear(obs_dim+action_dim, embed_dim)
        self.z_ln = nn.LayerNorm(embed_dim)
        self.input_ln = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, 1)
        self.reward_act = nn.Identity() if reward_act == "identity" else nn.Sigmoid()

    def forward(
        self,
        z: torch.Tensor,
        input: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        B, L, E = z.shape
        z_embedding = self.z_ln(
            self.pos_encoding(self.z_embed(z), timestep)
        )
        input_embedding = self.input_ln(
            self.pos_encoding(self.input_embed(input), timestep)
        )
        tgt_mask = ~torch.eye(L, L).bool().to(z.device)
        src_mask = ~torch.tril(torch.ones([L, L])).bool().to(z.device)
        out = super().forward(
            tgt=z_embedding,
            enc_src=input_embedding,
            timesteps=None,
            tgt_attention_mask=tgt_mask,
            src_attention_mask=src_mask,
            src_key_padding_mask=key_padding_mask,
            do_embedding=False
        )
        out = self.reward_act(self.out_proj(out))
        return out


class HindsightPreferenceLearningPOMDP(HindsightPreferenceLearning):
    def __init__(
        self, *args, label_seq_len: int=25, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.label_seq_len = label_seq_len

    def setup_network(self, network_kwargs) -> None:
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        enc_kwargs = network_kwargs["encoder"]
        future_encoder = GPT2(
            input_dim=self.obs_dim+self.action_dim,
            embed_dim=enc_kwargs["embed_dim"],
            num_layers=enc_kwargs["num_layers"],
            num_heads=enc_kwargs["num_heads"],
            attention_dropout=enc_kwargs["dropout"],
            residual_dropout=enc_kwargs["dropout"],
            embed_dropout=enc_kwargs["dropout"],
            causal=False,
            seq_len=self.seq_len
        )
        future_proj = MLP(
            input_dim=enc_kwargs["embed_dim"],
            output_dim=self.z_dim if self.discrete else 2*self.z_dim
        )
        dec_kwargs = network_kwargs["decoder"]
        decoder = Decoder(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            z_dim=self.z_dim,
            num_time_delta=self.future_len+1,  # +1 because sometimes we may predict the s-a itself
            embed_dim=self.z_dim,
            hidden_dims=dec_kwargs["hidden_dims"]
        )
        prior_kwargs = network_kwargs["prior"]
        prior = MLP(
            input_dim=self.obs_dim+self.action_dim,
            output_dim=self.z_dim if self.discrete else 2*self.z_dim,
            hidden_dims=prior_kwargs["hidden_dims"]
        )
        reward_kwargs = network_kwargs["reward"]
        reward = RewardDecoder(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            z_dim=self.z_dim,
            embed_dim=reward_kwargs["embed_dim"],
            num_layers=reward_kwargs["num_layers"],
            num_heads=reward_kwargs["num_heads"],
            seq_len=self.seq_len,
            episode_len=self.seq_len,
            reward_act=reward_kwargs["reward_act"]
        )
        actor = vars(wiserl.module)[network_kwargs["actor"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=self.action_space.shape[0],
            **network_kwargs["actor"]
        )
        critic = vars(wiserl.module)[network_kwargs["critic"].pop("class")](
            input_dim=self.observation_space.shape[0]+self.action_space.shape[0],
            output_dim=1,
            **network_kwargs["critic"]
        )
        value = vars(wiserl.module)[network_kwargs["value"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=1,
            **network_kwargs["value"]
        )

        self.network = nn.ModuleDict({
            "future_encoder": future_encoder.to(self.device),
            "future_proj": future_proj.to(self.device),
            "decoder": decoder.to(self.device),
            "prior": prior.to(self.device),
            "reward": reward.to(self.device),
            "actor": actor.to(self.device),
            "critic": critic.to(self.device),
            "value": value.to(self.device)
        })
        self.target_network = nn.ModuleDict({
            "critic": make_target(self.network.critic)
        })

    def setup_optimizers(self, optim_kwargs):
        super().setup_optimizers(optim_kwargs)
        default_kwargs = optim_kwargs.get("default", {})
        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(optim_kwargs.get("reward", {}))
        self.optim["reward"] = vars(torch.optim)[reward_kwargs.pop("class")](
            self.network.reward.parameters(), **reward_kwargs
        )

    def setup_schedulers(self, scheduler_kwargs):
        reward_kwargs = scheduler_kwargs.pop("reward")
        self.schedulers["reward"] = LinearWarmupCosineAnnealingLR(
            self.optim["reward"],
            warmup_epochs=reward_kwargs["warmup_steps"],
            max_epochs=reward_kwargs["max_steps"]
        )
        return super().setup_schedulers(scheduler_kwargs)

    def select_reward(self, batch, deterministic=False):
        reward = []
        obs, action = batch["obs"], batch["action"]

        repeated_obs_action = torch.concat([obs, action], dim=-1)
        repeated_obs_action = repeated_obs_action.repeat([self.prior_sample, ] + [1,]*len(batch["obs"].shape))
        # repeated_out = self.network.prior(repeated_obs_action)
        # z_prior_dist = self.get_z_distribution(repeated_out)
        # z_prior = self.get_z_sample(z_prior_dist, reparameterize=False, deterministic=False)
        if "mask" in batch:
            mask = ~batch["mask"].to(torch.bool).repeat([self.prior_sample, ] + [1,]*len(batch["obs"].shape))
        else:
            mask = None

        tlen = repeated_obs_action.shape[2]
        for i_seg in range((tlen - 1) // self.label_seq_len + 1):
            repeated_obs_action_seg = repeated_obs_action[:, :, i_seg*self.label_seq_len:min((i_seg+1)*self.label_seq_len, tlen)]
            z_prior_dist = self.get_z_distribution(self.network.prior(repeated_obs_action_seg))
            z_prior_seg = self.get_z_sample(z_prior_dist, reparameterize=False, deterministic=False)
            # z_prior_seg = z_prior[:, :, i_seg*self.label_seq_len:min((i_seg+1)*self.label_seq_len, tlen)]
            P, B, L, E = repeated_obs_action_seg.shape
            if mask is not None:
                mask_seg = mask[:, :, i_seg*self.label_seq_len:min((i_seg+1)*self.label_seq_len, tlen), 0]
                mask_seg = mask_seg.reshape(P*B, L)
            else:
                mask_seg = None
            reward_seg = self.network.reward(
                z=z_prior_seg.reshape(P*B, L, -1),
                input=repeated_obs_action_seg.reshape(P*B, L, -1),
                timestep=None,
                key_padding_mask=mask_seg
            )
            reward_seg = reward_seg.reshape(P, B, L, -1)
            reward.append(reward_seg)
        reward = torch.concat(reward, dim=2).detach()
        reward = torch.nan_to_num(reward, nan=0.0)  # they will be masked anyway
        return reward.mean(dim=0).detach()

    def update_reward(self, obs_1, obs_2, action_1, action_2, label, extra_obs, extra_action):
        obs_action_1 = torch.concat([obs_1, action_1], dim=-1)
        obs_action_2 = torch.concat([obs_2, action_2], dim=-1)
        obs_action_total = torch.concat([obs_action_1, obs_action_2], dim=0)
        with torch.no_grad():
            # sample from posterior z distribution
            posterior_out = self.network.future_encoder(
                inputs=obs_action_total,
                timesteps=None, # consistent with vae training
                attention_mask=self.future_attention_mask,
                do_embedding=True
            )
            posterior_out = self.network.future_proj(posterior_out)
            z_posterior_dist = self.get_z_distribution(posterior_out)
            z_posterior = self.get_z_sample(z_posterior_dist, reparameterize=False, deterministic=not self.stoc_encoding)
            # sample from prior z distribution for regularization
            obs_action_extra = torch.concat([extra_obs, extra_action], dim=-1)
            repeated_obs_action_extra = obs_action_extra.repeat([self.prior_sample, 1, 1])
            repeated_prior_out = self.network.prior(repeated_obs_action_extra)
            z_prior_dist = self.get_z_distribution(repeated_prior_out)
            z_prior = self.get_z_sample(z_prior_dist, reparameterize=False, deterministic=False)
        # cross entropy loss
        reward_total = self.network.reward(
            z=z_posterior,
            input=obs_action_total,
            timestep=None,
            key_padding_mask=None # all data is valid
        )
        r1, r2 = torch.chunk(reward_total, 2, dim=0)
        logit = r2.sum(dim=1) - r1.sum(dim=1)
        label = label.float()
        reward_loss = self.reward_criterion(logit, label).mean()
        with torch.no_grad():
            reward_acc = ((logit > 0) == torch.round(label)).float().mean()
        # regularization
        reward_prior = torch.tensor([0.0, ])
        reg_loss = torch.tensor(0.0)
        self.optim["reward"].zero_grad()
        (reward_loss+self.reg_coef*reg_loss).backward()
        self.optim["reward"].step()
        return {
            "loss/reward_loss": reward_loss.item(),
            "loss/reg_loss": reg_loss.item(),
            "loss/reward_acc": reward_acc.item(),
            "misc/reward_post": reward_total.mean().item(),
            "misc/reward_post_abs": reward_total.abs().mean().item(),
            "misc/reward_prior_abs": reward_prior.abs().mean().item(),
        }
