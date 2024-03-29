import itertools
from operator import itemgetter
from typing import Any, Dict, Optional, Sequence, Tuple, Type

import gym
import imageio
import numpy as np
import torch
import torch.nn as nn
from offlinerllib.module.net.attention.gpt2 import GPT2

import wiserl.module
from wiserl.algorithm.base import Algorithm
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.module.encoder_decoder import MLPEncDec
from wiserl.module.net.mlp import MLP
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target


class Decoder(nn.Module):
    def __init__(
        self,
        z_dim: int,
        obs_dim: int,
        action_dim: int,
        num_time_delta: int,
        embed_dim: int,
        hidden_dims: Sequence[int] = [],
    ):
        super().__init__()
        self.obs_act_encoder = torch.nn.Linear(obs_dim+action_dim, embed_dim)
        self.z_encoder = torch.nn.Linear(z_dim, embed_dim)
        # self.time_encoder = torch.nn.Linear(1, embed_dim)
        self.time_encoder = torch.nn.Embedding(num_time_delta, embed_dim)
        self.unify = MLP(
            input_dim=3*embed_dim,
            output_dim=obs_dim+action_dim,
            hidden_dims=hidden_dims
        )

    def forward(self, obs_act, z, delta_t):
        out = torch.concat([
            self.obs_act_encoder(obs_act),
            self.z_encoder(z),
            self.time_encoder(delta_t),
        ], dim=-1)
        out = torch.nn.functional.relu(out)
        return self.unify(out)


class HindsightPreferenceLearning(Algorithm):
    def __init__(
        self,
        *args,
        seq_len: int = 50,
        z_dim: int = 64,
        kl_loss_coef: float = 1.0,
        **kwargs
    ):
        self.seq_len = seq_len
        self.z_dim = z_dim
        self.kl_loss_coef = kl_loss_coef
        super().__init__(*args, **kwargs)
        self.future_attention_mask = torch.tril(torch.ones([seq_len, seq_len]), diagonal=-1).to(self.device)

    def setup_network(self, network_kwargs) -> None:
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        enc_kwargs = network_kwargs["encoder"]
        future_encoder = GPT2(
            input_dim=self.obs_dim+self.action_dim,
            embed_dim=enc_kwargs["embed_dim"],
            num_layers=enc_kwargs["num_layers"],
            num_heads=enc_kwargs["num_heads"],
            causal=False,
        )
        future_proj = GaussianActor(
            input_dim=enc_kwargs["embed_dim"],
            output_dim=self.z_dim,
            reparameterize=True,
            conditioned_logstd=True
        )
        dec_kwargs = network_kwargs["decoder"]
        decoder = Decoder(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            z_dim=self.z_dim,
            num_time_delta=self.seq_len,
            embed_dim=dec_kwargs["embed_dim"],
            hidden_dims=dec_kwargs["hidden_dims"]
        )
        prior_kwargs = network_kwargs["prior"]
        prior = GaussianActor(
            input_dim=self.obs_dim+self.action_dim,
            output_dim=self.z_dim,
            reparameterize=True,
            conditioned_logstd=True,
            hidden_dims=prior_kwargs["hidden_dims"]
        )

        self.network = nn.ModuleDict({
            "future_encoder": future_encoder.to(self.device),
            "future_proj": future_proj.to(self.device),
            "decoder": decoder.to(self.device),
            "prior": prior.to(self.device)
        })

    def setup_optimizers(self, optim_kwargs):
        self.optim = {}
        default_kwargs = optim_kwargs.get("default", {})
        for k in {"future_encoder", "future_proj", "decoder", "prior"}:
            this_kwargs = default_kwargs.copy()
            this_kwargs.update(optim_kwargs.get(k, {}))
            self.optim[k] = vars(torch.optim)[this_kwargs.pop("class")](
                self.network[k].parameters(),
                **this_kwargs
            )

    def select_action(self, *args, **kwargs):
        return None

    def train_step(self, batches, step: int, total_steps: int):
        obs, action, terminal, timestep, mask = itemgetter("obs", "action", "terminal", "timestep", "mask")(batches[0])
        B, L, *_ = obs.shape
        obs_action = torch.concat([obs, action], dim=-1)
        out = self.network.future_encoder(
            inputs=obs_action,
            timesteps=timestep,
            attention_mask=self.future_attention_mask,
            key_padding_mask=(1-mask).squeeze(-1).bool(),
            do_embedding=True
        )
        z_posterior, _, info = self.network.future_proj.sample(out, deterministic=False, return_mean_logstd=True)
        z_mean, z_logstd = info["mean"], info["logstd"]

        # select the time index
        num_select = B * 4
        x = torch.randint(0, L-1, [num_select, ]).to(self.device)
        y = torch.randint(0, 999999, [num_select, ]).to(self.device)
        y = torch.remainder(y, L-x-1) + x + 1

        input_obs_action = obs_action[:, x, :]
        input_z_posterior = z_posterior[:, x, :]
        input_delta_t = (y - x).repeat(B, 1)
        # input_loss_mask = mask[:, y, :]
        target_obs_action = obs_action[:, y, :]
        pred_obs_action = self.network.decoder(
            input_obs_action,
            input_z_posterior,
            input_delta_t
        )

        recon_loss = torch.nn.functional.mse_loss(pred_obs_action, target_obs_action, reduction="none")
        # recon_loss = (recon_loss * input_loss_mask).mean(-1).sum() / input_loss_mask.sum()
        recon_loss = recon_loss.mean()

        # KL divergence
        z_prior_mean, z_prior_logstd = self.network.prior.forward(obs_action)
        z_var = (2*z_logstd).exp()
        z_piror_var = (2*z_prior_logstd).exp()
        kl_loss = z_prior_logstd - z_logstd + (z_var + (z_mean - z_prior_mean).square()) / (2*z_piror_var) - 0.5
        kl_loss = kl_loss.mean()

        self.optim["future_encoder"].zero_grad()
        self.optim["future_proj"].zero_grad()
        self.optim["decoder"].zero_grad()
        self.optim["prior"].zero_grad()
        (recon_loss + self.kl_loss_coef * kl_loss).backward()
        self.optim["future_encoder"].step()
        self.optim["future_proj"].step()
        self.optim["decoder"].step()
        self.optim["prior"].step()

        return {
            "loss/recon_loss": recon_loss.item(),
            "loss/kl_loss": kl_loss.item()
        }
