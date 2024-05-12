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
        expectile: float = 0.7,
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
        **kwargs
    ):
        self.expectile = expectile
        self.beta = beta
        self.max_exp_clip = max_exp_clip
        self.discount = discount
        self.tau = tau
        self.seq_len = seq_len
        self.future_len = future_len
        self.z_dim = z_dim
        self.prior_sample = prior_sample
        self.kl_loss_coef = kl_loss_coef
        self.kl_balance_coef = kl_balance_coef
        self.reg_coef = reg_coef
        self.vae_steps = vae_steps
        self.reward_steps = reward_steps
        self.rm_label = rm_label
        super().__init__(*args, **kwargs)
        # define the attention mask for future prediction
        causal_mask = torch.tril(torch.ones([seq_len, seq_len]), diagonal=-1).bool()
        future_mask = torch.triu(torch.ones([seq_len, seq_len]), diagonal=future_len+1).bool()
        self.future_attention_mask = torch.bitwise_or(causal_mask, future_mask).to(self.device)
        # self.vae_causal_mask = torch.tril(torch.ones([future_len, future_len]), diagonal=-1).bool().to(self.device)
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

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
            num_time_delta=self.future_len+1,  # +1 because sometimes we may predict the s-a itself
            embed_dim=self.z_dim,
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
        reward_act = network_kwargs["reward"].pop("reward_act")
        reward = vars(wiserl.module)[network_kwargs["reward"].pop("class")](
            input_dim=self.observation_space.shape[0]+self.action_space.shape[0]+self.z_dim,
            output_dim=1,
            **network_kwargs["reward"]
        )
        reward = nn.Sequential(
            reward,
            nn.Sigmoid() if reward_act == "sigmoid" else nn.Identity()
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
        self.optim = {}
        default_kwargs = optim_kwargs.get("default", {})
        for k in {"future_encoder", "future_proj", "decoder", "prior", "reward", "actor", "critic", "value"}:
            this_kwargs = default_kwargs.copy()
            this_kwargs.update(optim_kwargs.get(k, {}))
            self.optim[k] = vars(torch.optim)[this_kwargs.pop("class")](
                self.network[k].parameters(),
                **this_kwargs
            )

    def select_action(self, batch, deterministic: bool=True):
        obs = batch["obs"]
        action, *_ = self.network.actor.sample(obs, deterministic=deterministic)
        return action.squeeze().cpu().numpy()

    def select_reward(self, batch, deterministic=False):
        obs, action = batch["obs"], batch["action"]
        repeated_obs_action = torch.concat([obs, action], dim=-1)
        repeated_obs_action = repeated_obs_action.repeat([self.prior_sample, ] + [1,]*len(repeated_obs_action.shape))
        z_prior, *_ = self.network.prior.sample(repeated_obs_action, deterministic=False)
        reward = self.network.reward(torch.concat([repeated_obs_action, z_prior], dim=-1)).mean(dim=0)
        return reward.detach()

    def pretrain_step(self, batches, step: int, total_steps: int):
        unlabel_batch, pref_batch, rl_batch = batches
        assert step <= self.reward_steps + self.vae_steps, "pretrain step overflow"
        if step < self.vae_steps:
            return self.update_vae(
                obs=unlabel_batch["obs"],
                action=unlabel_batch["action"],
                timestep=unlabel_batch["timestep"],
                mask=unlabel_batch["mask"]
            )
        else:
            return self.update_reward(
                obs_1=pref_batch["obs_1"],
                obs_2=pref_batch["obs_2"],
                action_1=pref_batch["action_1"],
                action_2=pref_batch["action_2"],
                label=pref_batch["label"],
                extra_obs=rl_batch["obs"],
                extra_action=rl_batch["action"]
            )

    def train_step(self, batches, step: int, total_steps: int):
        rl_batch = batches[0]
        return self.update_agent(
            obs=rl_batch["obs"],
            action=rl_batch["action"],
            next_obs=rl_batch["next_obs"],
            reward=rl_batch["reward"],
            terminal=rl_batch["terminal"]
        )

    def update_agent(self, obs, action, next_obs, reward, terminal):
        with torch.no_grad():
            self.target_network.eval()
            q_old = self.target_network.critic(obs, action)
            q_old = torch.min(q_old, dim=0)[0]
        # update value
        v_pred = self.network.value(obs)
        v_loss = expectile_regression(v_pred, q_old, expectile=self.expectile).mean()
        self.optim["value"].zero_grad()
        v_loss.backward()
        self.optim["value"].step()

        # update actor
        with torch.no_grad():
            adv = q_old - v_pred
            exp_adv = (adv / self.beta).exp().clip(max=self.max_exp_clip)
        if isinstance(self.network.actor, DeterministicActor):
            policy_out = torch.sum((self.network.actor.sample(obs)[0] - action)**2, dim=-1, keepdim=True)
        elif isinstance(self.network.actor, GaussianActor):
            policy_out = - self.network.actor.evaluate(obs, action)[0]
        actor_loss = (exp_adv * policy_out).mean()
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()

        sync_target(self.network.critic, self.target_network.critic, tau=self.tau)

        # update critic
        with torch.no_grad():
            if self.rm_label:
                reward = reward
            else:
                reward = self.select_reward({"obs": obs, "action": action}, deterministic=False)
            target_q = self.network.value(next_obs)
            target_q = reward + self.discount * (1-terminal.float())*target_q
        q_pred = self.network.critic(obs, action)
        q_loss = (q_pred - target_q.unsqueeze(0)).pow(2).sum(0).mean()
        self.optim["critic"].zero_grad()
        q_loss.backward()
        self.optim["critic"].step()

        return {
            "loss/q_loss": q_loss.item(),
            "loss/v_loss": v_loss.item(),
            "loss/actor_loss": actor_loss.item(),
            "misc/q_pred": q_pred.mean().item(),
            "misc/v_pred": v_pred.mean().item(),
            "misc/adv": adv.mean().item(),
            "misc/reward_prior": reward.mean().item()
        }

    def update_reward(self, obs_1, obs_2, action_1, action_2, label, extra_obs, extra_action):
        obs_action_1 = torch.concat([obs_1, action_1], dim=-1)
        obs_action_2 = torch.concat([obs_2, action_2], dim=-1)
        obs_action_total = torch.concat([obs_action_1, obs_action_2], dim=0)
        with torch.no_grad():
            z_posterior, _, info = self.network.future_proj.sample(
                self.network.future_encoder(
                    inputs=obs_action_total,
                    timesteps=None, # consistent with vae training
                    attention_mask=self.future_attention_mask,
                    do_embedding=True
                )
            )
            obs_action_extra = torch.concat([extra_obs, extra_action], dim=-1)
            repeated_obs_action_extra = obs_action_extra.repeat([self.prior_sample, 1, 1])
            z_prior, _, info = self.network.prior.sample(repeated_obs_action_extra, deterministic=False)
        # cross entropy loss
        reward_total = self.network.reward(torch.concat([obs_action_total, z_posterior], dim=-1))
        r1, r2 = torch.chunk(reward_total, 2, dim=0)
        logit = r2.sum(dim=1) - r1.sum(dim=1)
        label = label.float()
        reward_loss = self.reward_criterion(logit, label).mean()
        with torch.no_grad():
            reward_acc = ((logit > 0) == torch.round(label)).float().mean()
        # regularization
        reward_prior = self.network.reward(torch.concat([repeated_obs_action_extra, z_prior], dim=-1))
        if self.reg_coef > 0.0:
            reg_loss = torch.nn.functional.huber_loss(reward_prior, torch.zeros_like(reward_prior), delta=1.0)
        else:
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

    def update_vae(self, obs: torch.Tensor, action: torch.Tensor, timestep: torch.Tensor, mask: torch.Tensor):
        B, L, *_ = obs.shape
        obs_action = torch.concat([obs, action], dim=-1)
        out = self.network.future_encoder(
            inputs=obs_action,
            timesteps=None, # here we don't use the timestep from dataset, but use the default `np.arange(len)`
            attention_mask=self.future_attention_mask,
            key_padding_mask=(1-mask).squeeze(-1).bool(),
            do_embedding=True
        )
        z_posterior, _, info = self.network.future_proj.sample(out, deterministic=False, return_mean_logstd=True)
        z_mean, z_logstd = info["mean"], info["logstd"]

        def compute_kl_loss(mean_post, logstd_post, mean_prior, logstd_prior):
            var_post = (2*logstd_post).exp()
            var_prior = (2*logstd_prior).exp()
            return logstd_prior - logstd_post + (var_post + (mean_post - mean_prior).square()) / (2*var_prior) - 0.5

        # select the time index
        num_select = B * 4
        x = torch.randint(0, L, [num_select, ]).to(self.device)
        delta_t = torch.randint(0, self.future_len+1, [num_select, ]).to(self.device)
        y = (x+delta_t).clip(max=L-1)
        delta_t = (y-x).repeat(B, 1)
        input_obs_action = obs_action[:, x, :]
        input_z_posterior = z_posterior[:, x, :]
        target_obs_action = obs_action[:, y, :]
        pred_obs_action = self.network.decoder(
            input_obs_action,
            input_z_posterior,
            delta_t
        )

        recon_loss = torch.nn.functional.mse_loss(pred_obs_action, target_obs_action, reduction="none")
        recon_loss = recon_loss.mean()

        # KL divergence
        z_prior_mean, z_prior_logstd = self.network.prior.forward(obs_action)
        prior_kl_loss = compute_kl_loss(z_mean.detach(), z_logstd.detach(), z_prior_mean, z_prior_logstd).mean()
        post_kl_loss = compute_kl_loss(z_mean, z_logstd, z_prior_mean.detach(), z_prior_logstd.detach()).mean()
        kl_loss = self.kl_balance_coef * prior_kl_loss + (1-self.kl_balance_coef) * post_kl_loss

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
            "loss/kl_loss": kl_loss.item(),
            "loss/prior_kl_loss": prior_kl_loss.item(),
            "loss/post_kl_loss": post_kl_loss.item(),
        }

    def load_pretrain(self, path):
        for attr in ["future_encoder", "future_proj", "decoder", "prior", "reward"]:
            state_dict = torch.load(os.path.join(path, attr+".pt"), map_location=self.device)
            self.network.__getattr__(attr).load_state_dict(state_dict)

    def save_pretrain(self, path):
        os.makedirs(path, exist_ok=True)
        for attr in ["future_encoder", "future_proj", "decoder", "prior", "reward"]:
            state_dict = self.network.__getattr__(attr).state_dict()
            torch.save(state_dict, os.path.join(path, attr+".pt"))
