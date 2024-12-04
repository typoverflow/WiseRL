import itertools
import os
from operator import itemgetter
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.oracle_iql import OracleIQL
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target


class RBTIQL(OracleIQL):
    def __init__(
        self,
        *args,
        expectile: float = 0.7,
        beta: float = 0.3333,
        max_exp_clip: float = 100.0,
        discount: float = 0.99,
        tau: float = 0.005,
        target_freq: int = 1,
        reward_reg: float = 0.0,
        rm_label: bool = True,
        logstd_coeff: float = 0.1,
        logstd_threshold: float = 0.1,
        use_std_weights: bool = False,
        use_std_weights_for_actor: bool = False,
        use_low_quantile: bool = False,
        low_quantile_ratio: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__(
            *args,
            expectile=expectile,
            beta=beta,
            max_exp_clip=max_exp_clip,
            discount=discount,
            tau=tau,
            target_freq=target_freq,
            **kwargs
        )
        self.reward_reg = reward_reg
        self.rm_label = rm_label
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.threshold_criterion = torch.relu
        self.logstd_coeff = logstd_coeff
        self.logstd_threshold = logstd_threshold
        self.use_std_weights = use_std_weights
        self.use_std_weights_for_actor = use_std_weights_for_actor
        self.use_low_quantile = use_low_quantile
        self.low_quantile_ratio = low_quantile_ratio

    def setup_network(self, network_kwargs):
        super().setup_network(network_kwargs)
        reward_act = {
            "identity": nn.Identity(),
            "sigmoid": nn.Sigmoid(),
        }.get(network_kwargs["reward"].pop("reward_act"))
        reward = vars(wiserl.module)[network_kwargs["reward"].pop("class")](
            input_dim=self.observation_space.shape[0] + self.action_space.shape[0],
            output_dim=2,  # Output mean and logstd
            **network_kwargs["reward"]
        )
        self.network["reward"] = nn.Sequential(self.network["encoder"], reward)
        self.network["reward_act"] = reward_act

    def setup_optimizers(self, optim_kwargs):
        super().setup_optimizers(optim_kwargs)
        default_kwargs = optim_kwargs.get("default", {})
        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(optim_kwargs.get("reward", {}))
        self.optim["reward"] = vars(torch.optim)[reward_kwargs.pop("class")](self.network.reward.parameters(), **reward_kwargs)

    def select_action(self, batch, deterministic: bool=True):
        return super().select_action(batch, deterministic)

    def select_reward(self, batch, deterministic=False):
        obs, action = batch["obs"], batch["action"]
        reward_mean_logstd = self.network.reward(torch.concat([obs, action], dim=-1))
        reward_mean, reward_logstd = torch.chunk(reward_mean_logstd, 2, dim=-1)
        reward_mean = self.network.reward_act(reward_mean)
        if self.use_low_quantile:
            reward = reward_mean - self.low_quantile_ratio * torch.exp(reward_logstd)
        else:
            reward = reward_mean
        return reward.mean(0).detach()

    def pretrain_step(self, batches, step: int, total_steps: int) -> Dict:
        batch = batches[0]
        F_B, F_S = batch["obs_1"].shape[0:2]
        all_obs = torch.concat([
            batch["obs_1"].reshape(-1, self.obs_dim),
            batch["obs_2"].reshape(-1, self.obs_dim)
        ])
        all_action = torch.concat([
            batch["action_1"].reshape(-1, self.action_dim),
            batch["action_2"].reshape(-1, self.action_dim)
        ])
        self.network.reward.train()
        all_reward_mean_logstd = self.network.reward(torch.concat([all_obs, all_action], dim=-1))
        reward_mean, reward_logstd = torch.chunk(all_reward_mean_logstd, 2, dim=-1)
        reward_mean = self.network.reward_act(reward_mean)
        reward_std = torch.exp(reward_logstd)
        reward_low_quantile = reward_mean - self.low_quantile_ratio * reward_std
        r1_mean, r2_mean = torch.chunk(reward_mean, 2, dim=1)
        r1_std, r2_std = torch.chunk(reward_std, 2, dim=1)
        E = r1_mean.shape[0]
        r1_mean = r1_mean.reshape(E, F_B, F_S, 1)
        r2_mean = r2_mean.reshape(E, F_B, F_S, 1)
        r1_std = r1_std.reshape(E, F_B, F_S, 1)
        r2_std = r2_std.reshape(E, F_B, F_S, 1)
        pref_std = torch.sqrt((r1_std**2).sum(dim=2) + (r2_std**2).sum(dim=2))
        logits = (r2_mean.sum(dim=2) - r1_mean.sum(dim=2)) / pref_std
        labels = batch["label"].float().unsqueeze(0).expand_as(logits)
        reward_loss = self.reward_criterion(logits, labels).sum(0).mean()
        logstd_loss = self.threshold_criterion(self.logstd_threshold - reward_logstd).sum(0).mean()
        reg_loss = (r1_mean**2).sum(0).mean() + (r2_mean**2).sum(0).mean()
        with torch.no_grad():
            reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()

        self.optim["reward"].zero_grad()
        (reward_loss + self.reward_reg * reg_loss + self.logstd_coeff * logstd_loss).backward()
        self.optim["reward"].step()

        metrics = {
            "loss/reward_loss": reward_loss.item(),
            "loss/reward_reg_loss": reg_loss.item(),
            "misc/logstd_loss": logstd_loss.item(),
            "misc/reward_acc": reward_accuracy.item(),
            "misc/reward_value": reward_mean.mean().item(),
            "misc/reward_low_quantile": reward_low_quantile.mean().item(),
            "misc/reward_logstd": reward_logstd.mean().item(),
            "misc/reward_std": reward_std.mean().item(),
            "misc/reward_std_std": reward_std.std().item(),
            "misc/pref_std": pref_std.mean().item(),
            "misc/pref_std_std": pref_std.std().item(),
        }
        return metrics

    def train_step(self, batches, step: int, total_steps: int) -> Dict:
        rl_batch = batches[0]
        obs, action, next_obs, terminal = itemgetter("obs", "action", "next_obs", "terminal")(rl_batch)
        terminal = terminal.float()
        if self.rm_label:
            reward = itemgetter("reward")(rl_batch)
        else:
            with torch.no_grad():
                reward = self.select_reward({"obs": obs, "action": action}, deterministic=True)

        # Compute weights if use_std_weights is True
        if self.use_std_weights:
            with torch.no_grad():
                reward_mean_logstd = self.network.reward(torch.concat([obs, action], dim=-1))
                _, reward_logstd = torch.chunk(reward_mean_logstd, 2, dim=-1)
                weights = 1 / (torch.exp(reward_logstd) ** 2)
        else:
            weights = None

        with torch.no_grad():
            self.target_network.eval()
            q_old = self.target_network.critic(obs, action)
            q_old = torch.min(q_old, dim=0)[0]

        # compute the loss for value network
        v_loss, v_pred = self.v_loss(obs.detach(), q_old, weights=weights)
        self.optim["value"].zero_grad()
        v_loss.backward()
        self.optim["value"].step()

        # compute the loss for actor
        actor_loss, advantage = self.actor_loss(obs, action, q_old, v_pred.detach(), weights=weights if self.use_std_weights_for_actor else None)
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()

        # compute the loss for q
        q_loss, q_pred = self.q_loss(obs, action, next_obs, reward, terminal)
        self.optim["critic"].zero_grad()
        q_loss.backward()
        self.optim["critic"].step()

        for _, scheduler in self.schedulers.items():
            scheduler.step()

        if step % self.target_freq == 0:
            sync_target(self.network.critic, self.target_network.critic, tau=self.tau)

        metrics = {
            "loss/q_loss": q_loss.item(),
            "loss/v_loss": v_loss.item(),
            "loss/actor_loss": actor_loss.item(),
            "misc/q_pred": q_pred.mean().item(),
            "misc/v_pred": v_pred.mean().item(),
            "misc/advantage": advantage.mean().item(),
            "misc/weights": weights.mean().item() if weights is not None else 1.0,
            "misc/weights_std": weights.std().item() if weights is not None else 0.0,
        }
        return metrics

    def load_pretrain(self, path):
        for attr in ["reward"]:
            state_dict = torch.load(os.path.join(path, attr+".pt"), map_location=self.device)
            self.network.__getattr__(attr).load_state_dict(state_dict)

    def save_pretrain(self, path):
        os.makedirs(path, exist_ok=True)
        for attr in ["reward"]:
            state_dict = self.network.__getattr__(attr).state_dict()
            torch.save(state_dict, os.path.join(path, attr+".pt"))
