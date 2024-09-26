import itertools
import os
from operator import itemgetter
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.oracle_iql import OracleIQL
from wiserl.module.net.attention.preference_discriminator import PreferenceDiscriminator
from wiserl.utils.misc import sync_target


class Discriminator_PRIOR_IQL(OracleIQL):
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
        max_seq_len: int = 100,
        bc_steps: int = 15000,
        discriminator_steps: int = 15000,
        prior_coef: float = 1.0,
        rm_label: bool = True,
        **kwargs
    ) -> None:
        self.max_seq_len = max_seq_len
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
        self.prior_coef = prior_coef
        self.bc_steps = bc_steps
        self.discriminator_steps = discriminator_steps
        self.rm_label = rm_label
        assert rm_label
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.bc_criterion = torch.nn.MSELoss(reduction="none")
        self.discriminator_criterion = torch.nn.BCELoss(reduction="none")
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.prior_criterion = torch.nn.MSELoss(reduction="none")

    def setup_network(self, network_kwargs):
        super().setup_network(network_kwargs)
        # discriminator model
        discriminator_kwargs = network_kwargs["discriminator"]
        discriminator = PreferenceDiscriminator(
            obs_dim=self.observation_space.shape[0],
            action_dim=self.action_space.shape[0],
            embed_dim=discriminator_kwargs["embed_dim"],
            num_layers=discriminator_kwargs["num_layers"],
            seq_len=self.max_seq_len,
            num_heads=discriminator_kwargs["num_heads"],
        )
        self.network["discriminator"] = discriminator
        # reward model
        reward_act = {
            "identity": nn.Identity(),
            "sigmoid": nn.Sigmoid(),
        }.get(network_kwargs["reward"].pop("reward_act"))
        reward = vars(wiserl.module)[network_kwargs["reward"].pop("class")](
            input_dim=self.observation_space.shape[0]+self.action_space.shape[0],
            output_dim=1,
            **network_kwargs["reward"]
        )
        self.network["reward"] = nn.Sequential(self.network["encoder"], reward, reward_act)

    def setup_optimizers(self, optim_kwargs):
        super().setup_optimizers(optim_kwargs)
        default_kwargs = optim_kwargs.get("default", {})
        for attr in ["discriminator", "reward"]:
            kwargs = default_kwargs.copy()
            kwargs.update(optim_kwargs.get(attr, {}))
            optim = vars(torch.optim)[kwargs.pop("class")](
                self.network.__getattr__(attr).parameters(), **kwargs
            )
            self.optim[attr] = optim

    def select_action(self, batch, deterministic: bool=True):
        return super().select_action(batch, deterministic)

    def select_reward(self, batch, deterministic=False):
        obs, action = batch["obs"], batch["action"]
        reward = self.network.reward(torch.concat([obs, action], dim=-1))
        return reward.mean(0).detach()

    def pretrain_step(self, batches, step: int, total_steps: int) -> Dict:
        traj_batch, pref_batch = batches
        if step < self.bc_steps:
            return self.update_discriminator_bc(
                obs=traj_batch["obs"],
                action=traj_batch["action"],
                next_obs=traj_batch["next_obs"],
                next_action=traj_batch["next_action"],
            )
        elif step < self.bc_steps + self.discriminator_steps:
            return self.update_discriminator(
                obs_1=pref_batch["obs_1"],
                obs_2=pref_batch["obs_2"],
                action_1=pref_batch["action_1"],
                action_2=pref_batch["action_2"],
                label=pref_batch["label"],
            )
        else:
            return self.update_reward(
                obs_1=pref_batch["obs_1"],
                obs_2=pref_batch["obs_2"],
                action_1=pref_batch["action_1"],
                action_2=pref_batch["action_2"],
                label=pref_batch["label"],
            )
    
    def update_discriminator_bc(self, obs, action, next_obs, next_action) -> Dict:
        # obs [F_B, F_S, obs_dim], action [F_B, F_S, action_dim]
        obs1, obs2 = obs.chunk(2, dim=1)
        action1, action2 = action.chunk(2, dim=1)
        next_obs1, next_obs2 = next_obs.chunk(2, dim=1)
        next_action1, next_action2 = next_action.chunk(2, dim=1)
        pred_obs1, pred_act1, pred_obs2, pred_act2, pred_label, attentions = self.network.discriminator(obs1, action1, obs2, action2)
        bc_loss = (
            self.bc_criterion(pred_obs1, next_obs1).sum(0).mean() + self.bc_criterion(pred_act1, next_action1).sum(0).mean() +
            self.bc_criterion(pred_obs2, next_obs2).sum(0).mean() + self.bc_criterion(pred_act2, next_action2).sum(0).mean()
        ) / 4

        self.optim["discriminator"].zero_grad()
        bc_loss.backward()
        self.optim["discriminator"].step()

        metrics = {
            "loss/bc_loss": bc_loss.item()
        }
        return metrics


    def update_discriminator(self, obs_1, obs_2, action_1, action_2, label) -> Dict:
        # obs [F_B, F_S, obs_dim], action [F_B, F_S, action_dim]
        pred_obs1, pred_act1, pred_obs2, pred_act2, pred_label, attentions = self.network.discriminator(obs_1, action_1, obs_2, action_2)
        discriminator_loss = self.discriminator_criterion(pred_label, label.float()).sum(0).mean()
        self.optim["discriminator"].zero_grad()
        discriminator_loss.backward()
        self.optim["discriminator"].step()

        metrics = {
            "loss/discriminator_loss": discriminator_loss.item()
        }
        return metrics


    def update_reward(self, obs_1, obs_2, action_1, action_2, label) -> Dict:
        F_B, F_S = obs_1.shape[0:2]
        all_obs = torch.concat([
            obs_1.reshape(-1, self.obs_dim),
            obs_2.reshape(-1, self.obs_dim)
        ])
        all_action = torch.concat([
            action_1.reshape(-1, self.action_dim),
            action_2.reshape(-1, self.action_dim)
        ])
        self.network.reward.train()
        all_reward = self.network.reward(torch.concat([all_obs, all_action], dim=-1))
        r1, r2 = torch.chunk(all_reward, 2, dim=1)
        E = r1.shape[0]
        r1, r2 = r1.reshape(E, F_B, F_S, 1), r2.reshape(E, F_B, F_S, 1)
        logits = r2.sum(dim=2) - r1.sum(dim=2)
        labels = label.float().unsqueeze(0).expand_as(logits)
        reward_loss = self.reward_criterion(logits, labels).sum(0).mean()
        reg_loss = (r1**2).sum(0).mean() + (r2**2).sum(0).mean()
        # hindsight prior loss by attention weights
        predicted_return1, predicted_return2 = r1.sum(dim=2), r2.sum(dim=2)
        _, _, _, _, _, attentions = self.network.discriminator(obs_1, action_1, obs_2, action_2)
        # get last attentions -> [F_B, num_layers, 2 * (F_S + 1)]
        attentions = torch.stack([attn[:, -1] for attn in attentions], dim=1)
        # alpha = 1/L * sum_{l=1}^{L} (attn_{s_t}^l + attn_{a_t}^l) -> [F_B, F_S]
        attentions = attentions.reshape(*attentions.shape[:-1], -1, 2).sum(dim=-1)
        prior_importance = attentions.mean(dim=1)
        prior_importance1, prior_importance2 = torch.chunk(prior_importance, 2, dim=1)
        prior_importance1, prior_importance2 = prior_importance1[:, :-1], prior_importance2[:, :-1]
        # normalize the importance
        prior_importance1 = prior_importance1 / prior_importance1.sum(dim=1, keepdim=True)
        prior_importance2 = prior_importance2 / prior_importance2.sum(dim=1, keepdim=True)
        r_target1 = prior_importance1 * predicted_return1
        r_target1 = r_target1.unsqueeze(-1)
        r_target2 = prior_importance2 * predicted_return2
        r_target2 = r_target2.unsqueeze(-1)
        prior_loss = (self.prior_criterion(r1, r_target1).mean() + self.prior_criterion(r2, r_target2).mean()) / 2
        with torch.no_grad():
            reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()

        self.optim["reward"].zero_grad()
        (reward_loss + self.reward_reg * reg_loss + self.prior_coef * prior_loss).backward()
        self.optim["reward"].step()

        metrics = {
            "loss/reward_loss": reward_loss.item(),
            "loss/reward_reg_loss": reg_loss.item(),
            "loss/prior_loss": reg_loss.item(),
            "misc/reward_acc": reward_accuracy.item(),
            "misc/reward_value": all_reward.mean().item()
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

        with torch.no_grad():
            self.target_network.eval()
            q_old = self.target_network.critic(obs, action)
            q_old = torch.min(q_old, dim=0)[0]

        # compute the loss for value network
        v_loss, v_pred = self.v_loss(obs.detach(), q_old)
        self.optim["value"].zero_grad()
        v_loss.backward()
        self.optim["value"].step()

        # compute the loss for actor
        actor_loss, advantage = self.actor_loss(obs, action, q_old, v_pred.detach())
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
            "misc/advantage": advantage.mean().item()
        }
        return metrics

    def load_pretrain(self, path):
        for attr in ["discriminator", "reward"]:
            state_dict = torch.load(os.path.join(path, attr+".pt"), map_location=self.device)
            self.network.__getattr__(attr).load_state_dict(state_dict)

    def save_pretrain(self, path):
        os.makedirs(path, exist_ok=True)
        for attr in ["discriminator", "reward"]:
            state_dict = self.network.__getattr__(attr).state_dict()
            torch.save(state_dict, os.path.join(path, attr+".pt"))
