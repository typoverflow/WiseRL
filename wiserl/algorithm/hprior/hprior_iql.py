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
from wiserl.module.net import attention
from wiserl.module.net.attention.twm import TransformerBasedWorldModel
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target
from wiserl.utils.optim import LinearWarmupCosineAnnealingLR


class Hindsight_PRIOR_IQL(OracleIQL):
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
        world_steps: int = 50000,
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
        self.world_steps = world_steps
        self.rm_label = rm_label
        assert rm_label
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.world_criterion = torch.nn.L1Loss(reduction="none")
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.prior_criterion = torch.nn.MSELoss(reduction="none")

    def setup_network(self, network_kwargs):
        super().setup_network(network_kwargs)
        # world model
        world_kwargs = network_kwargs["world"]
        world = TransformerBasedWorldModel(
            obs_dim=self.observation_space.shape[0],
            action_dim=self.action_space.shape[0],
            embed_dim=world_kwargs["embed_dim"],
            num_layers=world_kwargs["num_layers"],
            seq_len=self.max_seq_len,
            num_heads=world_kwargs["num_heads"],
        )
        self.network["world"] = world
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
        for attr in ["world", "reward"]:
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
        if step < self.world_steps:
            return self.update_world(
                obs=traj_batch["obs"],
                action=traj_batch["action"],
                next_obs=traj_batch["next_obs"],
            )
        else:
            return self.update_reward(
                obs_1=pref_batch["obs_1"],
                obs_2=pref_batch["obs_2"],
                action_1=pref_batch["action_1"],
                action_2=pref_batch["action_2"],
                label=pref_batch["label"],
            )
    
    def update_world(self, obs, action, next_obs) -> Dict:
        # obs [F_B, F_S, obs_dim], action [F_B, F_S, action_dim]
        F_B, F_S = obs.shape[:2]
        timestep = torch.arange(F_S, device=self.device).unsqueeze(0).expand(F_B, -1)
        pred_obs, attentions = self.network.world(obs, action, timestep)
        world_loss = self.world_criterion(pred_obs, next_obs).sum(0).mean()

        self.optim["world"].zero_grad()
        world_loss.backward()
        self.optim["world"].step()

        metrics = {
            "loss/world_loss": world_loss.item()
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
        obs = torch.concat([obs_1, obs_2], dim=0)
        action = torch.concat([action_1, action_2], dim=0)
        r = torch.concat([r1, r2], dim=1)
        predicted_return = r.sum(dim=2)
        _, attentions = self.network.world(obs, action, None)
        # get last attentions -> [F_B, num_layers, 2 * F_S]
        attentions = torch.stack([attn[:, -1] for attn in attentions], dim=1)
        # alpha = 1/L * sum_{l=1}^{L} (attn_{s_t}^l + attn_{a_t}^l) -> [F_B, F_S]
        attentions = attentions.reshape(*attentions.shape[:-1], -1, 2).sum(dim=-1)
        prior_importance = attentions.mean(dim=1)
        r_target = prior_importance * predicted_return
        r_target = r_target.unsqueeze(-1)
        prior_loss = self.prior_criterion(r, r_target).mean()
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
        for attr in ["world", "reward"]:
            state_dict = torch.load(os.path.join(path, attr+".pt"), map_location=self.device)
            self.network.__getattr__(attr).load_state_dict(state_dict)

    def save_pretrain(self, path):
        os.makedirs(path, exist_ok=True)
        for attr in ["world", "reward"]:
            state_dict = self.network.__getattr__(attr).state_dict()
            torch.save(state_dict, os.path.join(path, attr+".pt"))
