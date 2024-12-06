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


class RPL_IQL(OracleIQL):
    def __init__(
        self,
        *args,
        num_tasks: int = 4,
        alpha: float = 0.7,
        expectile: float = 0.7,
        beta: float = 0.3333,
        reward_reg: float = 0.0,
        max_exp_clip: float = 100.0,
        discount: float = 0.99,
        tau: float = 0.005,
        target_freq: int = 1,
        **kwargs
    ):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.reward_reg = reward_reg
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
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
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

    def setup_network(self, network_kwargs):
        super().setup_network(network_kwargs)
        reward_act = {
            "identity": nn.Identity(),
            "sigmoid": nn.Sigmoid()
        }.get(network_kwargs["reward"].pop("reward_act"))
        reward = vars(wiserl.module)[network_kwargs["reward"].pop("class")](
            input_dim=self.observation_space.shape[0]+self.action_space.shape,
            output_dim=1,
            **network_kwargs["reward"]
        )
        optimal = vars(wiserl.module)[network_kwargs["optimal"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=1,
            ensemble_size=self.num_tasks*network_kwargs["opt"]["ensemble_size"]
        )

        self.network["reward"] = nn.Sequential(self.network["encoder"], reward, reward_act)
        self.network["optimal"] = optimal
        self.target_network["reward"] = make_target(self.network["reward"])
        self.target_network["optimal"] = make_target(self.network["optimal"])

    def setup_optimizers(self, optim_kwargs):
        super().setup_optimizers(optim_kwargs)
        default_kwargs = optim_kwargs.get("default", {})
        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(optim_kwargs.get("reward", {}))
        optimal_kwargs = default_kwargs.copy()
        optimal_kwargs.update(optim_kwargs.get("optimal", {}))
        self.optim["reward"] = vars(torch.optim)[reward_kwargs.pop("class")](self.network.reward.parameters(), **reward_kwargs)
        self.optim["optimal"] = vars(torch.optim)[optimal_kwargs.pop("class")](self.network.optimal.parameters(), **optimal_kwargs)

    def select_reward(self, batch, deterministic=False):
        obs, action = batch["obs"], batch["action"]
        reward = self.network.reward(torch.concat([obs, action], dim=-1))
        return reward.mean(0).detach()

    def get_optimal_values(self, model, obs, task_id):
        original_shape = obs.shape[:-1]
        optimal_values = model(obs).reshape(self.num_tasks, -1, *original_shape, 1)
        optimal_values = torch.gather(
            optimal_values,
            0,
            task_id.unsqueeze(0).expand(*optimal_values.shape[1:]).unsqueeze(0)
        )
        return optimal_values

    def pretrain_step(self, batches, step: int, total_steps: int) -> Dict:
        batch = batches[0]
        B, S = batch["obs_1"].shape[0:2]
        all_obs = torch.concat([
            batch["obs_1"],
            batch["obs_2"]
        ], dim=0)
        all_action = torch.concat([
            batch["action_1"],
            batch["action_2"]
        ], dim=0)
        all_next_obs = torch.concat([
            batch["next_obs_1"],
            batch["next_obs_2"]
        ], dim=0)
        all_task_id = torch.concat([
            batch["task_id"],
            batch["task_id"]
        ], dim=0)

        # train the optimal networks
        with torch.no_grad():
            all_reward_target = self.target_network.reward(torch.concat([all_obs, all_action], dim=-1))
            all_optimal_target = self.get_optimal_values(
                self.target_network.optimal,
                all_next_obs,
                all_task_id
            )
            all_target = all_optimal_target.min(0)[0]
            all_target = all_reward_target + self.discount * all_target # CHECK: default no terminal
        all_optimal_pred = self.get_optimal_values(
            self.network.optimal,
            all_obs,
            all_task_id
        )
        optimal_loss = expectile_regression(all_optimal_pred.unsqueeze(0), all_target, expectile=self.expectile)
        optimal_loss = optimal_loss.sum(0).mean()

        self.optim["optimal"].zero_grad()
        optimal_loss.backward()
        self.optim["optimal"].step()

        # train the reward networks
        all_reward = self.network.reward(torch.concat([all_obs, all_action], dim=-1))
        all_adv = all_reward + self.discount * all_target - all_optimal_pred.detach()
        adv1, adv2 = torch.chunk(all_adv, 2, dim=0)
        logits = adv2.sum(dim=1) - adv1.sum(dim=1)
        labels = batch["label"].float()
        reward_loss = self.reward_criterion(logits, labels).mean()
        reg_loss = (all_reward**2).mean()
        with torch.no_grad():
            reward_accuracy = ((logits > 0) == torch.round(labels)).float()

        self.optim["reward"].zero_grad()
        (reward_loss + self.reward_reg * reg_loss).backward()
        self.optim["reward"].step()

        metrics = {
            "loss/reward_loss": reward_loss.item(),
            "loss/reward_reg_loss": reg_loss.item(),
            "loss/optimal_loss": optimal_loss.item(),
            "misc/reward_acc": reward_accuracy.item(),
            "misc/reward_value": all_reward.mean().item(),
            "misc/optimal_value": all_optimal_pred.mean().item()
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
        for attr in ["reward"]:
            state_dict = torch.load(os.path.join(path, attr+".pt"), map_location=self.device)
            self.network.__getattr__(attr).load_state_dict(state_dict)

    def save_pretrain(self, path):
        os.makedirs(path, exist_ok=True)
        for attr in ["reward"]:
            state_dict = self.network.__getattr__(attr).state_dict()
            torch.save(state_dict, os.path.join(path, attr+".pt"))
