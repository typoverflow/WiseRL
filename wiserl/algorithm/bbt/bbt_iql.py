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


class BBTIQL(OracleIQL):
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
        bc_steps: int = 0,
        bc_data: str = "total",
        min_importance_weight: float = 0.1,
        max_importance_weight: float = 10.0,
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
        self.bc_steps = bc_steps
        self.bc_data = bc_data
        self.min_importance_weight = min_importance_weight
        self.max_importance_weight = max_importance_weight

        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def setup_network(self, network_kwargs):
        super().setup_network(network_kwargs)
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
        self.network["pref_actor"] = vars(wiserl.module)[network_kwargs["pref_actor"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=self.action_space.shape[0],
            **network_kwargs["pref_actor"]
        )
        self.network["unlabeled_actor"] = vars(wiserl.module)[network_kwargs["unlabeled_actor"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=self.action_space.shape[0],
            **network_kwargs["unlabeled_actor"]
        )



    def setup_optimizers(self, optim_kwargs):
        super().setup_optimizers(optim_kwargs)
        default_kwargs = optim_kwargs.get("default", {})
        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(optim_kwargs.get("reward", {}))
        self.optim["reward"] = vars(torch.optim)[reward_kwargs.pop("class")](self.network.reward.parameters(), **reward_kwargs)
        
        pref_actor_kwargs = default_kwargs.copy()
        pref_actor_kwargs.update(optim_kwargs.get("pref_actor", {}))
        self.optim["pref_actor"] = vars(torch.optim)[pref_actor_kwargs.pop("class")](
            self.network.pref_actor.parameters(), **pref_actor_kwargs
        )

        unlabeled_actor_kwargs = default_kwargs.copy()
        unlabeled_actor_kwargs.update(optim_kwargs.get("unlabeled_actor", {}))
        self.optim["unlabeled_actor"] = vars(torch.optim)[unlabeled_actor_kwargs.pop("class")](
            self.network.unlabeled_actor.parameters(), **unlabeled_actor_kwargs
        )

    def select_action(self, batch, deterministic: bool=True):
        return super().select_action(batch, deterministic)

    def select_reward(self, batch, deterministic=False):
        obs, action = batch["obs"], batch["action"]
        reward = self.network.reward(torch.concat([obs, action], dim=-1))
        return reward.mean(0).detach()

    def pretrain_step(self, batches, step: int, total_steps: int) -> Dict:
        if step <= self.bc_steps:
            return self.bc_step(batches, step, total_steps)
        else:
            return self.reward_step(batches, step, total_steps)
        
    def compute_logprob(self, actor, obs, action):
        if isinstance(actor, DeterministicActor):
            logprob = - torch.square(action - actor.sample(obs)[0]).sum(dim=-1, keepdim=True)
        elif isinstance(actor, GaussianActor):
            logprob = actor.evaluate(obs, action)[0]
        return logprob

    def bc_step(self, batches, step: int, total_steps: int):
        feedback_batch, replay_batch, *_ = batches

        # Combine observations and actions from both sequences
        pref_obs = torch.concat([feedback_batch["obs_1"][:, :-1], feedback_batch["obs_2"][:, :-1]], dim=0)
        pref_action = torch.concat([feedback_batch["action_1"][:, :-1], feedback_batch["action_2"][:, :-1]], dim=0)
        label = feedback_batch["label"].float()
        
        # Compute log probabilities for pref_actor
        logprob = self.compute_logprob(self.network.pref_actor, pref_obs, pref_action)

        # Create mask based on bc_data
        if self.bc_data == "total":
            mask = torch.concat([torch.ones_like(label), torch.ones_like(label)], dim=0)
        elif self.bc_data == "win":
            mask = torch.concat([1-label, label], dim=0)
        elif self.bc_data == "lose":
            mask = torch.concat([label, 1-label], dim=0)
        pref_bc_loss = -(logprob.mean(dim=1) * mask).sum() / mask.sum()

        # Update pref_actor
        self.optim["pref_actor"].zero_grad()
        pref_bc_loss.backward()
        self.optim["pref_actor"].step()

        # Behavior cloning for unlabeled_actor
        unlabeled_obs = replay_batch["obs"].reshape(-1, self.observation_space.shape[0])
        unlabeled_action = replay_batch["action"].reshape(-1, self.action_space.shape[0])

        log_prob = self.network.unlabeled_actor.evaluate(unlabeled_obs, unlabeled_action)[0]
        unlabeled_actor_loss = -log_prob.mean()

        self.optim["unlabeled_actor"].zero_grad()
        unlabeled_actor_loss.backward()
        self.optim["unlabeled_actor"].step()

        metrics = {
            "loss/pref_actor_loss": pref_bc_loss.item(),
            "loss/unlabeled_actor_loss": unlabeled_actor_loss.item(),
        }
        return metrics

    def reward_step(self, batches, step: int, total_steps: int) -> Dict:
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
        all_reward = self.network.reward(torch.concat([all_obs, all_action], dim=-1))
        r1, r2 = torch.chunk(all_reward, 2, dim=1)
        E = r1.shape[0]
        r1, r2 = r1.reshape(E, F_B, F_S, 1), r2.reshape(E, F_B, F_S, 1)
        logits = r2.sum(dim=2) - r1.sum(dim=2)
        labels = batch["label"].float().unsqueeze(0).expand_as(logits)
        reward_loss = self.reward_criterion(logits, labels).sum(0).mean()
        reg_loss = (r1**2).sum(0).mean() + (r2**2).sum(0).mean()
        with torch.no_grad():
            reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()

        self.optim["reward"].zero_grad()
        (reward_loss + self.reward_reg * reg_loss).backward()
        self.optim["reward"].step()

        metrics = {
            "loss/reward_loss": reward_loss.item(),
            "loss/reward_reg_loss": reg_loss.item(),
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

        # Compute importance weights
        with torch.no_grad():
            log_pref_prob = self.network.pref_actor.evaluate(obs, action)[0].mean(dim=1)
            log_unlabeled_prob = self.network.unlabeled_actor.evaluate(obs, action)[0].mean(dim=1)
            importance_weight = torch.exp(log_pref_prob - log_unlabeled_prob)
            importance_weight = importance_weight.clamp(self.min_importance_weight, self.max_importance_weight)

        # Compute v_loss with importance weights
        v_loss, v_pred = self.v_loss(obs.detach(), q_old, importance_weight=importance_weight)
        self.optim["value"].zero_grad()
        v_loss.backward()
        self.optim["value"].step()

        # compute the loss for actor
        actor_loss, advantage = self.actor_loss(obs, action, q_old, v_pred.detach(), importance_weight=importance_weight)
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
            "misc/importance_weight": importance_weight.mean().item(),
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
