import itertools
import os
from operator import itemgetter
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.oracle_awac import OracleAWAC
from wiserl.utils.misc import make_target, sync_target


class BTAWAC(OracleAWAC):
    def __init__(
        self,
        *args,
        beta: float = 0.3333,
        max_exp_clip: float = 100.0,
        discount: float = 0.99,
        tau: float = 0.005,
        target_freq: int = 1,
        reward_reg: float = 0.0,
        rm_label: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            *args,
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
        reward = self.network.reward(torch.concat([obs, action], dim=-1))
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

        # compute the loss for actor
        actor_loss, advantage = self.actor_loss(obs, action)
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()

        # compute the loss for q, offset by 1
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
            "loss/actor_loss": actor_loss.item(),
            "misc/q_pred": q_pred.mean().item(),
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
