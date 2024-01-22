import itertools
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
        reward_steps: Optional[int] = None,
        reward_reg: float = 0.0,
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
        self.reward_steps = reward_steps
        self.reward_reg = reward_reg

        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def setup_network(self, network_kwargs):
        super().setup_network(network_kwargs)
        self.network["reward"] = vars(wiserl.module)[network_kwargs["reward"]["class"]](
            input_dim=self.observation_space.shape[0]+self.action_space.shape[0],
            output_dim=1,
            **network_kwargs["reward"]["kwargs"]
        )

    def setup_optimizers(self, optim_kwargs):
        super().setup_optimizers(optim_kwargs)
        if "default" in optim_kwargs:
            default_class = optim_kwargs["default"]["class"]
            default_kwargs = optim_kwargs["default"]["kwargs"]
        else:
            default_class, default_kwargs = None, {}
        reward_class = optim_kwargs.get("reward", {}).get("class", None) or default_class
        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(optim_kwargs.get("reward", {}).get("kwargs", {}))
        self.optim["reward"] = vars(torch.optim)[reward_class](self.network.reward.parameters(), **reward_kwargs)

    def select_action(self, batch, deterministic: bool=True):
        return super().select_action(batch, deterministic)

    def train_step(self, batches, step: int, total_steps: int) -> Dict:
        batch, *_ = batches
        obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (B, S+1)
        action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)
        terminal = torch.cat([batch["terminal_1"], batch["terminal_2"]], dim=0)

        encoded_obs = self.network.encoder(obs)

        if step < self.reward_steps:
            self.network.reward.train()
            reward = self.network.reward(torch.cat([obs, action], dim=-1))
            r1, r2 = torch.chunk(reward.sum(dim=2), 2, dim=1)
            logits = r2 - r1
            labels = batch["label"].float().unsqueeze(0).expand_as(logits)

            reward_loss = self.reward_criterion(logits, labels).mean()
            reg_loss = (r1**2).mean() + (r2**2).mean()
            with torch.no_grad():
                reward_accuracy = ((r2 > r1) == torch.round(labels)).float().mean()

            self.optim["reward"].zero_grad()
            (reward_loss + self.reward_reg * reg_loss).backward()
            self.optim["reward"].step()

            reward = reward.detach().mean(dim=0)
        else:
            reward = self.network.reward(torch.cat([obs, action], dim=-1)).detach().mean(dim=0)

        # compute the loss for actor
        actor_loss, advantage = self.actor_loss(encoded_obs, action)
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()

        # compute the loss for q, offset by 1
        q_loss, q_pred = self.q_loss(
            encoded_obs[:, :-1],
            action[:, :-1],
            encoded_obs[:, 1:],
            reward[:, :-1],
            terminal[:, :-1]
        )
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
        if step < self.reward_steps:
            metrics.update({
                "loss/reward_loss": reward_loss.item(),
                "loss/reward_reg_loss": reg_loss.item(),
                "misc/reward_acc": reward_accuracy.item(),
                "misc/reward_value": reward.mean().item()
            })
        return metrics
