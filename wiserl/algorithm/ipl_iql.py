import itertools
from typing import Any, Dict, Optional, Tuple, Type

import gym
import imageio
import numpy as np
import torch

from wiserl.algorithm.base import Algorithm
from wiserl.algorithm.oracle_iql import OracleIQL
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target


class IPL_IQL(OracleIQL):
    def __init__(
        self,
        *args,
        expectile: float = 0.7,
        beta: float = 0.3333,
        reward_reg: float = 0.5,
        reg_replay_weight: Optional[float] = None,
        actor_replay_weight: Optional[float] = None,
        value_replay_weight: Optional[float] = None,
        max_exp_clip: float = 100.0,
        discount: float = 0.99,
        tau: float = 0.005,
        target_freq: int = 1,
        **kwargs
    ):
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
        self.reg_replay_weight = reg_replay_weight
        self.actor_replay_weight = actor_replay_weight
        self.value_replay_weight = value_replay_weight
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def setup_network(self, network_kwargs):
        super().setup_network(network_kwargs)

    def setup_optimizers(self, optim_kwargs):
        super().setup_optimizers(optim_kwargs)

    def select_action(self, batch, deterministic: bool=True):
        return super().select_action(batch, deterministic)

    def train_step(self, batches, step: int, total_steps: int):
        if len(batches) > 1:
            feedback_batch, replay_batch, *_ = batches
        else:
            feedback_batch, replay_batch = batches[0], None
        using_replay_batch = replay_batch is not None

        F_B, F_S = feedback_batch["obs_1"].shape[0:2]
        F_S -= 1
        R_B = replay_batch["obs"].shape[0] if using_replay_batch else 0
        split = [F_B*F_S, F_B*F_S, R_B]

        obs = torch.concat([
            feedback_batch["obs_1"][:, :-1].reshape(F_B*F_S, -1),
            feedback_batch["obs_2"][:, :-1].reshape(F_B*F_S, -1),
            *((replay_batch["obs"], ) if using_replay_batch else ())
        ], dim=0)
        next_obs = torch.concat([
            feedback_batch["obs_1"][:, 1:].reshape(F_B*F_S, -1),
            feedback_batch["obs_2"][:, 1:].reshape(F_B*F_S, -1),
            *((replay_batch["next_obs"], ) if using_replay_batch else ())
        ], dim=0)
        action = torch.concat([
            feedback_batch["action_1"][:, :-1].reshape(F_B*F_S, -1),
            feedback_batch["action_2"][:, :-1].reshape(F_B*F_S, -1),
            *((replay_batch["action"], ) if using_replay_batch else ())
        ], dim=0)
        terminal = torch.concat([
            feedback_batch["terminal_1"][:, :-1].reshape(F_B*F_S, -1),
            feedback_batch["terminal_2"][:, :-1].reshape(F_B*F_S, -1),
            *((replay_batch["terminal"], ) if using_replay_batch else ())
        ], dim=0)

        # encode every thing
        encoded_obs = self.network.encoder(obs)
        encoded_next_obs = self.network.encoder(next_obs).detach()

        # compute value loss
        with torch.no_grad():
            self.target_network.eval()
            q_old = self.target_network.critic(encoded_obs, action)
            q_old = torch.min(q_old, dim=0)[0]
        v_loss, v_pred = self.v_loss(encoded_obs.detach(), q_old, reduce=False)
        if using_replay_batch and self.value_replay_weight is not None:
            v1, v2, vr = torch.split(v_loss, split, dim=0)
            v_loss_fb = (v1.mean() + v2.mean()) / 2
            v_loss_re = vr.mean()
            v_loss = (1-self.value_replay_weight) * v_loss_fb + self.value_replay_weight * v_loss_re
        else:
            v_loss = v_loss.mean()
        self.optim["value"].zero_grad()
        v_loss.backward()
        self.optim["value"].step()

        # compute actor loss
        actor_loss, advantage = self.actor_loss(encoded_obs, action, q_old, v_pred.detach(), reduce=False)
        if using_replay_batch and self.actor_replay_weight is not None:
            a1, a2, ar = torch.split(actor_loss, split, dim=0)
            actor_loss_fb = (a1.mean() + a2.mean()) / 2
            actor_loss_re = ar.mean()
            actor_loss = (1-self.actor_replay_weight) * actor_loss_fb + self.actor_replay_weight * actor_loss_re
        else:
            actor_loss = actor_loss.mean()
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()

        # compute the critic loss
        q_pred = self.network.critic(encoded_obs.detach(), action)
        with torch.no_grad():
            next_v = self.network.value(encoded_next_obs)
        reward = q_pred - (1-terminal) * self.discount * next_v.unsqueeze(0)
        r1, r2, rr = torch.split(reward, split, dim=1)
        E = r1.shape[0]
        r1, r2 = r1.reshape(E, F_B, F_S), r2.reshape(E, F_B, F_S)
        logits = r2.sum(dim=-1) - r1.sum(dim=-1)
        labels = feedback_batch["label"].float().unsqueeze(0).squeeze(-1).expand(E, -1)
        q_loss = self.reward_criterion(logits, labels).mean()
        if using_replay_batch and self.reg_replay_weight is not None:
            reg_loss_fb = (r1.square().mean() + r2.square().mean()) / 2
            reg_loss_re = rr.square().mean()
            reg_loss = (1-self.reg_replay_weight) * reg_loss_fb + self.reg_replay_weight * reg_loss_re
        else:
            reg_loss = reward.square().mean()
        self.optim["critic"].zero_grad()
        (q_loss + self.reward_reg * reg_loss).backward()
        self.optim["critic"].step()

        with torch.no_grad():
            reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()

        for _, scheduler in self.schedulers.items():
            scheduler.step()

        if step % self.target_freq == 0:
            sync_target(self.network.critic, self.target_network.critic, tau=self.tau)

        metrics = {
            "loss/q_loss": q_loss.item(),
            "loss/v_loss": v_loss.item(),
            "loss/actor_loss": actor_loss.item(),
            "loss/reg_loss": reg_loss.item(),
            "misc/q_pred": q_pred.mean().item(),
            "misc/v_pred": v_pred.mean().item(),
            "misc/advantage": advantage.mean().item(),
            "misc/reward_value": reward.mean().item(),
            "misc/reward_acc": reward_accuracy.mean().item()
        }
        if using_replay_batch and self.actor_replay_weight is not None:
            metrics.update({
                "detail/actor_loss_fb": actor_loss_fb.item(),
                "detail/actor_loss_re": actor_loss_re.item(),
            })
        if using_replay_batch and self.value_replay_weight is not None:
            metrics.update({
                "detail/v_loss_fb": v_loss_fb.item(),
                "detail/v_loss_re": v_loss_re.item(),
            })
        if using_replay_batch and self.reg_replay_weight is not None:
            metrics.update({
                "detail/reg_loss_fb": reg_loss_fb.item(),
                "detail/reg_loss_re": reg_loss_re.item()
            })
        return metrics
