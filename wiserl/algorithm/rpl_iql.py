import itertools
from typing import Any, Dict, Optional, Tuple, Type

import gym
import imageio
import numpy as np
import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.base import Algorithm
from wiserl.algorithm.oracle_iql import OracleIQL
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target


class RPL_IQL(OracleIQL):
    def __init__(
        self,
        *args,
        expectile: float = 0.7,
        beta: float = 0.5,
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
        network = {}
        network["actor"] = vars(wiserl.module)[network_kwargs["actor"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=self.action_space.shape[0],
            **network_kwargs["actor"]
        )
        network["value"] = vars(wiserl.module)[network_kwargs["value"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=1,
            **network_kwargs["value"]
        )
        network["reward"] = vars(wiserl.module)[network_kwargs["reward"].pop("class")](
            input_dim=self.observation_space.shape[0]+self.action_space.shape[0],
            output_dim=1,
            **network_kwargs["reward"]
        )
        if "encoder" in network_kwargs:
            network["encoder"] = vars(wiserl.module)[network_kwargs["encoder"].pop("class")](
                input_dim=self.observation_space.shape[0],
                output_dim=1,
                **network_kwargs["encoder"]
            )
        else:
            network["encoder"] = nn.Identity()
        self.network = nn.ModuleDict(network)
        self.target_network = nn.ModuleDict({
            "reward": make_target(self.network.reward),
            "value": make_target(self.network.value)
        })

    def setup_optimizers(self, optim_kwargs):
        self.optim = {}
        default_kwargs = optim_kwargs.get("default", {})

        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(optim_kwargs.get("actor", {}))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = vars(torch.optim)[actor_kwargs.pop("class")](actor_params, **actor_kwargs)

        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(optim_kwargs.get("reward", {}))
        self.optim["reward"] = vars(torch.optim)[reward_kwargs.pop("class")](self.network.reward.parameters(), **reward_kwargs)

        value_kwargs = default_kwargs.copy()
        value_kwargs.update(optim_kwargs.get("value", {}))
        self.optim["value"] = vars(torch.optim)[value_kwargs.pop("class")](self.network.value.parameters(), **value_kwargs)

    def select_action(self, batch, deterministic: bool = True):
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

        encoded_obs = self.network.encoder(obs)
        encoded_next_obs = self.network.encoder(next_obs)

        # compute value loss
        with torch.no_grad():
            self.target_network.eval()
            v_old = self.target_network.value(encoded_obs.detach())
            reward_old = self.network.reward(torch.concat([encoded_obs.detach(), action], dim=-1))
            next_v_old = self.target_network.value(encoded_next_obs.detach())
            q_old = reward_old + self.discount * (1-terminal) * next_v_old.min(0)[0]

        v_loss, v_pred = self.v_loss(encoded_obs.detach(), q_old, reduce=False)
        if using_replay_batch and self.value_replay_weight is not None:
            v1, v2, vr = torch.split(v_loss, split, dim=1)
            v_loss_fb = (v1.mean() + v2.mean()) / 2
            v_loss_re = vr.mean()
            v_loss = (1-self.value_replay_weight) * v_loss_fb + self.value_replay_weight * v_loss_re
        else:
            v_loss = v_loss.mean()
        self.optim["value"].zero_grad()
        v_loss.backward()
        self.optim["value"].step()

        # compute actor loss
        actor_loss, advantage = self.actor_loss(encoded_obs, action, q_old, v_old.mean(0), reduce=False)
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

        # compute the reward loss
        reward_pred = self.network.reward(torch.concat([encoded_obs.detach(), action], dim=-1))
        adv_pred = reward_pred + self.discount * (1-terminal) * next_v_old.mean(0) - v_old.mean(0)
        adv1, adv2, advr = torch.split(adv_pred, split, dim=0)
        # E = adv1.shape[0]
        adv1, adv2 = adv1.reshape(F_B, F_S), adv2.reshape(F_B, F_S)
        logits = adv2.sum(dim=-1) - adv1.sum(dim=-1)
        labels = feedback_batch["label"].float().squeeze(-1)
        reward_loss = self.reward_criterion(logits, labels).mean()
        if using_replay_batch and self.reg_replay_weight is not None:
            r1, r2, rr = torch.split(reward_pred, split, dim=0)
            reg_loss_fb = (r1.square().mean() + r2.square().mean()) / 2
            reg_loss_re = rr.square().mean()
            reg_loss = (1-self.reg_replay_weight) * reg_loss_fb + self.reg_replay_weight * reg_loss_re
        else:
            reg_loss = reward_pred.abs().mean()
        self.optim["reward"].zero_grad()
        (reward_loss + self.reward_reg * reg_loss).backward()
        self.optim["reward"].step()

        with torch.no_grad():
            reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()

        for _, scheduler in self.schedulers.items():
            scheduler.step()

        if step % self.target_freq == 0:
            sync_target(self.network.value, self.target_network.value, tau=self.tau)
            sync_target(self.network.reward, self.target_network.reward, tau=self.tau)

        metrics = {
            "loss/reward_loss": reward_loss.item(),
            "loss/v_loss": v_loss.item(),
            "loss/actor_loss": actor_loss.item(),
            "loss/reg_loss": reg_loss.item(),
            "misc/v_pred": v_pred.mean().item(),
            "misc/advantage": advantage.mean().item(),
            "misc/reward_value": reward_pred.mean().item(),
            "misc/reward_acc": reward_accuracy.mean().item(),
            "misc/residual_abs": (adv_pred - reward_pred).abs().mean().item()
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
