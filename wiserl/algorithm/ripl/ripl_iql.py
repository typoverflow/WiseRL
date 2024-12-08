import itertools
from typing import Any, Dict, Optional, Tuple, Type

import gym
import imageio
import numpy as np
import torch
import torch.nn as nn

import wiserl
from wiserl.algorithm.base import Algorithm
from wiserl.algorithm.oracle_iql import OracleIQL
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target


class RIPL_IQL(OracleIQL):
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
        logstd_coeff: float = 0.1,
        logstd_threshold: float = -0.5,
        logstd_max: float = 2.0,
        v_logstd_coeff: float = 1.0,
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
        self.logstd_coeff = logstd_coeff
        self.logstd_threshold = logstd_threshold
        self.logstd_max = logstd_max
        self.v_logstd_coeff = v_logstd_coeff
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.threshold_criterion = torch.relu

    def setup_network(self, network_kwargs):
        network = {}
        network["actor"] = vars(wiserl.module)[network_kwargs["actor"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=self.action_space.shape[0],
            **network_kwargs["actor"]
        )
        network["critic"] = vars(wiserl.module)[network_kwargs["critic"].pop("class")](
            input_dim=self.observation_space.shape[0]+self.action_space.shape[0],
            output_dim=2,  # Output mean and logstd
            **network_kwargs["critic"]
        )
        network["value"] = vars(wiserl.module)[network_kwargs["value"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=2,  # Output mean and logstd
            **network_kwargs["value"]
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
            "critic": make_target(self.network.critic)
        })

    def setup_optimizers(self, optim_kwargs):
        super().setup_optimizers(optim_kwargs)

    def select_action(self, batch, deterministic: bool=True):
        return super().select_action(batch, deterministic)

    def v_loss(self, encoded_obs, q_old, weights=None, reduce=True):
        v_pred_mean_logstd = self.network.value(encoded_obs)
        v_pred, _ = torch.chunk(v_pred_mean_logstd, 2, dim=-1)
        v_loss = expectile_regression(v_pred, q_old, expectile=self.expectile)
        if weights is not None:
            v_loss = v_loss * weights
        return v_loss.mean() if reduce else v_loss, v_pred

    def train_step(self, batches, step: int, total_steps: int):
        if len(batches) > 1:
            feedback_batch, replay_batch, *_ = batches
        else:
            feedback_batch, replay_batch = batches[0], None
        using_replay_batch = replay_batch is not None

        F_B, F_S = feedback_batch["obs_1"].shape[0:2]
        F_S -= 1
        if using_replay_batch:
            if len(replay_batch["obs"].shape) == 2:
                R_B = replay_batch["obs"].shape[0]
            else:
                R_B = replay_batch["obs"].shape[0] * replay_batch["obs"].shape[1]
        else:
            R_B = 0
        split = [F_B*F_S, F_B*F_S, R_B]

        obs = torch.concat([
            feedback_batch["obs_1"][:, :-1].reshape(F_B*F_S, -1),
            feedback_batch["obs_2"][:, :-1].reshape(F_B*F_S, -1),
            *((replay_batch["obs"].reshape(R_B, -1), ) if using_replay_batch else ())
        ], dim=0)
        next_obs = torch.concat([
            feedback_batch["obs_1"][:, 1:].reshape(F_B*F_S, -1),
            feedback_batch["obs_2"][:, 1:].reshape(F_B*F_S, -1),
            *((replay_batch["next_obs"].reshape(R_B, -1), ) if using_replay_batch else ())
        ], dim=0)
        action = torch.concat([
            feedback_batch["action_1"][:, :-1].reshape(F_B*F_S, -1),
            feedback_batch["action_2"][:, :-1].reshape(F_B*F_S, -1),
            *((replay_batch["action"].reshape(R_B, -1), ) if using_replay_batch else ())
        ], dim=0)
        terminal = torch.concat([
            feedback_batch["terminal_1"][:, :-1].reshape(F_B*F_S, -1),
            feedback_batch["terminal_2"][:, :-1].reshape(F_B*F_S, -1),
            *((replay_batch["terminal"].reshape(R_B, -1), ) if using_replay_batch else ())
        ], dim=0)

        # encode every thing
        encoded_obs = self.network.encoder(obs)
        encoded_next_obs = self.network.encoder(next_obs).detach()

        # compute value loss
        with torch.no_grad():
            self.target_network.eval()
            q_old_mean_logstd = self.target_network.critic(encoded_obs, action)
            q_old, _ = torch.chunk(q_old_mean_logstd, 2, dim=-1)
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
        q_pred_mean_logstd = self.network.critic(encoded_obs.detach(), action)
        q_pred, q_pred_logstd = torch.chunk(q_pred_mean_logstd, 2, dim=-1)
        q_pred_logstd = torch.clamp(q_pred_logstd, max=self.logstd_max)
        with torch.no_grad():
            next_v_mean_logstd = self.network.value(encoded_next_obs)
            next_v, next_v_logstd = torch.chunk(next_v_mean_logstd, 2, dim=-1)
            next_v_logstd = torch.clamp(next_v_logstd, max=self.logstd_max)
        q1_logstd, q2_logstd, _ = torch.split(q_pred_logstd, split, dim=1)
        q1_std, q2_std = torch.exp(q1_logstd), torch.exp(q2_logstd)
        next_v1_logstd, next_v2_logstd, _ = torch.split(next_v_logstd.unsqueeze(0), split, dim=1)
        next_v1_std, next_v2_std = torch.exp(next_v1_logstd), torch.exp(next_v2_logstd)
        terminal_1, terminal_2, _ = torch.split(terminal, split, dim=0)
        reward = q_pred - (1-terminal) * self.discount * next_v.unsqueeze(0)
        r1, r2, rr = torch.split(reward, split, dim=1)
        E = r1.shape[0]
        r1, r2 = r1.reshape(E, F_B, F_S), r2.reshape(E, F_B, F_S)
        q1_std, q2_std = q1_std.reshape(E, F_B, F_S), q2_std.reshape(E, F_B, F_S)
        next_v1_std, next_v2_std = next_v1_std.reshape(-1, F_B, F_S), next_v2_std.reshape(-1, F_B, F_S)
        terminal_1, terminal_2 = terminal_1.reshape(-1, F_B, F_S), terminal_2.reshape(-1, F_B, F_S)
        pref_std = torch.sqrt((q1_std**2).sum(dim=-1) + (q2_std**2).sum(dim=-1) + \
            self.v_logstd_coeff * (((1-terminal_1) * self.discount)**2 * next_v1_std**2).sum(dim=-1) + \
            self.v_logstd_coeff * (((1-terminal_2) * self.discount)**2 * next_v2_std**2).sum(dim=-1))
        logits = (r2.sum(dim=-1) - r1.sum(dim=-1)) / (pref_std + 1e-8)
        labels = feedback_batch["label"].float().unsqueeze(0).squeeze(-1).expand(E, -1)
        q_loss = self.reward_criterion(logits, labels).mean()
        all_pref_logstd = torch.cat([q1_logstd, q2_logstd, next_v1_logstd, next_v2_logstd], dim=0)
        logstd_loss = self.threshold_criterion(self.logstd_threshold - all_pref_logstd).mean()
        if using_replay_batch and self.reg_replay_weight is not None:
            reg_loss_fb = (r1.square().mean() + r2.square().mean()) / 2
            reg_loss_re = rr.square().mean()
            reg_loss = (1-self.reg_replay_weight) * reg_loss_fb + self.reg_replay_weight * reg_loss_re
        else:
            reg_loss = reward.square().mean()
        self.optim["critic"].zero_grad()
        (q_loss + self.reward_reg * reg_loss + self.logstd_coeff * logstd_loss).backward()
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
            "loss/logstd_loss": logstd_loss.item(),
            "misc/q_pred": q_pred.mean().item(),
            "misc/v_pred": v_pred.mean().item(),
            "misc/advantage": advantage.mean().item(),
            "misc/reward_value": reward.mean().item(),
            "misc/reward_acc": reward_accuracy.mean().item(),
            "misc/reward_std": reward.std().item(),
            "misc/pref_std": pref_std.mean().item(),
            "misc/pref_std_std": pref_std.std().item(),
            "misc/q_pred_logstd": q_pred_logstd.mean().item(),
            "misc/q_pred_logstd_std": q_pred_logstd.std().item(),
            "misc/q_pred_std": q_pred_logstd.exp().mean().item(),
            "misc/next_v_logstd": next_v_logstd.mean().item(),
            "misc/next_v_std": next_v_logstd.exp().mean().item(),
            "misc/next_v_logstd_std": next_v_logstd.std().item(),
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
