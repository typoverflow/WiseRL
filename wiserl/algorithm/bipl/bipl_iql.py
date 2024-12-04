import itertools
from typing import Any, Dict, Optional, Tuple, Type

import gym
import imageio
import numpy as np
import torch

import wiserl.module
from wiserl.algorithm.base import Algorithm
from wiserl.algorithm.oracle_iql import OracleIQL
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target


class BIPL_IQL(OracleIQL):
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
        bc_steps: int = 0,
        bc_data: str = "total",
        min_importance_weight: float = 0.1,
        max_importance_weight: float = 10.0,
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
        self.bc_steps = bc_steps
        self.bc_data = bc_data
        self.min_importance_weight = min_importance_weight
        self.max_importance_weight = max_importance_weight

    def setup_network(self, network_kwargs):
        super().setup_network(network_kwargs)
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

    def train_step(self, batches, step: int, total_steps: int):
        if step <= self.bc_steps:
            return self.bc_step(batches, step, total_steps)
        else:
            return self.pref_train_step(batches, step, total_steps)
        
    def compute_logprob(self, actor, obs, action):
        if isinstance(actor, DeterministicActor):
            logprob = - torch.square(action - actor.sample(obs)[0]).sum(dim=-1, keepdim=True)
        elif isinstance(actor, GaussianActor):
            logprob = actor.evaluate(obs, action)[0]
        return logprob

    def bc_step(self, batches, step: int, total_steps: int):
        if len(batches) > 1:
            feedback_batch, replay_batch, *_ = batches
        else:
            feedback_batch, replay_batch = batches[0], None

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
        unlabeled_actor_loss = torch.tensor(0.0)
        if replay_batch is not None:
            unlabeled_obs = replay_batch["obs"].reshape(-1, self.observation_space.shape[0])
            unlabeled_action = replay_batch["action"].reshape(-1, self.action_space.shape[0])

            log_prob = self.network.unlabeled_actor.evaluate(unlabeled_obs, unlabeled_action)[0]
            unlabeled_actor_loss = -log_prob.mean()

            self.optim["unlabeled_actor"].zero_grad()
            unlabeled_actor_loss.backward()
            self.optim["unlabeled_actor"].step()
        else:
            unlabeled_actor_loss = torch.tensor(0.0)

        metrics = {
            "loss/pref_actor_loss": pref_bc_loss.item(),
            "loss/unlabeled_actor_loss": unlabeled_actor_loss.item(),
        }
        return metrics

    def pref_train_step(self, batches, step: int, total_steps: int):
        # Original training logic moved here
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
            q_old = self.target_network.critic(encoded_obs, action)
            q_old = torch.min(q_old, dim=0)[0]

        # Compute importance weights
        with torch.no_grad():
            log_pref_prob = self.network.pref_actor.evaluate(obs, action)[0].mean(dim=1)
            log_unlabeled_prob = self.network.unlabeled_actor.evaluate(obs, action)[0].mean(dim=1)
            importance_weight = torch.exp(log_pref_prob - log_unlabeled_prob)
            importance_weight = importance_weight.clamp(self.min_importance_weight, self.max_importance_weight)

        # Compute v_loss with importance weights
        v_loss, v_pred = self.v_loss(encoded_obs.detach(), q_old, weights=importance_weight, reduce=False)
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
        actor_loss, advantage = self.actor_loss(encoded_obs, action, q_old, v_pred.detach(), weights=importance_weight, reduce=False)
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
            "misc/reward_acc": reward_accuracy.mean().item(),
            "misc/importance_weight": importance_weight.mean().item(),
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
