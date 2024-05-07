import itertools
from operator import itemgetter
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.oracle_iql import OracleIQL
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.module.net.attention.preference_transformer import PreferenceTransformer
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target
from wiserl.utils.optim import LinearWarmupCosineAnnealingLR


class PTIQL(OracleIQL):
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
        use_weighted_sum: bool = True,
        max_seq_len: int,
        rm_label: bool = True,
        **kwargs
    ) -> None:
        self.use_weighted_sum = use_weighted_sum
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
        self.rm_label = rm_label
        assert rm_label
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def setup_network(self, network_kwargs):
        super().setup_network(network_kwargs)
        reward_kwargs = network_kwargs["reward"]
        reward = PreferenceTransformer(
            obs_dim=self.observation_space.shape[0],
            action_dim=self.action_space.shape[0],
            embed_dim=reward_kwargs["embed_dim"],
            pref_embed_dim=reward_kwargs["pref_embed_dim"],
            num_layers=reward_kwargs["num_layers"],
            seq_len=self.max_seq_len,
            num_heads=reward_kwargs["num_heads"],
            reward_act=reward_kwargs["reward_act"],
            use_weighted_sum=self.use_weighted_sum
        )
        self.network["reward"] = reward

    def setup_optimizers(self, optim_kwargs):
        super().setup_optimizers(optim_kwargs)
        default_kwargs = optim_kwargs.get("default", {})
        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(optim_kwargs.get("reward", {}))
        self.optim["reward"] = vars(torch.optim)[reward_kwargs.pop("class")](
            self.network.reward.parameters(), **reward_kwargs
        )

    def setup_schedulers(self, scheduler_kwargs):
        reward_kwargs = scheduler_kwargs.pop("reward")
        self.schedulers["reward"] = LinearWarmupCosineAnnealingLR(
            self.optim["reward"],
            warmup_epochs=reward_kwargs["warmup_steps"],
            max_epochs=reward_kwargs["max_steps"],
        )

        return super().setup_schedulers(scheduler_kwargs)

    def select_action(self, batch, deterministic: bool=True):
        return super().select_action(batch, deterministic)

    def select_reward(self, batch, deterministic=False):
        reward = []
        obs, action = batch["obs"], batch["action"]
        if "mask" in batch:
            mask = ~batch["mask"].to(torch.bool)
        else:
            mask = None
        tlen = obs.shape[1]
        for i_seg in range((tlen - 1) // self.max_seq_len + 1):
            obs_seg = obs[:, i_seg*self.max_seq_len:min((i_seg+1)*self.max_seq_len, tlen)]
            action_seg = action[:, i_seg*self.max_seq_len:min((i_seg+1)*self.max_seq_len, tlen)]
            if mask is not None:
                mask_seg = mask[:, i_seg*self.max_seq_len:min((i_seg+1)*self.max_seq_len, tlen), 0]
            else:
                mask_seg = None
            reward_seg, _ = self.network.reward(
                states=obs_seg,
                actions=action_seg,
                timesteps=None,
                key_padding_mask=mask_seg
            )
            reward.append(reward_seg)
        reward = torch.concat(reward, dim=1).detach()
        reward = torch.nan_to_num(reward, nan=0.0)  # they will be masked anyway
        return reward

    def pretrain_step(self, batches, step: int, total_steps: int) -> Dict:
        batch = batches[0]
        F_B, F_S = batch["obs_1"].shape[0:2]
        all_obs = torch.concat([batch["obs_1"], batch["obs_2"]], dim=0)
        all_action = torch.concat([batch["action_1"], batch["action_2"]], dim=0)
        self.network.reward.train()
        all_reward, all_sum = self.network.reward(
            states=all_obs,
            actions=all_action,
            timesteps=None
        )

        # reduce using the sum
        all_sum = all_sum.mean(dim=1)
        sum1, sum2 = torch.chunk(all_sum, 2, dim=0)
        logits = sum2 - sum1
        labels = batch["label"].float()
        reward_loss = self.reward_criterion(logits, labels).mean()
        reg_loss = (sum1**2).mean() + (sum2**2).mean()
        with torch.no_grad():
            reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()
        self.optim["reward"].zero_grad()
        (reward_loss + self.reward_reg * reg_loss).backward()
        self.optim["reward"].step()
        self.schedulers["reward"].step()

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
        reward = itemgetter("reward")(rl_batch)

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
