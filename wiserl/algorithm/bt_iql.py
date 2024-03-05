import itertools
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.oracle_iql import OracleIQL
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target


class BTIQL(OracleIQL):
    def __init__(
        self,
        *args,
        expectile: float = 0.7,
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
            expectile=expectile,
            beta=beta,
            max_exp_clip=max_exp_clip,
            discount=discount,
            tau=tau,
            target_freq=target_freq,
            **kwargs
        )
        self.reward_steps = reward_steps
        self.reward_reg = reward_reg
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

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
        if len(batches) > 1:
            batch, replay_batch, *_ = batches
        else:
            batch, replay_batch = batches[0], None
        using_replay_batch = replay_batch is not None

        F_B, F_S = batch["obs_1"].shape[0:2]
        R_B = replay_batch["obs"].shape[0] if using_replay_batch else 0
        split1 = [F_B*F_S, F_B*F_S, R_B]

        all_obs = torch.concat([
            batch["obs_1"].reshape(-1, self.obs_dim),
            batch["obs_2"].reshape(-1, self.obs_dim),
            *((replay_batch["obs"], ) if using_replay_batch else ())
        ])
        all_action = torch.concat([
            batch["action_1"].reshape(-1, self.action_dim),
            batch["action_2"].reshape(-1, self.action_dim),
            *((replay_batch["action"], ) if using_replay_batch else ())
        ])
        all_obs_encoded = self.network.encoder(all_obs)

        if step < self.reward_steps:
            self.network.reward.train()
            all_reward = self.network.reward(torch.cat([all_obs, all_action], dim=-1))
            r1, r2, rr = torch.split(all_reward, split1, dim=1)
            E = r1.shape[0]
            r1, r2 = r1.reshape(E, F_B, F_S, 1), r2.reshape(E, F_B, F_S, 1)
            logits = r2.sum(dim=2) - r1.sum(dim=2)
            labels = batch["label"].float().unsqueeze(0).expand_as(logits)

            reward_loss = self.reward_criterion(logits, labels).mean()
            reg_loss = (r1**2).mean() + (r2**2).mean()
            with torch.no_grad():
                reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()

            self.optim["reward"].zero_grad()
            (reward_loss + self.reward_reg * reg_loss).backward()
            self.optim["reward"].step()

            all_reward = all_reward.detach().mean(dim=0)
            # E, B = all_reward.shape[:2]
            # select_idx = np.random.randint(0, E, size=B)
            # all_reward = all_reward.detach()[select_idx, np.arange(B)]

        else:
            all_reward = self.network.reward(torch.concat([all_obs, all_action], dim=-1)).detach().mean(dim=0)
            # all_reward = self.network.reward(torch.concat([all_obs, all_action], dim=-1))
            # E, B = all_reward.shape[:2]
            # select_idx = np.random.randint(0, E, size=B)
            # all_reward = all_reward.detach()[select_idx, np.arange(B)]
        # if using_replay_batch:
        #     replay_obs_encoded = self.network.encoder(replay_batch["obs"])
        #     replay_next_obs_encoded = self.network.encoder(replay_batch["next_obs"])
        # all_obs_encoded = torch.concat([
        #     feedback_obs_encoded.reshape(-1, self.obs_dim),
        #     *((replay_obs_encoded, ) if using_replay_batch else ())
        # ])
        # all_action = torch.concat([
        #     feedback_action.reshape(-1, self.action_dim),
        #     *((replay_batch["action"], ) if using_replay_batch else ())
        # ])
        # cur_obs_encoded = torch.concat([
        #     feedback_obs_encoded[:, :-1].reshape(-1, self.obs_dim)
        #     *((replay_obs_encoded, ) if using_replay_batch else ())
        # ], dim=0)
        # cur_action = torch.concat([
        #     feedback_action[:, :-1].reshape(-1, self.action_dim),
        #     *((replay_batch["action"], ) if using_replay_batch else ())
        # ], dim=0)
        # next_obs_encoded = torch.concat([
        #     feedback_obs_encoded[:, 1:].reshape(-1, self.obs_dim)
        #     *((replay_next_obs_encoded, ) if using_replay_batch else ())
        # ])
        # cur_terminal = torch.concat([
        #     batch["terminal_1"][:, :-1].reshape(F_B*F_S, -1),
        #     batch["terminal_2"][:, :-1].reshape(F_B*F_S, -1),
        #     *((replay_batch["terminal"], ) if using_replay_batch else ())
        # ], dim=0)
        # split1 = [F_B*F_S, F_B*F_S, R_B]
        # split2 = [F_B*(F_S-1), F_B*(F_S-1), R_B]

        with torch.no_grad():
            self.target_network.eval()
            q_old = self.target_network.critic(all_obs_encoded, all_action)
            q_old = torch.min(q_old, dim=0)[0]

        # compute the loss for value network
        v_loss, v_pred = self.v_loss(all_obs_encoded.detach(), q_old, reduce=False)
        if using_replay_batch:
            v1, v2, vr = torch.split(v_loss, split1, dim=0)
            v_loss_fb = (v1.mean() + v2.mean()) / 2
            v_loss_re = vr.mean()
            v_loss = (v_loss_fb + v_loss_re) / 2
        else:
            v_loss = v_loss.mean()
        self.optim["value"].zero_grad()
        v_loss.backward()
        self.optim["value"].step()

        # compute the loss for actor
        actor_loss, advantage = self.actor_loss(all_obs_encoded, all_action, q_old, v_pred.detach(), reduce=False)
        if using_replay_batch:
            a1, a2, ar = torch.split(actor_loss, split1, dim=0)
            actor_loss_fb = (a1.mean() + a2.mean()) / 2
            actor_loss_re = ar.mean()
            actor_loss = (actor_loss_fb + actor_loss_re) / 2
        else:
            actor_loss = actor_loss.mean()
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()

        # compute the loss for q, offset by 1
        with torch.no_grad():
            o1, o2, or_ = torch.split(all_obs_encoded, split1, dim=0)
            o1, o2 = o1.reshape(F_B, F_S, self.obs_dim), o2.reshape(F_B, F_S, self.obs_dim)
            onr = self.network.encoder(replay_batch["next_obs"]) if using_replay_batch else None

            all_next_obs_encoded = torch.concat([
                torch.concat([o1[:, 1:], torch.zeros_like(o1[:, [0]])], dim=1).reshape(-1, self.obs_dim),
                torch.concat([o2[:, 1:], torch.zeros_like(o2[:, [0]])], dim=1).reshape(-1, self.obs_dim),
                *((onr, ) if using_replay_batch else ())
            ], dim=0)
            all_action = torch.concat([
                batch["action_1"].reshape(-1, self.action_dim),
                batch["action_2"].reshape(-1, self.action_dim),
                *((replay_batch["action"], ) if using_replay_batch else ())
            ], dim=0)
            all_terminal = torch.concat([
                batch["terminal_1"].reshape(-1, 1),
                batch["terminal_2"].reshape(-1, 1),
                *((replay_batch["terminal"], ) if using_replay_batch else ())
            ], dim=0)
            td_mask = torch.ones([F_B, F_S, 1]).to(self.device)
            td_mask[:, -1] = 0
            td_mask = td_mask.reshape(-1, 1)
            all_mask = torch.concat([
                td_mask,
                td_mask,
                *((torch.ones([R_B, 1]).to(self.device), ) if using_replay_batch else ())
            ], dim=0)
        q_loss, q_pred = self.q_loss(
            all_obs_encoded.detach(),
            all_action,
            all_next_obs_encoded.detach(),
            all_reward,
            all_terminal,
            reduce=False
        )
        if using_replay_batch:
            q1, q2, qr = torch.split(q_loss, split1, dim=0)
            q_loss_fb = ((q1*td_mask).mean() + (q2*td_mask).mean()) / 2
            q_loss_re = qr.mean()
            q_loss = (q_loss_fb + q_loss_re) / 2
        else:
            q_loss = (q_loss*all_mask).mean()
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
        if step < self.reward_steps:
            metrics.update({
                "loss/reward_loss": reward_loss.item(),
                "loss/reward_reg_loss": reg_loss.item(),
                "misc/reward_acc": reward_accuracy.item(),
                "misc/reward_value": all_reward.mean().item()
            })
        if using_replay_batch:
            metrics.update({
                "detail/actor_loss_fb": actor_loss_fb.item(),
                "detail/actor_loss_re": actor_loss_re.item(),
                "detail/v_loss_fb": v_loss_fb.item(),
                "detail/v_loss_re": v_loss_re.item(),
                "detail/q_loss_fb": q_loss_fb.item(),
                "detail/q_loss_re": q_loss_re.item()
            })
        return metrics
