import itertools
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.base import Algorithm
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target


class OracleIQL(Algorithm):
    def __init__(
        self,
        *args,
        expectile: float = 0.7,
        beta: float = 0.3333,
        max_exp_clip: float = 100.0,
        discount: float = 0.99,
        tau: float = 0.005,
        target_freq: int = 1,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.expectile = expectile
        self.beta = beta
        self.max_exp_clip = max_exp_clip
        self.target_freq = target_freq
        self.discount = discount
        self.tau = tau

    def setup_network(self, network_kwargs):
        network = {}
        network["actor"] = vars(wiserl.module)[network_kwargs["actor"]["class"]](
            input_dim=self.observation_space.shape[0],
            output_dim=self.action_space.shape[0],
            **network_kwargs["actor"]["kwargs"]
        )
        network["critic"] = vars(wiserl.module)[network_kwargs["critic"]["class"]](
            input_dim=self.observation_space.shape[0]+self.action_space.shape[0],
            output_dim=1,
            **network_kwargs["critic"]["kwargs"]
        )
        network["value"] = vars(wiserl.module)[network_kwargs["value"]["class"]](
            input_dim=self.observation_space.shape[0],
            output_dim=1,
            **network_kwargs["value"]["kwargs"]
        )
        if "encoder" in network_kwargs:
            network["encoder"] = vars(wiserl.module)[network_kwargs["encoder"]["class"]](
                input_dim=self.observation_space.shape[0],
                output_dim=1,
                **network_kwargs["encoder"]["kwargs"]
            )
        else:
            network["encoder"] = nn.Identity()
        self.network = nn.ModuleDict(network)
        self.target_network = nn.ModuleDict({
            "critic": make_target(self.network.critic)
        })

    def setup_optimizers(self, optim_kwargs):
        self.optim = {}
        if "default" in optim_kwargs:
            default_class = optim_kwargs["default"]["class"]
            default_kwargs = optim_kwargs["default"]["kwargs"]
        else:
            default_class, default_kwargs = None, {}
        actor_class = optim_kwargs.get("actor", {}).get("class", None) or default_class
        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(optim_kwargs.get("actor", {}).get("kwargs", {}))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = vars(torch.optim)[actor_class](actor_params, **actor_kwargs)

        critic_class = optim_kwargs.get("critic", {}).get("class", None) or default_class
        critic_kwargs = default_kwargs.copy()
        critic_kwargs.update(optim_kwargs.get("critic", {}).get("kwargs", {}))
        self.optim["critic"] = vars(torch.optim)[critic_class](self.network.critic.parameters(), **critic_kwargs)

        value_class = optim_kwargs.get("value", {}).get("class", None) or default_class
        value_kwargs = default_kwargs.copy()
        value_kwargs.update(optim_kwargs.get("value", {}).get("kwargs", {}))
        self.optim["value"] = vars(torch.optim)[value_class](self.network.value.parameters(), **critic_kwargs)

    def select_action(self, batch, deterministic: bool=True):
        obs = batch["obs"]
        action, *_ = self.network.actor.sample(obs, deterministic=deterministic)
        return action.squeeze().cpu().numpy()

    def v_loss(self, encoded_obs, q_old):
        v_pred = self.network.value(encoded_obs.detach())
        v_loss = expectile_regression(v_pred, q_old, expectile=self.expectile).mean()
        return v_loss, v_pred

    def actor_loss(self, encoded_obs, action, q_old, v):
        with torch.no_grad():
            advantage = q_old - v
        exp_advantage = (advantage / self.beta).exp().clip(max=self.max_exp_clip)
        if isinstance(self.network.actor, DeterministicActor):
            policy_out = torch.sum((self.network.actor.sample(encoded_obs)[0] - action)**2, dim=-1)
        elif isinstance(self.network.actor, GaussianActor):
            policy_out = - self.network.actor.evaluate(encoded_obs, action)[0]
        actor_loss = (exp_advantage * policy_out.unsqueeze(-1)).mean()
        return actor_loss, advantage

    def q_loss(self, encoded_obs, action, next_encoded_obs, reward, terminal):
        with torch.no_grad():
            target_q = self.network.value(next_encoded_obs)
            target_q = reward + self.discount * (1-terminal) * target_q
        q_pred = self.network.critic(encoded_obs.detach(), action)
        q_loss = (q_pred - target_q.unsqueeze(0)).pow(2).sum(0).mean()
        return q_loss, q_pred

    def train_step(self, batches, step:int, total_steps: int):
        batch, *_ = batches
        obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (B, S+1)
        action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)
        reward = torch.cat([batch["reward_1"], batch["reward_2"]], dim=0)
        terminal = torch.cat([batch["terminal_1"], batch["terminal_2"]], dim=0)

        encoded_obs = self.network.encoder(obs)

        with torch.no_grad():
            q_old = self.target_network.critic(encoded_obs, action)
            q_old = torch.min(q_old, dim=0)[0]

        # compute the loss for value network
        v_loss, v_pred = self.v_loss(encoded_obs, q_old)
        self.optim["value"].zero_grad()
        v_loss.backward()
        self.optim["value"].step()

        # compute the loss for actor
        actor_loss, advantage = self.actor_loss(encoded_obs, action, q_old, v_pred.detach())
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
            "loss/v_loss": v_loss.item(),
            "loss/actor_loss": actor_loss.item(),
            "misc/q_pred": q_pred.mean().item(),
            "misc/v_pred": v_pred.mean().item(),
            "misc/advantage": advantage.mean().item()
        }
        return metrics
