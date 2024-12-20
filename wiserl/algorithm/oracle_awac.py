import itertools
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.base import Algorithm
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.utils.misc import make_target, sync_target


class OracleAWAC(Algorithm):
    def __init__(
        self,
        *args,
        beta: float = 0.3333,
        max_exp_clip: float = 100.0,
        discount: float = 0.99,
        tau: float = 0.005,
        target_freq: int = 1,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.max_exp_clip = max_exp_clip
        self.target_freq = target_freq
        self.discount = discount
        self.tau = tau

    def setup_network(self, network_kwargs):
        network = {}
        network["actor"] = vars(wiserl.module)[network_kwargs["actor"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=self.action_space.shape[0],
            **network_kwargs["actor"]
        )
        network["critic"] = vars(wiserl.module)[network_kwargs["critic"].pop("class")](
            input_dim=self.observation_space.shape[0]+self.action_space.shape[0],
            output_dim=1,
            **network_kwargs["critic"]
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
        self.optim = {}
        default_kwargs = optim_kwargs.get("default", {})

        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(optim_kwargs.get("actor", {}))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = vars(torch.optim)[actor_kwargs.pop("class")](actor_params, **actor_kwargs)

        critic_kwargs = default_kwargs.copy()
        critic_kwargs.update(optim_kwargs.get("critic", {}))
        self.optim["critic"] = vars(torch.optim)[critic_kwargs.pop("class")](self.network.critic.parameters(), **critic_kwargs)

    def select_action(self, batch, deterministic: bool=True):
        obs = self.network.encoder(batch["obs"])
        action, *_ = self.network.actor.sample(obs, deterministic=deterministic)
        return action.squeeze().cpu().numpy()

    def actor_loss(self, encoded_obs, action, reduce=True):
        with torch.no_grad():
            baseline_actions = self.network.actor.sample(encoded_obs)[0]
            v = self.network.critic(encoded_obs, baseline_actions).mean(0)
            q = self.network.critic(encoded_obs, action).mean(0)
            advantage = q - v
        exp_advantage = (advantage / self.beta).exp().clip(max=self.max_exp_clip)
        if isinstance(self.network.actor, DeterministicActor):
            policy_out = torch.sum((self.network.actor.sample(encoded_obs)[0] - action)**2, dim=-1, keepdim=True)
        elif isinstance(self.network.actor, GaussianActor):
            policy_out = - self.network.actor.evaluate(encoded_obs, action)[0]
        actor_loss = (exp_advantage * policy_out)
        return actor_loss.mean() if reduce else actor_loss, advantage

    def q_loss(self, encoded_obs, action, next_encoded_obs, reward, terminal, reduce=True):
        with torch.no_grad():
            self.target_network.eval()
            next_actions = self.network.actor.sample(next_encoded_obs)[0]
            target_q = self.target_network.critic(next_encoded_obs, next_actions).min(0)[0]
            target_q = reward + self.discount * (1-terminal) * target_q
        q_pred = self.network.critic(encoded_obs, action)
        q_loss = (q_pred - target_q.unsqueeze(0)).pow(2).sum(0)
        return q_loss.mean() if reduce else q_loss, q_pred

    def train_step(self, batches, step:int, total_steps: int):
        batch, *_ = batches
        if "obs_1" in batch:
            obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (B, S+1)
            action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)
            reward = torch.cat([batch["reward_1"], batch["reward_2"]], dim=0)
            terminal = torch.cat([batch["terminal_1"], batch["terminal_2"]], dim=0)

            encoded_obs = self.network.encoder(obs)

            q_loss, q_pred = self.q_loss(
                encoded_obs[:, :-1].detach(),
                action[:, :-1],
                encoded_obs[:, 1:].detach(),
                reward[:, :-1],
                terminal[:, :-1]
            )
        else:
            obs = batch["obs"]
            action = batch["action"]
            reward = batch["reward"]
            terminal = batch["terminal"].float()
            next_obs = batch["next_obs"]

            encoded_obs = self.network.encoder(obs)
            next_encoded_obs = self.network.encoder(next_obs)
            
            q_loss, q_pred = self.q_loss(encoded_obs, action, next_encoded_obs, reward, terminal)

        self.optim["critic"].zero_grad()
        q_loss.backward()
        self.optim["critic"].step()

        # compute the loss for actor
        actor_loss, advantage = self.actor_loss(encoded_obs, action)
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()

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
