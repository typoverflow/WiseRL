import itertools
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.base import Algorithm
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.utils.functional import expectile_regression
from wiserl.utils.misc import make_target, sync_target


class BehavioralCloningOAO(Algorithm):

    def setup_network(self, network_kwargs):
        network = {}
        network["actor"] = vars(wiserl.module)[network_kwargs["actor"].pop("class")](
            input_dim=self.observation_space.shape[0] + self.action_space.shape[0],
            output_dim=self.observation_space.shape[0],
            **network_kwargs["actor"]
        )
        if "encoder" in network_kwargs:
            network["encoder"] = vars(wiserl.module)[network_kwargs["encoder"].pop("class")](
                input_dim=self.observation_space.shape[0] + self.action_space.shape[0],
                output_dim=1,
                **network_kwargs["encoder"]
            )
        else:
            network["encoder"] = nn.Identity()
        self.network = nn.ModuleDict(network)

    def setup_optimizers(self, optim_kwargs):
        self.optim = {}
        default_kwargs = optim_kwargs.get("default", {})

        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(optim_kwargs.get("actor", {}))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = vars(torch.optim)[actor_kwargs.pop("class")](actor_params, **actor_kwargs)

    def select_action(self, batch, deterministic: bool=True):
        obs = self.network.encoder(batch["obs"])
        action, *_ = self.network.actor.sample(obs, deterministic=deterministic)
        return action.squeeze().cpu().numpy()

    def actor_loss(self, encoded_obs, action):
        if isinstance(self.network.actor, DeterministicActor):
            actor_loss = torch.sum((self.network.actor.sample(encoded_obs)[0] - action)**2, dim=-1, keepdim=True)
        elif isinstance(self.network.actor, GaussianActor):
            actor_loss = - self.network.actor.evaluate(encoded_obs, action)[0]
        return actor_loss.mean()

    def train_step(self, batches, step:int, total_steps: int):
        batch, *_ = batches
        obs = batch["obs"]  # (B, S)
        action = batch["action"] # (B, S)
        obs_action = torch.cat([obs, action], dim=-1)
        next_obs = batch["next_obs"]

        encoded_obs_action = self.network.encoder(obs_action)

        # compute the loss for actor
        actor_loss = self.actor_loss(encoded_obs_action, next_obs)
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()

        for _, scheduler in self.schedulers.items():
            scheduler.step()

        metrics = {
            "loss/actor_loss": actor_loss.item(),
        }
        return metrics
