import itertools
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.base import Algorithm
from wiserl.module.actor import DeterministicActor, GaussianActor


class SFT(Algorithm):
    def __init__(
        self,
        *args,
        bc_data: str = "win",  # ["win", "lose", "all"]
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert bc_data in {"win", "lose", "total"}, "SFT: bc_data should be in [win, loss, all]."
        self.bc_data = bc_data

    def setup_network(self, network_kwargs):
        network = {}
        if "encoder" in network_kwargs:
            network["encoder"] = vars(wiserl.module)[network_kwargs["encoder"]["class"]](
                input_dim=self.observation_space.shape[0],
                output_dim=1,
                **network_kwargs["encoder"]["kwargs"]
            )
        else:
            network["encoder"] = nn.Identity()
        network["actor"] = vars(wiserl.module)[network_kwargs["actor"]["class"]](
            input_dim=self.observation_space.shape[0],
            output_dim=self.action_space.shape[0],
            **network_kwargs["actor"]["kwargs"]
        )
        self.network = nn.ModuleDict(network)

    def setup_optimizers(self, optim_kwargs) -> None:
        self.optim = {}
        if "default" in optim_kwargs:
            default_class = optim_kwargs["default"]["class"]
            default_kwargs = optim_kwargs["default"]["kwargs"]
        else:
            default_class, default_kwargs = None, {}
        actor_class = optim_kwargs.get("actor", {}).get("kwargs", None) or default_class
        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(optim_kwargs.get("actor", {}).get("kwargs", {}))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = vars(torch.optim)[actor_class](actor_params, **actor_kwargs)

    def select_action(self, batch, deterministic: bool=True):
        obs = batch["obs"]
        action, *_ = self.network.actor.sample(obs, deterministic=deterministic)
        return action.squeeze().cpu().numpy()

    def train_step(self, batches, step: int, total_steps: bool=True):
        batch, *_ = batches
        obs = torch.stack([batch["obs_1"], batch["obs_2"]], dim=0)
        action = torch.stack([batch["action_1"], batch["action_2"]], dim=0)
        label = batch["label"]

        if self.bc_data == "total":
            mask = torch.stack([torch.ones_like(label), torch.ones_like(label)], dim=0)
        elif self.bc_data == "win":
            mask = torch.stack([1-label, label], dim=0)
            # select = (label > 0.5).long()
            # obs, action = obs[select], action[select]
        elif self.bc_data == "lose":
            mask = torch.stack([label, 1-label], dim=0)
            # select = (label < 0.5).long()
            # obs, action = obs[select], action[select]

        if isinstance(self.network.actor, DeterministicActor):
            actor_loss = ((self.network.actor.sample(obs)[0]-action)**2).sum(dim=-1)
        elif isinstance(self.network.actor, GaussianActor):
            actor_loss = - (self.network.actor.evaluate(obs, action)[0])
        actor_loss = (actor_loss * mask.unsqueeze(-1)).sum() / mask.sum()
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()

        for _, scheduler in self.schedulers.items():
            scheduler.step()

        metrics = {
            "loss/actor_loss": actor_loss.item()
        }
        return metrics
