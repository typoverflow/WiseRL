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


class MismatchedRPL(OracleIQL):
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
        network["reward"] = vars(wiserl.module)[network_kwargs["reward"].pop("class")](
            input_dim=self.observation_space.shape[0]+self.action_space.shape[0],
            output_dim=1,
            **network_kwargs["reward"]
        )
        network["pref_value"] = vars(wiserl.module)[network_kwargs["pref_value"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=1,
            **network_kwargs["pref_value"]
        )
        network["rep_value"] = vars(wiserl.module)[network_kwargs["rep_value"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=1,
            **network_kwargs["rep_value"]
        )
        self.network = nn.ModuleDict(network)
        self.target_network = nn.ModuleDict({
            "pref_value": make_target(self.network.pref_value),
            "rep_value": make_target(self.network.rep_value)
        })

    def setup_optimizers(self, optim_kwargs):
        self.optim = {}
        default_kwargs = optim_kwargs.get("default", {})

        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(optim_kwargs.get("actor", {}))
        self.optim["actor"] = vars(torch.optim)[actor_kwargs.pop("class")](self.network.actor.parameters(), **actor_kwargs)

        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(optim_kwargs.get("reward", {}))
        self.optim["reward"] = vars(torch.optim)[reward_kwargs.pop("class")](self.network.reward.parameters(), **reward_kwargs)

        value_kwargs = default_kwargs.copy()
        value_kwargs.update(optim_kwargs.get("pref_value", {}))
        self.optim["pref_value"] = vars(torch.optim)[value_kwargs.pop("class")](self.network.pref_value.parameters(), **value_kwargs)

        value_kwargs = default_kwargs.copy()
        value_kwargs.update(optim_kwargs.get("rep_value", {}))
        self.optim["rep_value"] = vars(torch.optim)[value_kwargs.pop("class")](self.network.rep_value.parameters(), **value_kwargs)

    def select_action(self, batch, deterministic: bool = True):
        return super().select_action(batch, deterministic)

    def train_step(self, batches, step: int, total_steps: int):
        pass
