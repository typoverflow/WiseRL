import itertools
from typing import Any, Dict, Optional, Type

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
        max_exp_clip: float = 0.005,
        discount: float = 0.99,
        tau: float = 0.005,
        target_freq: int = 1,
        reward_steps: Optional[int] = None,
        reward_reg: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(*args, expectile, beta, max_exp_clip, discount, tau, target_freq)
        self.reward_steps = reward_steps
        self.reward_reg = reward_reg

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
        super().select_action(batch, deterministic)

    def train_step(self, batches, step: int, total_steps: int) -> Dict:
        batch, *_ = batches
        obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (B, S+1)
        action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)

        if step < self.reward_steps:
            self.network.reward.train()
            reward = self.network.reward(obs, action)
            r1, r2 = torch.chunk(reward.sum(dim=-2), 2, dim=-3)
            logits = r2 - r1

            reward_loss = self.reward_criterion(logits, batch["label"]).mean()
            with torch.no_grad():
                reward_accuracy = None
