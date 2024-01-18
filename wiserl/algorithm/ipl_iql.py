import itertools
from typing import Any, Dict, Optional, Tuple, Type

import gym
import imageio
import numpy as np
import torch

from wiserl.algorithm.base import Algorithm
from wiserl.algorithm.oracle_iql import OracleIQL


class IPL_IQL(OracleIQL):
    def __init__(
        self,
        *args,
        expectile: float = 0.7,
        beta: float = 0.3333,
        chi2_coeff: float = 0.5,
        chi2_replay_weight: Optional[float] = None,
        policy_replay_weight: Optional[float] = None,
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
        self.chi2_coeff = chi2_coeff
        self.chi2_replay_weight = chi2_replay_weight
        self.policy_replay_weight = policy_replay_weight

    def setup_network(self, network_kwargs):
        super().setup_network(network_kwargs)

    def setup_optimizers(self, optim_kwargs):
        super().setup_optimizers(optim_kwargs)

    def select_action(self, batch, deterministic: bool=True):
        return super().select_action(batch, deterministic)

    def train_step(self, batches, step: int, total_steps: int):
        pass
