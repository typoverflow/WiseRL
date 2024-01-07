import os
import random
import tempfile
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import gym
import numpy as np
import torch
from tqdm import trange
from UtilsRL.logger import BaseLogger

import src


class OfflineTrainer(object):
    def __init__(
        self,
        algorithm,
        env_fn: Optional[Callable] = None,
        eval_env_fn: Optional[Callable] = None,
        env_runner: Optional[str] = None,
        eval_env_runner: Optional[str] = None,
        total_steps: int = 1000,
        log_freq: int = 100,
        env_freq: int = 1,
        eval_freq: int = 1000,
        profile_freq: int = -1,
        checkpoint_freq: Optional[int] = None,
        max_validation_steps: Optional[int] = None,
        loss_metric: Optional[str] = "loss",
        benchmark: bool = False,
        eval_fn: Optional[Any] = None,
        eval_kwargs: Optional[Dict] = None,
        train_dataloader_kwargs: Optional[Dict] = None,
        validation_dataloader_kwargs: Optional[Dict] = None,
        logger: Optional[BaseLogger] = None
    ):
        # The base model
        self.algorithm = algorithm

        # Environment parameters
        self._env = None
        self.env_fn = env_fn
        self.env_runner = env_runner
        self._eval_env = None
        self.eval_env_fn = eval_env_fn
        self.eval_env_runner = eval_env_runner

        # Logging parameters
        self.total_steps = total_steps
        self.log_freq = log_freq
        self.env_freq = env_freq
        self.eval_freq = eval_freq
        self.profile_freq = profile_freq
        self.checkpoint_freq = checkpoint_freq
        self.logger = logger

        # Dataloader parameters
        self._train_dataloader = None
        self.train_dataloader_kwargs = {} if train_dataloader_kwargs is None else train_dataloader_kwargs
        self._validation_dataloader = None
        self.validation_dataloader_kwargs = {} if validation_dataloader_kwargs is None else validation_dataloader_kwargs
        self._validation_iterator = None

    def train(self):
        # prepare the algorithm for training
        self.logger.info("Setting up optimizers and schedulers")
        self.algorithm.setup_optimizers()
        self.algorithm.setup_schedulers()

        self.logger.info("Setting up datasets and dataloaders")
        self.setup_datasets(self.env, self.total_steps)
        self.setup_dataloaders()


        # start training
        self.logger.info("Training")
        self.algorithm.train()
        if env_freq is not None:
            env_freq = int(self.env_freq) if self.env_freq >= 1 else 1
            env_iters = int(1 / self.env_freq) if self.env_freq < 1 else 1
        else:
            env_freq = env_iters = None

        for step in trange(1, self.total_steps+1):
            # do env step
            if env_freq and step % env_freq == 0:
                for _ in range(env_iters):
                    self.algorithm.env_step(self.env, step, self.total_steps)

            # do algorithm train step
            batches = [next(d) for d in self.train_dataloaders]
            metrics = self.algorithm.train_step(*batches, step=step, total_steps=self.total_steps)

            # log the metrics
            if step % self.log_freq == 0:
                self.logger.log_scalars("", metrics, step=step)

            # run eval and validation
            if step % self.eval_freq == 0:
                self.algorithm.eval()
                eval_metrics = self.evaluate()
                self.logger.log_scalars("", eval_metrics, step=step)
                self.algorithm.train()

        # clean up
        if self._env is not None:
            self._env.close()
        if self._eval_env is not None:
            self._eval_env.close()

    def setup_datasets(self, env: gym.Env, total_steps: int):
        observation_space = env.observation_space
        action_space = env.action_space
        self._datasets = []
        for kwargs in self.dataset_kwargs:
            self._datasets.append(
                vars(src.dataset)[kwargs["dataset_class"]](
                    observation_space,
                    action_space,
                    **kwargs
                )
            )

    def setup_dataloaders(self):
        self._train_dataloaders = []
        for d, kwargs in zip(self._datasets, self.train_dataloader_kwargs):
            if isinstance(d, torch.utils.data.IterableDataset):
                self._train_dataloaders.append(
                    torch.utils.data.DataLoader(
                        d,
                        pin_memory=True,
                        **kwargs
                    )
                )
