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

import wiserl.dataset
import wiserl.eval


class OfflineTrainer(object):
    def __init__(
        self,
        algorithm,
        env_fn: Optional[Callable] = None,
        eval_env_fn: Optional[Callable] = None,
        dataset_kwargs: Optional[Sequence[str]] = None,
        dataloader_kwargs: Optional[Sequence[Dict]] = None,
        eval_kwargs: Optional[dict] = None,
        total_steps: int = 1000,
        log_freq: int = 100,
        env_freq: int = 1,
        eval_freq: int = 1000,
        profile_freq: int = -1,
        checkpoint_freq: Optional[int] = None,
        logger: Optional[BaseLogger] = None,
        device: Union[str, torch.device] = "cpu"
    ):
        # The base model
        self.algorithm = algorithm

        # Environment parameters
        self._env = None
        self.env_fn = env_fn
        self._eval_env = None
        self.eval_env_fn = eval_env_fn

        # Logging parameters
        self.total_steps = total_steps
        self.log_freq = log_freq
        self.env_freq = env_freq
        self.eval_freq = eval_freq
        self.profile_freq = profile_freq
        self.checkpoint_freq = checkpoint_freq
        self.logger = logger

        # Datasets and dataloaders
        self._datasets = None
        self._dataloaders = None
        self.dataset_kwargs = dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs

        # evaluation
        self.eval_fn = None
        self.eval_kwargs = eval_kwargs

        self.algorithm = self.algorithm.to(device)
        self.device = device

    @property
    def env(self):
        if self._env is None and self.env_fn is not None:
            self._env = self.env_fn()
        return self._env

    @property
    def eval_env(self):
        if self._eval_env is None and self.eval_env_fn is not None:
            self._eval_env = self.eval_env_fn()
        return self._eval_env

    def train(self):
        self.logger.info("Set up datasets and dataloaders")
        self._datasets = self.setup_datasets(self.dataset_kwargs)
        self._dataloaders, self._dataloaders_iter = self.setup_dataloaders(self._datasets, self.dataloader_kwargs)

        # start training
        self.logger.info("Start Training")
        self.algorithm.train()
        if self.env_freq is not None:
            env_freq = int(self.env_freq) if self.env_freq >= 1 else 1
            env_iters = int(1 / self.env_freq) if self.env_freq < 1 else 1
        else:
            env_freq = env_iters = None

        for step in trange(0, self.total_steps+1):
            # do env step
            if env_freq and step % env_freq == 0:
                for _ in range(env_iters):
                    self.algorithm.env_step(self.env, step, self.total_steps)

            # do algorithm train step
            batches = [next(d) for d in self._dataloaders_iter]
            batches = self.algorithm.format_batch(batches)
            metrics = self.algorithm.train_step(batches, step=step, total_steps=self.total_steps)

            # log the metrics
            if step % self.log_freq == 0:
                self.logger.log_scalars("", metrics, step=step)

            # run eval and validation
            if self.eval_freq and step % self.eval_freq == 0:
                self.algorithm.eval()
                eval_metrics = self.evaluate()
                self.logger.log_scalars("eval", eval_metrics, step=step)
                self.algorithm.train()

            if self.checkpoint_freq and step % self.checkpoint_freq == 0:
                checkpoint_metadata = dict(step=step)
                self.algorithm.save(self.logger.output_dir, f"step_{step}.pt", checkpoint_metadata)

        # clean up
        checkpoint_metadata = dict(step=step)
        self.algorithm.save(self.logger.output_dir, "final.pt", checkpoint_metadata)
        if self._env is not None:
            self._env.close()
        if self._eval_env is not None:
            self._eval_env.close()

    def setup_datasets(self, dataset_kwargs=None):
        observation_space = self.env.observation_space
        action_space = self.env.action_space

        # parse the dataset arguments
        if dataset_kwargs is None:
            return
        if isinstance(dataset_kwargs, dict):
            dataset_kwargs = [dataset_kwargs, ]
        elif not isinstance(dataset_kwargs, list):
            raise TypeError(f"The type of dataset_kwargs should be either list or dict.")

        _datasets = []
        for kwargs in dataset_kwargs:
            cls = kwargs.pop("class")
            ds = vars(wiserl.dataset)[cls](
                    observation_space,
                    action_space,
                    **kwargs
                )
            _datasets.append(ds)

        return _datasets

    def setup_dataloaders(self, datasets, dataloader_kwargs=None):
        if dataloader_kwargs is None:
            dataloader_kwargs = {}
        if isinstance(dataloader_kwargs, dict):
            dataloader_kwargs = [dataloader_kwargs.copy() for _ in range(len(datasets))]
        elif not isinstance(dataloader_kwargs, list):
            raise TypeError(f"The type of dataloader kwargs should be either dict or list.")

        _dataloaders = []
        for ds, dl_kwargs in zip(datasets, dataloader_kwargs):
            _dataloaders.append(torch.utils.data.DataLoader(ds, **dl_kwargs))

        _dataloaders_iter = [iter(dl) for dl in _dataloaders]
        return _dataloaders, _dataloaders_iter

    def evaluate(self):
        assert not self.algorithm.training
        if self.eval_kwargs is None:
            return {}
        if self.eval_fn is None:
            self.eval_fn = vars(wiserl.eval)[self.eval_kwargs.pop("function")]
        eval_metrics = self.eval_fn(
            self.eval_env, self.algorithm,
            **self.eval_kwargs
        )
        return eval_metrics
