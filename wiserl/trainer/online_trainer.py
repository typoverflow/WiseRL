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
from UtilsRL.rl.buffer import TransitionSimpleReplay

import wiserl.dataset
import wiserl.eval


class OnlineTrainer(object):
    def __init__(
        self,
        algorithm,
        env_fn: Optional[Callable] = None,
        eval_env_fn: Optional[Callable] = None,
        eval_kwargs: Optional[dict] = None,
        max_buffer_size: int = 100000,
        batch_size: int = 256,
        buffer_kwargs: Optional[Sequence[Dict]] = None,
        random_policy_step: int = 5000,
        warmup_step: int = 2000,
        max_trajectory_length: int = 1000,
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
        self.warmup_step = warmup_step
        self.random_policy_step = random_policy_step
        self.max_trajectory_length =max_trajectory_length
        self.total_steps = total_steps
        self.log_freq = log_freq
        self.env_freq = env_freq
        self.eval_freq = eval_freq
        self.profile_freq = profile_freq
        self.checkpoint_freq = checkpoint_freq
        self.logger = logger

        # Buffer
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.buffer_kwargs = buffer_kwargs

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
        self.logger.info("Set up buffer")
        self.buffer = self.setup_buffer(self.max_buffer_size, self.buffer_kwargs)

        # warm up
        self.logger.info("Warm up")
        obs, terminal = self.env.reset(), False
        cur_traj_length = 0
        for step in trange(0, self.warmup_step+1):
            action =  self.env.action_space.sample()
            next_obs, reward, terminal, info = self.env.step(action)
            cur_traj_length += 1
            if cur_traj_length >= self.max_trajectory_length:
                terminal = False
            self.buffer.add_sample(
                {
                    "obs": obs,
                    "action": action,
                    "next_obs": next_obs,
                    "reward": reward,
                    "terminal": terminal,
                }
            )
            obs = next_obs
            if terminal or cur_traj_length >= self.max_trajectory_length:
                obs = self.env.reset()
                cur_traj_length = 0

        # start training
        self.logger.info("Start Training")
        self.algorithm.train()
        
        if self.env_freq is not None:
            env_freq = int(self.env_freq) if self.env_freq >= 1 else 1
            env_iters = int(1 / self.env_freq) if self.env_freq < 1 else 1
        else:
            env_freq = env_iters = None

        obs, terminal = self.env.reset(), False
        cur_traj_length = 0
        for step in trange(0, self.total_steps+1):
            # do env step
            if env_freq and step % env_freq == 0:
                for _ in range(env_iters):
                    self.algorithm.env_step(self.env, step, self.total_steps)
            
            if step < self.random_policy_step + 1:
                action = self.env.action_space.sample()
            else:
                batch = dict(obs=obs)
                batch = self.algorithm.format_batch(batch)
                with torch.no_grad():
                    action = self.algorithm.select_action(batch)

            next_obs, reward, terminal, info = self.env.step(action)
            cur_traj_length += 1
            if cur_traj_length >= self.max_trajectory_length:
                terminal = False
            self.buffer.add_sample(
                {
                    "obs": obs,
                    "action": action,
                    "next_obs": next_obs,
                    "reward": reward,
                    "terminal": terminal,
                }
            )
            obs = next_obs
            if terminal or cur_traj_length >= self.max_trajectory_length:
                obs = self.env.reset()
                cur_traj_length = 0

            # do algorithm train step
            batches = self.buffer.random_batch(self.batch_size)
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
    
    def setup_buffer(self, max_buffer_size, buffer_kwargs=None):
        obs_shape = self.env.observation_space.shape[0]
        action_shape = self.env.action_space.shape[-1]
        buffer = TransitionSimpleReplay(
            max_size=max_buffer_size,
            field_specs={
                "obs": {
                    "shape": [
                        obs_shape,
                    ],
                    "dtype": np.float32,
                },
                "action": {
                    "shape": [
                        action_shape,
                    ],
                    "dtype": np.float32,
                },
                "next_obs": {
                    "shape": [
                        obs_shape,
                    ],
                    "dtype": np.float32,
                },
                "reward": {
                    "shape": [
                        1,
                    ],
                    "dtype": np.float32,
                },
                "terminal": {
                    "shape": [
                        1,
                    ],
                    "dtype": np.float32,
                },
            },
            #**buffer_kwargs
        )
        buffer.reset()
        return buffer

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
