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
from wiserl.trainer.online_trainer import OnlineTrainer


class RewardModelBasedOnlineTrainer(OnlineTrainer):
    def __init__(
        self,
        algorithm,
        env_fn: Optional[Callable] = None,
        eval_env_fn: Optional[Callable] = None,
        rm_dataset_kwargs: Optional[Sequence[str]] = None,
        rm_dataloader_kwargs: Optional[Sequence[Dict]] = None,
        rm_steps: int = 1000,
        rm_eval_kwargs: Optional[dict] = None,
        max_buffer_size: int = 100000,
        batch_size: int = 256,
        buffer_kwargs: Optional[Sequence[Dict]] = None,
        random_policy_step: int = 5000,
        warmup_step: int = 2000,
        max_trajectory_length: int = 1000,
        rl_steps: int = 1000,
        rl_eval_kwargs: Optional[dict] = None,
        rm_label: bool=False,
        load_rm_path: Optional[str] = None,
        save_rm_path: Optional[str] = None,
        log_freq: int = 100,
        env_freq: int = 1,
        eval_freq: int = 1000,
        profile_freq: int = -1,
        checkpoint_freq: Optional[int] = None,
        logger: Optional[BaseLogger] = None,
        device: Union[str, torch.device] = "cpu"
    ):
        super().__init__(
            algorithm=algorithm,
            env_fn=env_fn,
            eval_env_fn=eval_env_fn,
            eval_kwargs=None,
            max_buffer_size=max_buffer_size,
            batch_size=batch_size,
            buffer_kwargs=buffer_kwargs,
            random_policy_step=random_policy_step,
            warmup_step=warmup_step,
            max_trajectory_length=max_trajectory_length,
            total_steps=rl_steps,
            log_freq=log_freq,
            env_freq=env_freq,
            eval_freq=eval_freq,
            profile_freq=profile_freq,
            checkpoint_freq=checkpoint_freq,
            logger=logger,
            device=device
        )
        self.rm_steps = rm_steps
        self.rl_steps = rl_steps
        self.rm_label = rm_label
        self.load_rm_path = load_rm_path
        self.save_rm_path = save_rm_path
        # rm & rl datasets, dataloaders, and evals
        self._rm_datasets = self._rl_datasets = None
        self._rm_dataloaders = self._rl_dataloaders = None
        self._rm_eval_fn = self._rl_eval_fm = None

        self.rm_dataset_kwargs = rm_dataset_kwargs
        self.rm_dataloader_kwargs = rm_dataloader_kwargs
        self.rm_eval_kwargs = rm_eval_kwargs
        self.rl_eval_kwargs = rl_eval_kwargs

    def train(self):
        self.algorithm.train()

        # first train the reward model
        if self.load_rm_path is not None:
            self.logger.info(f"Loading pretrained model from {self.load_rm_path} ... ")
            self.algorithm.load_pretrain(self.load_rm_path)
        else:
            self.logger.info("Setting up pretrain datasets and dataloaders ... ")
            self._rm_datasets = self.setup_datasets(self.rm_dataset_kwargs)
            self._rm_dataloaders, self._rm_dataloaders_iter = self.setup_dataloaders(self._rm_datasets, self.rm_dataloader_kwargs)

            self.logger.info("Starting pretraining ... ")
            for step in trange(0, self.rm_steps+1, desc="pretrain"):
                batches = [next(d) for d in self._rm_dataloaders_iter]
                batches = self.algorithm.format_batch(batches)
                pretrain_metrics = self.algorithm.pretrain_step(batches, step=step, total_steps=self.rm_steps)

                if step % self.log_freq == 0:
                    self.logger.log_scalars("pretrain", pretrain_metrics, step=step)

                if self.eval_freq and step % self.eval_freq == 0:
                    self.algorithm.eval()
                    eval_metrics = self.rm_evaluate()
                    self.logger.log_scalars("eval", eval_metrics, step=step)
                    self.algorithm.train()

            if self.save_rm_path is not None:
                self.logger.info(f"Saving pretrained model to {self.save_rm_path} ...")
                self.algorithm.save_pretrain(self.save_rm_path)

        # finally train the rl agent
        self.logger.info("Set up buffer")
        self.buffer = self.setup_buffer(self.max_buffer_size, self.buffer_kwargs)

        # self.logger.info(f"Setting up rl datasets and dataloaders ...")
        # self._rl_datasets = self.setup_datasets(self.rl_dataset_kwargs)
        # if self.rm_label:
        #     self.logger.info(f"Relabeling the reward using pretrained reward model ...")
        #     self.algorithm.eval()
        #     for d in self._rl_datasets:
        #         d.relabel_reward(self.algorithm)
        #     self.algorithm.train()
        # self._rl_dataloaders, self._rl_dataloaders_iter = self.setup_dataloaders(self._rl_datasets, self.rl_dataloader_kwargs)
        
        self.logger.info("Warm up")
        obs, terminal = self.env.reset(), False
        cur_traj_length = 0
        for step in trange(0, self.warmup_step+1):
            action =  self.env.action_space.sample()
            next_obs, reward, terminal, info = self.env.step(action)
            # Relabling reward
            if self.rm_label:
                batch = dict(obs=obs, action=action)
                batch = self.algorithm.format_batch(batch)
                reward = self.algorithm.select_reward(batch).detach().cpu().numpy()
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

        self.logger.info("Start Training")
        self.algorithm.train()

        obs, terminal = self.env.reset(), False
        cur_traj_length = 0
        traj_return = 0
        true_traj_return = 0
        
        for step in trange(0, self.rl_steps+1, desc="RL"):
            # random sampling
            if step < self.random_policy_step + 1:
                action = self.env.action_space.sample()
            else:
                batch = dict(obs=obs)
                batch = self.algorithm.format_batch(batch)
                with torch.no_grad():
                    action = self.algorithm.select_action(batch)
            
            next_obs, reward, terminal, info = self.env.step(action)
            true_traj_return += reward
            cur_traj_length += 1
            if cur_traj_length >= self.max_trajectory_length:
                terminal = False
            # Relabling reward
            if self.rm_label:
                batch = dict(obs=obs, action=action)
                batch = self.algorithm.format_batch(batch)
                reward = self.algorithm.select_reward(batch).detach().cpu().numpy()
            traj_return += reward
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
                obs, terminal = self.env.reset(), False
                train_metrics = dict(traj_return=traj_return, true_traj_return=true_traj_return)
                self.logger.log_scalars("train", train_metrics, step=step)
                cur_traj_length = 0
                traj_return = 0
                true_traj_return = 0  

            batches = self.buffer.random_batch(self.batch_size)
            batches = self.algorithm.format_batch(batches)
            rl_metrics = self.algorithm.train_step(batches, step=step, total_steps=self.rl_steps)

            if step % self.log_freq == 0:
                self.logger.log_scalars("", rl_metrics, step=step)

            if self.eval_freq and step % self.eval_freq == 0:
                self.algorithm.eval()
                eval_metrics = self.rl_evaluate()
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

    def rm_evaluate(self):
        if self.rm_eval_kwargs is None:
            return {}
        if not hasattr(self, "rm_eval_fn"):
            self.rm_eval_fn = vars(wiserl.eval)[self.rm_eval_kwargs.pop("function")]
        eval_metrics = self.rm_eval_fn(
            self.eval_env, self.algorithm,
            **self.rm_eval_kwargs
        )
        return eval_metrics

    def rl_evaluate(self):
        assert not self.algorithm.training
        if self.rl_eval_kwargs is None:
            return {}
        if not hasattr(self, "rl_eval_fn"):
            self.rl_eval_fn = vars(wiserl.eval)[self.rl_eval_kwargs.pop("function")]
        eval_metrics = self.rl_eval_fn(
            self.eval_env, self.algorithm,
            **self.rl_eval_kwargs
        )
        return eval_metrics
