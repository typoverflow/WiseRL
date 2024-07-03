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
from wiserl.trainer.offline_trainer import OfflineTrainer


class MultiSpatialRewardModelBasedOfflineTrainer(OfflineTrainer):
    def __init__(
        self,
        algorithm,
        env_fn: Optional[Callable] = None,
        eval_env_fn: Optional[Callable] = None,
        rm_dataset_kwargs: Optional[Sequence[str]] = None,
        rm_dataloader_kwargs: Optional[Sequence[Dict]] = None,
        embed_steps: int = 1000,
        cluster_steps: int = 10,
        rm_eval_kwargs: Optional[dict] = None,
        rl_dataset_kwargs: Optional[Sequence[str]] = None,
        rl_dataloader_kwargs: Optional[Sequence[Dict]] = None,
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
            dataset_kwargs=None,
            dataloader_kwargs=None,
            eval_kwargs=None,
            total_steps=rl_steps,
            log_freq=log_freq,
            env_freq=env_freq,
            eval_freq=eval_freq,
            profile_freq=profile_freq,
            checkpoint_freq=checkpoint_freq,
            logger=logger,
            device=device
        )
        self.embed_steps = embed_steps
        self.cluster_steps = cluster_steps
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
        self.rl_dataset_kwargs = rl_dataset_kwargs
        self.rl_dataloader_kwargs = rl_dataloader_kwargs
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
            self._embed_dataloaders, self._embed_dataloaders_iter = self.setup_dataloaders(self._rm_datasets, self.rm_dataloader_kwargs)
            
            for step in trange(0, self.embed_steps+1, desc="pretrain embedding"):
                batches = [next(d) for d in self._embed_dataloaders_iter]
                batches = self.algorithm.format_batch(batches)
                pretrain_metrics = self.algorithm.embedding_step(batches, step=step, total_steps=self.embed_steps)

            # use the whole dataset for clustering
            #for dataset in self._rm_datasets:
            #    dataset.batch_size = len(dataset)
            #self._cluster_dataloaders, self._cluster_dataloaders_iter = self.setup_dataloaders(self._rm_datasets, self.rm_dataloader_kwargs)

            for step in trange(0, self.cluster_steps+1, desc="pretrain clustering"):
                batches = [self._rm_datasets[0].data]
                batches = self.algorithm.format_batch(batches)
                pretrain_metrics = self.algorithm.cluster_step(batches)
                eval_metrics = self.algorithm.cluster_eval(batches)


        # finally train the rl agent
        # TODO rl training here
        

        # clean up
        #self.algorithm.save(self.logger.output_dir, "final.pt", checkpoint_metadata)
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
