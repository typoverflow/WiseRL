import collections
import os
from typing import Any, Dict, List, Optional, Sequence

import gym
import numpy as np
import torch

import wiserl.dataset
from wiserl.algorithm.base import Algorithm
from scipy import stats

@torch.no_grad()
def eval_reward_model(
    env: gym.Env,
    algorithm: Algorithm,
    eval_dataset_kwargs: Optional[Sequence[str]],
):
    rm_eval_loss = []
    rm_eval_acc = []
    kwargs = eval_dataset_kwargs.copy()
    eval_dataset_class = kwargs.pop("class")
    eval_dataset = vars(wiserl.dataset)[eval_dataset_class](
        env.observation_space,
        env.action_space,
        **kwargs
    )
    rewards = []
    true_rewards = []
    for batch in eval_dataset.create_sequential_iter():
        batch = algorithm.format_batch(batch)
        batch["obs"] = torch.concat([batch["obs_1"], batch["obs_2"]], dim=0)
        batch["action"] = torch.concat([batch["action_1"], batch["action_2"]], dim=0)
        reward = algorithm.select_reward(batch)
        r1, r2 = torch.chunk(reward, 2, dim=0)

        rewards.extend(r1.sum(dim=1).squeeze().cpu())
        rewards.extend(r2.sum(dim=1).squeeze().cpu())
        true_rewards.extend(batch['reward_1'].sum(dim=1).cpu())
        true_rewards.extend(batch['reward_2'].sum(dim=1).cpu())

        logit = r2.sum(dim=1) - r1.sum(dim=1)
        label = batch["label"].float()
        reward_loss = algorithm.reward_criterion(logit, label)
        reward_acc = ((logit > 0) == torch.round(label)).float()
        rm_eval_loss.extend(reward_loss)
        rm_eval_acc.extend(reward_acc)
        
    r, _ = stats.pearsonr(rewards, true_rewards)
    return {
        "val_loss": torch.as_tensor(rm_eval_loss).mean().item(),
        "val_acc": torch.as_tensor(rm_eval_acc).mean().item(),
        "val_pearsonr": r
    }
