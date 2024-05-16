import collections
import os
from typing import Any, Dict, List, Optional, Sequence

import gym
import numpy as np
import torch

import wiserl.dataset
from wiserl.algorithm.base import Algorithm

@torch.no_grad()
def eval_gambling_rm(
    env: gym.Env,
    algorithm: Algorithm, 
):
    test_data = {
        "obs": np.asarray([0, 0, 1, 2, 3]), 
        "action": np.asarray([0, 1, 2, 2, 2])
    }
    def convert_to_onehot(x, n):
        onehot = np.eye(n)
        x_onehot = onehot[x, :]
        return x_onehot.astype(np.float32)
    test_data = {
        "obs": convert_to_onehot(test_data["obs"], 5), 
        "action": convert_to_onehot(test_data["action"], 3)
    }
    test_data = algorithm.format_batch(test_data)
    reward = algorithm.select_reward(test_data)
    reward = reward.cpu().numpy().tolist()
    print("reward: ", reward)
    # print(reward)
    # import time
    # time.sleep(0.5)
    return {}

    

@torch.no_grad()
def eval_cliffwalking_rm(
    env: gym,
    algorithm: Algorithm,
    eval_dataset_kwargs: Optional[Sequence[str]]
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
    for batch in eval_dataset.create_sequential_iter():
        batch = algorithm.format_batch(batch)
        batch["obs"] = torch.concat([batch["obs_1"], batch["obs_2"]], dim=0)
        batch["action"] = torch.concat([batch["action_1"], batch["action_2"]], dim=0)
        reward = algorithm.select_reward(batch)
        r1, r2 = torch.chunk(reward, 2, dim=0)
        logit = r2.sum(dim=1) - r1.sum(dim=1)
        label = batch["label"].float()
        reward_loss = algorithm.reward_criterion(logit, label)
        reward_acc = ((logit > 0) == torch.round(label)).float()
        rm_eval_loss.extend(reward_loss)
        rm_eval_acc.extend(reward_acc)
    test_states = torch.tensor([[0., 0.], [0., 0.]], dtype=torch.float32)
    test_actions = torch.tensor([[0,0,1,0], [0,1,0,0]], dtype=torch.float32)
    test_batch = {"obs": test_states, "action": test_actions}
    test_batch = algorithm.format_batch(test_batch)
    reward = algorithm.select_reward(test_batch)
    return {
        "val_loss": torch.as_tensor(rm_eval_loss).mean().item(),
        "val_acc": torch.as_tensor(rm_eval_acc).mean().item(),
        "go_up_reward": reward[0].item(),
        "go_right_reward": reward[1].item()
    }

@torch.no_grad()
def eval_fourrooms_rm(
    env: gym, 
    algorithm: Algorithm, 
    eval_dataset_kwargs
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
    for batch in eval_dataset.create_sequential_iter():
        batch = algorithm.format_batch(batch)
        batch["obs"] = torch.concat([batch["obs_1"], batch["obs_2"]], dim=0)
        batch["action"] = torch.concat([batch["action_1"], batch["action_2"]], dim=0)
        reward = algorithm.select_reward(batch)
        r1, r2 = torch.chunk(reward, 2, dim=0)
        logit = r2.sum(dim=1) - r1.sum(dim=1)
        label = batch["label"].float()
        reward_loss = algorithm.reward_criterion(logit, label)
        reward_acc = ((logit > 0) == torch.round(label)).float()
        rm_eval_loss.extend(reward_loss)
        rm_eval_acc.extend(reward_acc)
    val_loss = torch.as_tensor(rm_eval_loss).mean().item()
    val_acc = torch.as_tensor(rm_eval_acc).mean().item()
    
    test_obs = np.zeros([env.num_rows, env.num_cols, 2])
    for i in range(env.num_rows):
        test_obs[i, :, 0] = i
    for j in range(env.num_cols):
        test_obs[:, j, 1] = j
    test_obs = test_obs.reshape(-1, 2)
    test_obs = np.stack([test_obs]*4, axis=0)
    test_actions = np.zeros([4, test_obs.shape[1], 4])
    for i in range(4):
        test_actions[i, :, i] = 1.
    test_batch = {
        "obs": test_obs, 
        "action": test_actions
    }
    test_batch = algorithm.format_batch(test_batch)
    reward = algorithm.select_reward(test_batch).cpu().numpy()
    argmax_action = np.argmax(reward, axis=0)
    argmax_action = argmax_action.reshape(env.num_rows, env.num_cols)
    env.render_action(argmax_action)
    print(f"val_loss: {val_loss}, val_acc: {val_acc}")
    print("--------------------------------", flush=True)
    return {
        "val_loss": val_loss,
        "val_acc": val_acc
    }

@torch.no_grad()
def eval_fourrooms_rl(
    env: gym.Env, 
    algorithm: Algorithm,
    deterministic=True
):
    test_obs = np.zeros([env.num_rows, env.num_cols, 2])
    for i in range(env.num_rows):
        test_obs[i, :, 0] = i
    for j in range(env.num_cols):
        test_obs[:, j, 1] = j
    test_obs = test_obs.reshape(-1, 2)
    test_batch = {
        "obs": test_obs, 
    }
    test_batch = algorithm.format_batch(test_batch)
    action = algorithm.select_action(test_batch, deterministic=deterministic)
    action = action.reshape(env.num_rows, env.num_cols)
    env.render_action(action)
    print("--------------------------------", flush=True)
    return {}