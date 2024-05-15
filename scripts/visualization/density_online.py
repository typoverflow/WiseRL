import argparse
import functools
import os
import shutil
import numpy as np
import torch
from PIL import Image
from tqdm import trange

from UtilsRL.exp import parse_args, setup

import wiserl.algorithm
from wiserl.env import get_env
from wiserl.dataset import D4RLOfflineDataset

import matplotlib.pyplot as plt

import d4rl

if __name__ == "__main__":
    args = parse_args(convert=False)
    name_prefix = f"{args['algorithm']['class']}/{args['name']}/{args['env']}"
    setup(args)

    # process the environment
    env_fn = functools.partial(get_env, args["env"], args["env_kwargs"], args["env_wrapper"], args["env_wrapper_kwargs"])
    eval_env_fn = functools.partial(get_env, args["env"], args["env_kwargs"], args["env_wrapper"], args["env_wrapper_kwargs"])
    env = env_fn()
    
    with torch.no_grad():

        # define the algorithm
        algorithm = vars(wiserl.algorithm)[args["algorithm"].pop("class")](
            env.observation_space,
            env.action_space,
            args["network"],
            args["optim"],
            args["schedulers"],
            args["processor"],
            args["checkpoint"],
            **args["algorithm"],
            device=args["device"]
        )
        
        # load pretrained model
        pretrain_model_path = f"ckpts/{args['env']}/{args['name']}/seed{args['seed']}"
        algorithm.load_pretrain(pretrain_model_path)
        
        # load BehavioralCloning
        bc_args = parse_args("scripts/configs/bc/gym.yaml", convert=False)
        setup(bc_args)
        bc = vars(wiserl.algorithm)[bc_args["algorithm"].pop("class")](
            env.observation_space,
            env.action_space,
            bc_args["network"],
            bc_args["optim"],
            bc_args["schedulers"],
            bc_args["processor"],
            bc_args["checkpoint"],
            **bc_args["algorithm"],
            device=bc_args["device"]
        ).to(bc_args["device"])
        # bc_path = args["bc_path"] if "bc_path" in args else "log/BehavioralCloning/default/hopper-medium-replay-v2/seed0-05-14-11-08-314273/output/final.pt"
        bc_path = args["bc_path"] if "bc_path" in args else "log/BehavioralCloning/default/hopper-medium-replay-v2/seed0-05-15-00-20-841938/output/final.pt"
        bc.load(bc_path)
        
        # load dataset and trajectory
        dataset = D4RLOfflineDataset(
            observation_space=env.observation_space,
            action_space=env.action_space,
            env=args["env"],
            batch_size=5,
            mode="trajectory",
            segment_length=100,
            padding_mode="none",
        )
        index = args["index"] if "index" in args else 0
        start = args["start"] if "start" in args else 0
        traj_len = args["traj_len"] if "traj_len" in args else 5
        N = args["N"] if "N" in args else 1000
        default_traj = next(iter(dataset))
        # default_traj = algorithm.format_batch(default_traj)
        B, _, _ = default_traj["obs"].shape
        L = traj_len + 1
        
        # original observation and action
        origin_action = default_traj["action"][index][start]
        origin_obs = default_traj["obs"][index][start]
        # origin_obs_action = torch.cat([origin_obs, origin_action], dim=-1)
        
        # shape [N, L, obs_dim]
        pred_obs = torch.zeros(N, L, env.observation_space.shape[0]).to(bc.device)
        pred_action = torch.zeros(N, L, env.action_space.shape[0]).to(bc.device)
        pred_logprob = torch.zeros(N, L).to(bc.device)
        
        env.reset()
        for i in trange(N):
            qpos, qvel = np.pad(origin_obs[:env.model.nq-1], (1, 0)), origin_obs[env.model.nq-1:]
            env.set_state(qpos, qvel)
            obs, reward, done, _ = env.step(origin_action)
            pred_obs[i, 0] = torch.tensor(origin_obs, dtype=torch.float).to(bc.device)
            pred_action[i, 0] = torch.tensor(origin_action, dtype=torch.float).to(bc.device)
            for j in range(1, L):
                action, logprob, _ = bc.network.actor.sample(torch.tensor(obs, dtype=torch.float).to(bc.device))
                pred_obs[i, j] = torch.tensor(obs, dtype=torch.float).to(bc.device)
                pred_action[i, j] = action
                pred_logprob[i, j] = logprob.squeeze().item()
                obs, reward, done, _ = env.step(action.cpu().numpy())

        tau_logprob = pred_logprob.sum(-1)
        # print(tau_logprob)
        
        # # sample N of z by algorithm.network.future_encoder
        pred_obs_action = torch.concat([pred_obs, pred_action], dim=-1)
        causal_mask = torch.tril(torch.ones([L, L]), diagonal=-1).bool()
        future_mask = torch.triu(torch.ones([L, L]), diagonal=algorithm.future_len+1).bool()
        future_attention_mask = torch.bitwise_or(causal_mask, future_mask).to(algorithm.device)
        out = algorithm.network.future_encoder(
            inputs=pred_obs_action,
            timesteps=None, # here we don't use the timestep from dataset, but use the default `np.arange(len)`
            attention_mask=future_attention_mask,
            do_embedding=True
        )
        z_posterior, *_ = algorithm.network.future_proj.sample(out, deterministic=True, return_mean_logstd=True)
        z_logprob_all, _ = algorithm.network.future_proj.evaluate(out, z_posterior, return_logprob=True)

        z_logprob = z_logprob_all[:, 0, :].squeeze()
        
        # draw plot for z_logprob and tau_logprob, save to scripts/visualization/imgs/density.png
        plt.figure()
        plt.scatter(z_logprob.cpu().numpy(), tau_logprob.cpu().numpy())
        plt.xlabel("z_logprob")
        plt.ylabel("tau_logprob")
        plt.savefig("scripts/visualization/imgs/density.png")
        