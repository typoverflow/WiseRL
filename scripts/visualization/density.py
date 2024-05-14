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
        
        # load dataset and trajectory
        dataset = D4RLOfflineDataset(
            observation_space=env.observation_space,
            action_space=env.action_space,
            env=args["env"],
            batch_size=2,
            mode="trajectory",
            segment_length=100,
            padding_mode="none",
        )
        start = args["start"] if "start" in args else 0
        traj_len = args["traj_len"] if "traj_len" in args else 5
        default_traj = next(iter(dataset))
        default_traj = algorithm.format_batch(default_traj)
        B, L, _ = default_traj["obs"].shape
        
        # original observation and action
        origin_action = default_traj["action"][0][start]
        origin_obs = default_traj["obs"][0][start]
        origin_obs_action = torch.cat([origin_obs, origin_action], dim=-1)
        
        # sample N of z by algorithm.network.prior
        # GaussianActor(input_dim=self.obs_dim+self.action_dim, output_dim=self.z_dim)
        N = 1000
        # duplicate origin_obs_act to N
        z, z_logprob, _ = algorithm.network.prior.sample(origin_obs_action.repeat(N, 1))
        z_logprob = z_logprob.squeeze()
        
        # sample N of traj by algorithm.network.decoder
        delta_t = torch.arange(1, traj_len + 1).unsqueeze(0).repeat(N, 1).to(algorithm.device)
        input_obs_action = origin_obs_action.repeat(N, traj_len, 1)
        input_z = z.unsqueeze(1).repeat(1, traj_len, 1)
        pred_obs_action = algorithm.network.decoder(input_obs_action, input_z, delta_t)
        pred_obs = pred_obs_action[..., :env.observation_space.shape[0]]
        pred_action = pred_obs_action[..., env.observation_space.shape[0]:]

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
        bc_path = args["bc_path"] if "bc_path" in args else "log/BehavioralCloning/default/hopper-medium-replay-v2/seed0-05-14-11-08-314273/output/final.pt"
        bc.load(bc_path)
        logprob, _ = bc.network.actor.evaluate(pred_obs, pred_action)
        sum_logprob = logprob.squeeze().sum(-1)
        
        # draw plot for z_logprob and sum_logprob, save to scripts/visualization/imgs/density.png
        plt.figure()
        plt.scatter(z_logprob.cpu().numpy(), sum_logprob.cpu().numpy())
        plt.xlabel("z_logprob")
        plt.ylabel("sum_logprob")
        plt.savefig("scripts/visualization/imgs/density.png")
        