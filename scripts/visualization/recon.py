import argparse
import functools
import os
import shutil
import numpy as np
import torch
from PIL import Image
from tqdm import trange
import cv2

from UtilsRL.exp import parse_args, setup

import wiserl.algorithm
from wiserl.env import get_env
from wiserl.dataset import D4RLOfflineDataset

import d4rl

if __name__ == "__main__":
    args = parse_args(convert=False)
    name_prefix = f"{args['algorithm']['class']}/{args['name']}/{args['env']}"
    setup(args)

    # process the environment
    env_fn = functools.partial(get_env, args["env"], args["env_kwargs"], args["env_wrapper"], args["env_wrapper_kwargs"])
    eval_env_fn = functools.partial(get_env, args["env"], args["env_kwargs"], args["env_wrapper"], args["env_wrapper_kwargs"])
    env = env_fn()

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
    traj = next(iter(dataset))
    traj = algorithm.format_batch(traj)
    B, L, _ = traj["obs"].shape

    # visualize the trajectory
    for i in trange(traj_len):
        qpos, qvel = np.pad(traj["obs"][0, start+i, :env.model.nq-1].cpu().numpy(), (1, 0)), traj["obs"][0, start+i, env.model.nq-1:].cpu().numpy()
        env.set_state(qpos, qvel)
        img = env.render(mode='rgb_array')
        # save to imgs/img_{i}.png
        img = Image.fromarray(img)
        img.save(f"scripts/visualization/imgs/gt_{i}.png")

    # reconstruct the trajectory
    with torch.no_grad():
        obs_action = torch.concat([traj["obs"], traj["action"]], dim=-1)
        out = algorithm.network.future_encoder(
            inputs=obs_action,
            timesteps=None, # here we don't use the timestep from dataset, but use the default `np.arange(len)`
            attention_mask=algorithm.future_attention_mask,
            key_padding_mask=(1-traj["mask"]).squeeze(-1).bool(),
            do_embedding=True
        )
        z_posterior, _, info = algorithm.network.future_proj.sample(out, deterministic=False, return_mean_logstd=True)
        input_obs_action = obs_action[:, start, :]
        input_z_posterior = z_posterior[:, start, :]
        for i in trange(traj_len):
            delta_t = i * torch.ones([B,], dtype=torch.int).to(algorithm.device)
            pred_obs_action = algorithm.network.decoder(
                input_obs_action,
                input_z_posterior,
                delta_t
            )
            obs = pred_obs_action[0, :env.observation_space.shape[0]]
            qpos, qvel = np.pad(obs[:env.model.nq-1].cpu().numpy(), (1, 0)), obs[env.model.nq-1:].cpu().numpy()
            env.set_state(qpos, qvel)
            img = env.render(mode='rgb_array')
            # save to imgs/img_{i}.png
            img = Image.fromarray(img)
            img.save(f"scripts/visualization/imgs/pred_{i}.png")
            
    gt_imgs = [cv2.imread(f'/home/gcx/workspace/fsj/WiseRL/scripts/visualization/imgs/gt_{i}.png') for i in range(31)][::6]
    pred_imgs = [cv2.imread(f'/home/gcx/workspace/fsj/WiseRL/scripts/visualization/imgs/pred_{i}.png') for i in range(31)][::6]

    # 横着排列
    compare_imgs = np.concatenate([np.concatenate(gt_imgs, axis=1), np.concatenate(pred_imgs, axis=1)], axis=0)
    cv2.imwrite('/home/gcx/workspace/fsj/WiseRL/scripts/visualization/imgs/compare.png', compare_imgs)