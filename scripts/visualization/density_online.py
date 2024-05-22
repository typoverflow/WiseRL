import argparse
import functools
import os
import shutil
from click import style
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
        alpha = args["alpha"] if "alpha" in args else 0.5
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
        z_logprob = z_logprob.cpu().numpy()
        tau_logprob = tau_logprob.cpu().numpy()
        # 异常值处理
        z_logprob = z_logprob[tau_logprob > -12]
        tau_logprob = tau_logprob[tau_logprob > -12]
        
        # draw plot for z_logprob and tau_logprob, save to scripts/visualization/imgs/density.png
        plt.figure()
        plt.scatter(z_logprob, tau_logprob, alpha=alpha)
        # 设置字体大小为原来 1.5 倍
        plt.xlabel(r"$\log p(z_t|s_t, a_t)$", fontsize=15)
        plt.ylabel(r"$\log p(\sigma_{t:t+k}|s_t, a_t)$", fontsize=15)
        # 关闭刻度
        plt.xticks([])
        plt.yticks([])
        # 进行 pca 降维算出斜率
        from sklearn.decomposition import PCA
        
        # 使用 PCA 进行主成分分析
        pca = PCA(n_components=2)
        X = np.array([z_logprob, tau_logprob]).T
        pca.fit(X)
        components = pca.components_
        mean = pca.mean_

        # 计算长轴（第一主成分）的斜率
        vector = components[0]
        slope = vector[1] / vector[0] * 0.3

        print(f'长轴（第一主成分）的斜率: {slope}')

        # 绘制数据点
        # plt.scatter(X[:, 0], X[:, 1], alpha=0.2, label='数据点')

        # 绘制长轴对角线
        x_vals = np.array([100, 110])
        y_vals = mean[1] + slope * (x_vals - mean[0])
        plt.plot(x_vals, y_vals, color="#909090", linestyle='--', lw=2)
        
        # 将线上移 5 个单位，然后画出来
        y_vals_up = mean[1] + slope * (x_vals - mean[0]) + 5
        x_vals_up = x_vals
        # 截断大于 6 的部分
        # x_vals_up = x_vals[y_vals_up < 6]
        # y_vals_up = y_vals_up[y_vals_up < 6]
        plt.plot(x_vals_up, y_vals_up, color="#909090", linestyle='--', lw=2)
        
        # 将线下移 8 个单位，然后画出来
        y_vals_down = mean[1] + slope * (x_vals - mean[0]) - 8.5
        x_vals_down = x_vals
        # 截断小于 -12.5 的部分
        # x_vals_down = x_vals[y_vals_down > -12.5]
        # y_vals_down = y_vals_down[y_vals_down > -12.5]
        plt.plot(x_vals_down, y_vals_down, color="#909090", linestyle='--', lw=2)
        
        # 限定 y 轴范围
        plt.ylim(-12.5, 6)
        # 限定 x 轴范围
        plt.xlim(106.5, 107.8)
        
        # 去除边框
        # plt.gca().spines['top'].set_visible(False)
        # plt.gca().spines['right'].set_visible(False)
        # plt.gca().spines['bottom'].set_visible(False)
        # plt.gca().spines['left'].set_visible(False)
        # 收窄边距
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        # 设置清晰度
        plt.savefig("scripts/visualization/imgs/density.pdf")
        