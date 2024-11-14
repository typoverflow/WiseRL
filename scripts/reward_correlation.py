# python scripts/reward_correlation.py --config scripts/configs/bt_iql/gym/default.yaml --ckpt "log/BTIQL/reward-correlation/hopper-medium-replay-v2/seed0-11-13-19-58-778533/output/final.pt" --env hopper-medium-replay-v2
# python scripts/reward_correlation.py --config scripts/configs/bt_iql/gym/default.yaml --ckpt "log/BTIQL/reward-correlation/hopper-medium-expert-v2/seed0-11-13-23-42-879673/output/final.pt" --env hopper-medium-expert-v2
# python scripts/reward_correlation.py --config scripts/configs/bt_iql/gym/default.yaml --ckpt "log/BTIQL/reward-correlation/walker2d-medium-replay-v2/seed0-11-14-15-03-950718/output/final.pt" --env walker2d-medium-replay-v2
# python scripts/reward_correlation.py --config scripts/configs/bt_iql/gym/default.yaml --ckpt "log/BTIQL/reward-correlation/walker2d-medium-expert-v2/seed0-11-14-15-04-953448/output/final.pt" --env walker2d-medium-expert-v2
# python scripts/reward_correlation.py --config scripts/configs/hpl/discrete/gym.yaml --ckpt "log/HindsightPreferenceLearning/reward-correlation/hopper-medium-replay-v2/seed0-11-13-19-58-779067/output/final.pt" --env hopper-medium-replay-v2
# python scripts/reward_correlation.py --config scripts/configs/hpl/discrete/gym.yaml --ckpt "log/HindsightPreferenceLearning/reward-correlation/hopper-medium-expert-v2/seed0-11-13-23-45-881769/output/final.pt" --env hopper-medium-expert-v2
# python scripts/reward_correlation.py --config scripts/configs/hpl/discrete/gym.yaml --ckpt "log/HindsightPreferenceLearning/reward-correlation/walker2d-medium-replay-v2/seed0-11-14-01-08-882667/output/final.pt" --env walker2d-medium-replay-v2
# python scripts/reward_correlation.py --config scripts/configs/hpl/discrete/gym.yaml --ckpt "log/HindsightPreferenceLearning/reward-correlation/walker2d-medium-expert-v2/seed0-11-14-02-21-883247/output/final.pt" --env walker2d-medium-expert-v2
import functools
import numpy as np
import torch
from PIL import Image

from UtilsRL.exp import parse_args, setup

import wiserl.algorithm
from wiserl.env import get_env
import matplotlib.pyplot as plt
from wiserl.dataset import D4RLOfflineDataset

import d4rl

config_map = {
    'BTIQL': 'scripts/configs/bt_iql/gym/default.yaml',
    'HindsightPreferenceLearning': 'scripts/configs/hpl/discrete/gym.yaml',
}

ckpt_map = {
    'BTIQL': {
        'hopper-medium-replay-v2': 'log/BTIQL/reward-correlation/hopper-medium-replay-v2/seed0-11-13-19-58-778533/output/final.pt',
        'hopper-medium-expert-v2': 'log/BTIQL/reward-correlation/hopper-medium-expert-v2/seed0-11-13-23-42-879673/output/final.pt',
        'walker2d-medium-replay-v2': 'log/BTIQL/reward-correlation/walker2d-medium-replay-v2/seed0-11-14-15-03-950718/output/final.pt',
        'walker2d-medium-expert-v2': 'log/BTIQL/reward-correlation/walker2d-medium-expert-v2/seed0-11-14-15-04-953448/output/final.pt',
    },
    'HindsightPreferenceLearning': {
        'hopper-medium-replay-v2': 'log/HindsightPreferenceLearning/reward-correlation/hopper-medium-replay-v2/seed0-11-13-19-58-779067/output/final.pt',
        'hopper-medium-expert-v2': 'log/HindsightPreferenceLearning/reward-correlation/hopper-medium-expert-v2/seed0-11-13-23-45-881769/output/final.pt',
        'walker2d-medium-replay-v2': 'log/HindsightPreferenceLearning/reward-correlation/walker2d-medium-replay-v2/seed0-11-14-01-08-882667/output/final.pt',
        'walker2d-medium-expert-v2': 'log/HindsightPreferenceLearning/reward-correlation/walker2d-medium-expert-v2/seed0-11-14-02-21-883247/output/final.pt',
    },
}


# file name prefix
algorithm_name_map = {
    'BTIQL': 'MR',
    'HindsightPreferenceLearning': 'HPL',
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
envs = ['hopper-medium-replay-v2', 'hopper-medium-expert-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2']
colors = {'BTIQL': 'blue', 'HindsightPreferenceLearning': 'red'}

for i, env in enumerate(envs):
    for j, algorithm_class in enumerate(['BTIQL', 'HindsightPreferenceLearning']):
        args = parse_args(config_map[algorithm_class], convert=False)
        args['env'] = env
        args['algorithm'].pop('class')
        setup(args)

        algorithm_name = algorithm_name_map.get(algorithm_class, algorithm_class)
        
        env_fn = functools.partial(get_env, env, args["env_kwargs"], args["env_wrapper"], args["env_wrapper_kwargs"])
        eval_env_fn = functools.partial(get_env, env, args["env_kwargs"], args["env_wrapper"], args["env_wrapper_kwargs"])
        env_instance = env_fn()
        
        algorithm = vars(wiserl.algorithm)[algorithm_class](
            env_instance.observation_space,
            env_instance.action_space,
            args["network"],
            args["optim"],
            args["schedulers"],
            args["processor"],
            args["checkpoint"],
            **args["algorithm"],
            # device=args["device"],
        )
        
        algorithm.load(ckpt_map[algorithm_class][env])
        
        data = D4RLOfflineDataset(None, None, env).data

        # length of the dataset to use
        length = 200000
        # shuffle the dataset
        np.random.seed(0)
        indices = np.random.permutation(len(data['obs']))
        # select a subset of the dataset
        indices = indices[:length + 1]
        # select the subset of the dataset
        for key in data.keys():
            data[key] = data[key][indices]
        
        data['pred_reward'] = algorithm.select_reward({
            'obs': torch.tensor(data['obs'], dtype=torch.float32),
            'action': torch.tensor(data['action'], dtype=torch.float32),
        }).detach().numpy().squeeze()
        data['reward'] = data['reward'].squeeze()
        
        data['reward'] = (data['reward'] - data['reward'].min()) / (data['reward'].max() - data['reward'].min())
        
        reward_correlation = np.corrcoef(data['reward'], data['pred_reward'])[0, 1]
        
        # plot the correlation between the script and predicted rewards
        ax = axes[i // 2, (i % 2) * 2 + j]
        ax.scatter(data['reward'], data['pred_reward'], s=1, alpha=0.1)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        ax.set_title(f'({chr(97 + i)}) {env} ({algorithm_name})')
        ax.text(0.05, 0.95, f'Correlation: {reward_correlation:.2f}', transform=ax.transAxes, verticalalignment='top', fontsize=12)

plt.tight_layout()
plt.savefig('scripts/output/reward_correlation_comparison.png')
