# python scripts/reward_correlation.py --config scripts/configs/bt_iql/gym/default.yaml --ckpt "log/BTIQL/reward-correlation/hopper-medium-replay-v2/seed0-11-13-19-58-778533/output/final.pt" --data_path "datasets/pt_ipl/hopper-medium-replay-v2/num500_script_train.npz" --name train --env hopper-medium-replay-v2
# python scripts/reward_correlation.py --config scripts/configs/bt_iql/gym/default.yaml --ckpt "log/BTIQL/reward-correlation/hopper-medium-expert-v2/seed0-11-13-23-42-879673/output/final.pt" --data_path "datasets/pt_ipl/hopper-medium-expert-v2/num100_script_train.npz" --name train --env hopper-medium-expert-v2
# python scripts/reward_correlation.py --config scripts/configs/bt_iql/gym/default.yaml --ckpt "log/BTIQL/reward-correlation/walker2d-medium-replay-v2/seed0-11-13-19-58-778533/output/final.pt" --data_path "datasets/pt_ipl/walker2d-medium-replay-v2/num500_script_train.npz" --name train --env walker2d-medium-replay-v2
# python scripts/reward_correlation.py --config scripts/configs/bt_iql/gym/default.yaml --ckpt "log/BTIQL/reward-correlation/walker2d-medium-expert-v2/seed0-11-13-23-42-879673/output/final.pt" --data_path "datasets/pt_ipl/walker2d-medium-expert-v2/num100_script_train.npz" --name train --env walker2d-medium-expert-v2
# python scripts/reward_correlation.py --config scripts/configs/hpl/discrete/gym.yaml --ckpt "log/HindsightPreferenceLearning/reward-correlation/hopper-medium-replay-v2/seed0-11-13-19-58-779067/output/final.pt" --data_path "datasets/pt_ipl/hopper-medium-replay-v2/num500_script_train.npz" --name train --env hopper-medium-replay-v2
# python scripts/reward_correlation.py --config scripts/configs/hpl/discrete/gym.yaml --ckpt "log/HindsightPreferenceLearning/reward-correlation/hopper-medium-expert-v2/seed0-11-13-23-45-881769/output/final.pt" --data_path "datasets/pt_ipl/hopper-medium-expert-v2/num100_script_train.npz" --name train --env hopper-medium-expert-v2
# python scripts/reward_correlation.py --config scripts/configs/hpl/discrete/gym.yaml --ckpt "log/HindsightPreferenceLearning/reward-correlation/walker2d-medium-expert-v2/seed0-11-14-02-21-883247/output/final.pt" --data_path "datasets/pt_ipl/walker2d-medium-replay-v2/num500_script_train.npz" --name train --env walker2d-medium-replay-v2
# python scripts/reward_correlation.py --config scripts/configs/hpl/discrete/gym.yaml --ckpt "log/HindsightPreferenceLearning/reward-correlation/walker2d-medium-expert-v2/seed0-11-14-02-21-883247/output/final.pt" --data_path "datasets/pt_ipl/walker2d-medium-expert-v2/num100_script_train.npz" --name train --env walker2d-medium-expert-v2
import functools
import numpy as np
import torch
from PIL import Image

from UtilsRL.exp import parse_args, setup

import wiserl.algorithm
from wiserl.env import get_env
import matplotlib.pyplot as plt

import d4rl

args = parse_args(convert=False)
setup(args)

# file name prefix
algorithm_class = args['algorithm']['class']
prefix = f"scripts/output/{args['algorithm']['class']}_{args['env']}"

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
    # device=args["device"]
)

# load pretrained model
algorithm.load(args['ckpt'])

import numpy as np

# data_path = 'datasets/pt_ipl/hopper-medium-replay-v2/num500_script_train.npz'
data = {**np.load(args['data_path'])}
data['pred_reward_1'] = algorithm.select_reward({'obs': torch.tensor(data['obs_1'], dtype=torch.float32), 'action': torch.tensor(data['action_1'], dtype=torch.float32)}).squeeze(-1).detach().numpy()
data['pred_reward_2'] = algorithm.select_reward({'obs': torch.tensor(data['obs_2'], dtype=torch.float32), 'action': torch.tensor(data['action_2'], dtype=torch.float32)}).squeeze(-1).detach().numpy()

data['script_reward'] = np.concatenate([data['script_reward_1'], data['script_reward_2']], axis=1)
data['pred_reward'] = np.concatenate([data['pred_reward_1'], data['pred_reward_2']], axis=1)
data['script_return'] = np.sum(data['script_reward'], axis=1)
data['pred_return'] = np.sum(data['pred_reward'], axis=1)

# flatten and normalize data to [0, 1]
data['script_reward'] = data['script_reward'].flatten()
data['pred_reward'] = data['pred_reward'].flatten()
data['script_reward'] = (data['script_reward'] - data['script_reward'].min()) / (data['script_reward'].max() - data['script_reward'].min())
data['pred_reward'] = (data['pred_reward'] - data['pred_reward'].min()) / (data['pred_reward'].max() - data['pred_reward'].min())
data['script_return'] = (data['script_return'] - data['script_return'].min()) / (data['script_return'].max() - data['script_return'].min())
data['pred_return'] = (data['pred_return'] - data['pred_return'].min()) / (data['pred_return'].max() - data['pred_return'].min())

# display the shapes of the data
# obs_1 (500, 100, 11)
# obs_2 (500, 100, 11)
# action_1 (500, 100, 3)
# action_2 (500, 100, 3)
# label (500,)
# script_label (500,)
# script_reward_1 (500, 100)
# script_reward_2 (500, 100)
# timestep_1 (500, 100)
# timestep_2 (500, 100)
# pred_reward_1 (500, 100)
# pred_reward_2 (500, 100)
# script_reward (500, 200)
# pred_reward (500, 200)
# script_return (500,)
# pred_return (500,)
for key in data:
    print(key, data[key].shape)

# calculate correlations
reward_correlation = np.corrcoef(data['script_reward'], data['pred_reward'])[0, 1]
return_correlation = np.corrcoef(data['script_return'], data['pred_return'])[0, 1]

# plot the correlation between the script and predicted rewards
plt.figure()
plt.scatter(data['script_reward'], data['pred_reward'], s=1, alpha=0.5)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
plt.title(f'Reward Correlation ({algorithm_class})')
plt.text(0.05, 0.95, f'Correlation: {reward_correlation:.2f}', transform=plt.gca().transAxes, verticalalignment='top', fontsize=12)
plt.savefig(prefix + 'reward_correlation.png')

# plot the correlation between the script and predicted returns
plt.figure()
plt.scatter(data['script_return'], data['pred_return'], s=1, alpha=0.5, color='red')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
plt.title(f'Return Correlation ({algorithm_class})')
plt.text(0.05, 0.95, f'Correlation: {return_correlation:.2f}', transform=plt.gca().transAxes, verticalalignment='top', fontsize=12)
plt.savefig(prefix + 'return_correlation.png')