from typing import Any, Optional

import gym
import numpy as np
import pickle
import random
import torch

from wiserl.utils.functional import discounted_cum_sum

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

class OPPODataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        env: str,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = 1024,
        max_ep_len: Optional[int] = 1024,
        foresee: Optional[int] = 200,
    ):
        super().__init__()
        self.env_name = env
        self.batch_size = 1 if batch_size is None else batch_size
        self.seq_len = seq_len
        self.max_ep_len = max_ep_len
        self.obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.foresee = foresee

        self.load_dataset()

    def __len__(self):
        return self.num_trajectories

    def __iter__(self):
        while True:
            batch_inds = np.random.choice(
                np.arange(self.num_trajectories),
                size=self.batch_size,
                replace=True,
                p=self.p_sample,  # reweights so we sample according to timesteps
            )

            s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
            for i in range(self.batch_size):
                traj = self.trajectories[int(self.sorted_inds[batch_inds[i]])]
                start_idx = random.randint(0, traj['rewards'].shape[0] - 1)

                s.append(traj['observations'][start_idx:start_idx + self.seq_len].reshape(1, -1, self.obs_dim))
                a.append(traj['actions'][start_idx:start_idx + self.seq_len].reshape(1, -1, self.action_dim))
                r.append(traj['rewards'][start_idx:start_idx + self.seq_len].reshape(1, -1, 1))

                if 'terminals' in traj:
                    d.append(traj['terminals'][start_idx:start_idx + self.seq_len].reshape(1, -1))
                else:
                    d.append(traj['dones'][start_idx:start_idx + self.seq_len].reshape(1, -1))
                timesteps.append(np.arange(start_idx, start_idx + s[-1].shape[1]).reshape(1, -1))
                timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len-1

                rtg.append(discount_cumsum(traj['rewards'][start_idx:start_idx+self.foresee], gamma=1.)[0].reshape(1, 1, 1).repeat(s[-1].shape[1] + 1, axis=1))

                # padding and state + reward normalization
                tlen = s[-1].shape[1]
                s[-1] = np.concatenate([s[-1], np.zeros((1, self.seq_len - tlen, self.obs_dim))], axis=1)
                s[-1] = (s[-1] - self.state_mean) / self.state_std
                a[-1] = np.concatenate([a[-1], np.ones((1, self.seq_len - tlen, self.action_dim)) * -10.], axis=1)
                r[-1] = np.concatenate([r[-1], np.zeros((1, self.seq_len - tlen, 1))], axis=1)
                d[-1] = np.concatenate([d[-1], np.ones((1, self.seq_len - tlen)) * 2], axis=1)
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, self.seq_len - tlen, 1))], axis=1) / 1000
                timesteps[-1] = np.concatenate([timesteps[-1], np.zeros((1, self.seq_len - tlen))], axis=1)
                mask.append(np.concatenate([np.ones((1, tlen)), np.zeros((1, self.seq_len - tlen))], axis=1))
            
            s = np.concatenate(s, axis=0).astype(np.float32)
            a = np.concatenate(a, axis=0).astype(np.float32)
            r = np.concatenate(r, axis=0).astype(np.float32)
            d = np.concatenate(d, axis=0).astype(np.float32)
            rtg = np.concatenate(rtg, axis=0).astype(np.float32)
            timesteps = np.concatenate(timesteps, axis=0).astype(np.int32)
            mask = np.concatenate(mask, axis=0).astype(np.int32)
            yield {
                    "obs": s,
                    "action": a,
                    "reward": r,
                    "terminal": d,
                    "return_to_go": rtg,
                    "mask": mask,
                    "timestep": timesteps,
                }
        
    def load_dataset(self):
        dataset_path = '../wiserl/dataset/oppo_data/'+self.env_name+'.pkl'
        print(dataset_path)
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        self.trajectories = trajectories
        
        print(len(trajectories))
        for k in trajectories[0].keys():
            print(k, trajectories[0][k].shape)
        
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        states = np.concatenate(states, axis=0)
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        
        # used for input normalization
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-6
        
        self.sorted_inds = np.argsort(returns)
        self.p_sample = traj_lens[self.sorted_inds] / sum(traj_lens[self.sorted_inds])
        self.num_trajectories = len(self.trajectories)