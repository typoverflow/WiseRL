"""
4rooms gridworld environment.
"""

import numpy as np
import random
from gym.spaces import Box
import os
from tqdm import trange
import time

class FourRoomsEnv:
    _num_rows = -1
    _num_cols = -1
    _num_states = -1
    _reward_function = None
    _num_actions = 4

    _start_x = 0
    _start_y = 0
    _goal_x = 0
    _goal_y = 0

    _r = None
    _P = None

    def __init__(self,
                random_start=False,
                use_negative_rewards=False,
                episode_len=50, 
                use_onehot_action=False) -> None:
        """
        Initialize by reading from a gridworld definitation string.
        """
        self._parse_string(os.path.join(os.path.dirname(__file__), 'assets/4rooms.mdp'))
        self._curr_x = self._start_x
        self._curr_y = self._start_y
        self._num_states = self._num_rows * self._num_cols
        self._use_negative_rewards = use_negative_rewards
        self._use_onehot_action = use_onehot_action
        self._episode_len = episode_len
        self._random_start = random_start
        if self._use_negative_rewards:
            self._rewards = (-1., 0.)
        else:
            self._rewards = (0., 1.)
        self._build()

        self.state_dim = (2,)
        self.action_dim = 4

        # for gym compatibility
        self.observation_space = Box(shape=self.state_dim, low=-np.inf, high=np.inf, dtype=np.float32)
        self.action_space = Box(low=0.0, high=1.0, shape=(self.action_dim, ), dtype=np.float32)

    @property
    def episode_len(self):
        return self._episode_len

    @property
    def actions(self):
        self.directions = ["A", ">", "V", "<"]
        # return ['up', 'right', 'down', 'left', 'stay']
        return ['up', 'right', 'down', 'left']
    
    @property
    def P(self):
        return self._P

    @property
    def r(self):
        return self._r

    @property
    def num_states(self):
        return self._num_states
        
    @property
    def num_actions(self):
        return self._num_actions

    @property
    def matrix_mdp(self):
        return self._matrix_mdp

    @property
    def num_rows(self):
        return self._num_rows
        
    @property
    def num_cols(self):
        return self._num_cols

    def get_curr_state(self):
        return self.pos_to_state(self._curr_x, self._curr_y)

    def random_action(self):
        """
        Randomly sample an action.
        """
        return random.randrange(self._num_actions)

    def _parse_string(self, path):
        """
        Parsing the definition string.
        """
        # read string
        file_name = open(path, 'r')
        gw_str = ''
        for line in file_name:
            gw_str += line

        # read file
        data = gw_str.split('\n')
        self._num_rows = int(data[0].split(',')[0])
        self._num_cols = int(data[0].split(',')[1])
        self._matrix_mdp = np.zeros((self._num_rows, self._num_cols))
        mdp_data = data[1:]

        for i in range(len(mdp_data)):
            for j in range(len(mdp_data[1])):
                if mdp_data[i][j] == 'X':
                    self._matrix_mdp[i][j] = -1 # wall
                elif mdp_data[i][j] == '.':
                    self._matrix_mdp[i][j] = 0 # ground
                elif mdp_data[i][j] == 'S':
                    self._matrix_mdp[i][j] = 0 # start
                    self._start_x = i
                    self._start_y = j
                elif mdp_data[i][j] == 'G':
                    self._matrix_mdp[i][j] = 0 # goal
                    self._goal_x = i
                    self._goal_y = j
        self.mdp_data = mdp_data
    
          
    def get_state_space(self):
        obs = np.where(self._matrix_mdp == -1, 1, 0)
        empty = np.argwhere(obs == 0)
        obs = np.argwhere(obs != 0)
        return empty, obs

    def get_goal_coord(self):
        return [self._goal_x, self._goal_y]

    def state_to_pos(self, idx):
        """
        Compute pos of the given state.
        """
        y = int(idx % self._num_cols)
        x = int(idx / self._num_cols)
        return x, y

    def pos_to_state(self, x, y):
        """
        Return the index of a state given a coordinate
        """
        idx = x * self._num_cols + y
        return idx
    
    def _get_next_state(self, x, y, a):
        a = a[0] if type(a) == list else a
        """
        Compute the next state by taking an action at (x,y).

        Note that the boarder of the mdp must be walls.
        """
        assert self._matrix_mdp[x][y] != -1
        next_x = x
        next_y = y
        action = self.actions[a]

        if action == 'up' and x > 0:
            next_x = x - 1
            next_y = y
        elif action == 'right' and y < self._num_cols - 1:
            next_x = x
            next_y = y + 1
        elif action == 'down' and x < self._num_rows - 1:
            next_x = x + 1
            next_y = y
        elif action == 'left' and y > 0:
            next_x = x
            next_y = y - 1

        if self._matrix_mdp[next_x][next_y] != -1:
            return next_x, next_y
        else:
            return x, y

    def _get_next_reward(self, next_x, next_y):
        """
        Get reward by taking an action.
        """
        if next_x == self._goal_x and next_y == self._goal_y:
            return self._rewards[1]
        else:
            return self._rewards[0]
      
    def step(self, action):
        """
        One environment step.
        """
        if self._use_onehot_action:
            action = np.argmax(action)
        next_x, next_y = self._get_next_state(
                    self._curr_x,
                    self._curr_y,
                    action
                )
        reward = self._get_next_reward(next_x, next_y)
        self._curr_x = next_x
        self._curr_y = next_y
        obs = np.array((self._curr_x, self._curr_y))
        info = {'state_idx': self.pos_to_state(self._curr_x, self._curr_y)}
        return self._normalize_pos(obs), reward, False, info

    def reset(self):
        """
        Reset the agent to the start position.
        """
        if self._random_start:
            pos = self._random_empty_grids(1)[0]
            self._curr_x = pos[0]
            self._curr_y = pos[1]
            obs = np.array((self._curr_x, self._curr_y))
        else:
            self._curr_x = self._start_x
            self._curr_y = self._start_y
            obs = np.array((self._curr_x, self._curr_y))
        return self._normalize_pos(obs)

    def _normalize_pos(self, pos):
        return pos

    def _random_empty_grids(self, k):
        """
        Return k random empty positions.
        """
        ground = np.argwhere(self._matrix_mdp==0)
        selected = np.random.choice(
                np.arange(ground.shape[0]),
                size=k,
                replace=False
                )
        return ground[selected]

    def _build(self):
        """
        Build reward and transition matrix.
        """
        def one_hot(i, n):
            vec = np.zeros(n)
            vec[i] = 1
            return vec

        self._r = np.zeros((self._num_states, self._num_actions))
        self._P = np.zeros((self._num_states, self._num_actions, self._num_states))

        for x in range(self._num_rows):
            for y in range(self._num_cols):
                if self._matrix_mdp[x][y] == -1:
                    continue
                s_idx = self.pos_to_state(x, y)
                for a in range(self._num_actions):
                    nx, ny = self._get_next_state(x, y, a)
                    ns_idx = self.pos_to_state(nx, ny)
                    self._P[s_idx][a] = one_hot(ns_idx, self._num_states)
                    self._r[s_idx][a] = self._get_next_reward(nx, ny)
                    
    def render(self):
        render_data = self.mdp_data.copy()
        render_data = np.asarray([list(s) for s in render_data])
        render_data[self._curr_x][self._curr_y] = 'A'
        for row in render_data:
            print(' '.join(row))
            
    def distance_to_goal(self):
        return abs(self._curr_x - self._goal_x) + abs(self._curr_y - self._goal_y)
            
    def render_action(self, actions):
        render_data = self.mdp_data.copy()
        render_data = np.asarray([list(s) for s in render_data])
        for i, row in enumerate(render_data):
            for j, col in enumerate(row):
                if col != "X":
                    c = ["^", ">", "v", "<"][actions[i][j]]
                else:
                    c = col
                print(f"  {c}", end="")
            print("")
                

def get_cliff_dataset(agent, env, num_episodes, episode_len=10):
    np.random.seed(42)
    obss = []
    actions = []
    next_obss = []
    returns = []
    # rewards = []
    # terminals = []
    # returns = []
    # traj_returns = []
    # traj_lengths = []
    for e in trange(num_episodes):
        ep_obss = []
        ep_actions = []
        ep_next_obss = []
        # ep_rewards = []
        # ep_returns = []
        obs = env.reset()
        init_distance = env.distance_to_goal()
        for _ in range(episode_len):
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            onehot_action = np.zeros(4, dtype=np.float32)
            onehot_action[action] = 1.0
            
            ep_obss.append(obs)
            ep_actions.append(onehot_action)
            ep_next_obss.append(next_obs)
            
            obs = next_obs
        final_distance = env.distance_to_goal()
        ep_return = init_distance - final_distance
        
        obss.append(np.stack(ep_obss, axis=0))
        actions.append(np.stack(ep_actions, axis=0))
        next_obss.append(np.stack(ep_next_obss, axis=0))
        returns.append(ep_return)
    obss = np.stack(obss, axis=0)
    actions = np.stack(actions, axis=0)
    next_obss = np.stack(next_obss, axis=0)
    returns = np.stack(returns, axis=0)
    return {
        "obs": obss, "action": actions, "next_obs": next_obss, "return": returns
    }
    

class RandomAgent():
    def __init__(self, *args, **kwargs):
        pass
    
    def select_action(self, state):
        return np.random.randint(4)

class OptimalAgent():
    def __init__(self, *args, **kwargs):
        self.optimal_map = [
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
            4, [1, 2], [1, 2], [1, 2], [1, 2], 2, 4, 1, 1, 1, 1, [0, 1], 4, 
            4, [1, 2], [1, 2], [1, 2], [1, 2], 2, 4, [0, 1], [0, 1], [0, 1], [0, 1], 0, 4, 
            4, 1, 1, 1, 1, 1, 1, [0, 1], [0, 1], [0, 1], [0, 1], 0, 4, 
            4, [0, 1], [0, 1], [0, 1], [0, 1], 0, 4, [0, 1], [0, 1], [0, 1], [0, 1], 0, 4, 
            4, [0, 1], [0, 1], [0, 1], [0, 1], 0, 4, [0, 1], [0, 1], [0, 1], [0, 1], 0, 4, 
            4, 4, 0, 4, 4, 4, 4, [0, 1], [0, 1], [0, 1], [0, 1], 0, 4, 
            4, 1, 0, 3, 3, 3, 4, 4, 4, 0, 4, 4, 4, 
            4, [0, 1], 0, [0, 3], [0, 1, 2, 3], 2, 4, 1, 1, 0, 3, 3, 4, 
            4, [0, 1], 0, [0, 1, 2, 3], [1, 2], 2, 4, [0, 1], [0, 1], 0, [0, 3], [0, 3], 4,
            4, 1, [0, 1], 1, 1, 1, 1, [0, 1], [0, 1], 0, [0, 3], [0, 3], 4, 
            4, [0, 1], 0, [0, 1], [0, 1], 0, 4, [0, 1], [0, 1], 0, [0, 3], [0, 3], 4, 
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  
        ]
        actions = []
        for i, a in enumerate(self.optimal_map):
            if a == 4:
                actions.append(np.zeros([4, ]))
                continue
            if type(a) == int:
                a = [a, ]
            act = np.zeros([4, ])
            act[a] = 1/len(a)
            actions.append(act)
        actions = np.stack(actions, axis=0)
        actions = actions.reshape(13, 13, 4)
        self.optimal_actions = actions
    
    def select_action(self, state):
        prob = self.optimal_actions[state[0], state[1]]
        return np.random.choice(4, p=prob)
    

class EpsilonGreedyAgent():
    def __init__(self, eps=0.5, **kwargs):
        self.eps = eps
        self.optimal_agent = OptimalAgent()
        
    def select_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(4)
        else:
            return self.optimal_agent.select_action(state)
    
if __name__ == "__main__":
    # env = FourRoomsEnv()
    # action_map = {
    #     "w": 0, 
    #     "d": 1, 
    #     "s": 2, 
    #     "a": 3
    # }
    # obs = env.reset()
    # done = False
    # env.render()
    # while not done:
    #     action = action_map.get(input().strip())
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    # print("--------------------------------")

    # env = FourRoomsEnv(random_start=True)
    # agent = EpsilonGreedyAgent(0.6)
    # obs = env.reset()
    # done = False
    # env.render()
    # while not done:
    #     time.sleep(0.3)
    #     action = agent.select_action(obs)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    # print("--------------------------------")

    agent = EpsilonGreedyAgent(eps=1.0)
    env = FourRoomsEnv(random_start=True)
    dataset = get_cliff_dataset(agent, env, num_episodes=5000, episode_len=20)
    np.savez("datasets/fourrooms/random_num5000_len20.npz", **dataset)

    agent = EpsilonGreedyAgent(0.0)
    env = FourRoomsEnv(random_start=True)
    dataset = get_cliff_dataset(agent, env, num_episodes=5000, episode_len=50)
    np.savez("datasets/fourrooms/eps0.0_num5000_len50.npz", **dataset)

    agent = EpsilonGreedyAgent(0.3)
    env = FourRoomsEnv(random_start=True)
    dataset = get_cliff_dataset(agent, env, num_episodes=5000, episode_len=50)
    np.savez("datasets/fourrooms/eps0.3_num5000_len50.npz", **dataset)

    agent = EpsilonGreedyAgent(0.6)
    env = FourRoomsEnv(random_start=True)
    dataset = get_cliff_dataset(agent, env, num_episodes=5000, episode_len=50)
    np.savez("datasets/fourrooms/eps0.6_num5000_len50.npz", **dataset)

    agent = EpsilonGreedyAgent(1.0)
    env = FourRoomsEnv(random_start=True)
    dataset = get_cliff_dataset(agent, env, num_episodes=5000, episode_len=50)
    np.savez("datasets/fourrooms/eps1.0_num5000_len50.npz", **dataset)