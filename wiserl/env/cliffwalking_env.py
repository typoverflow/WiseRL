import gym
import numpy as np
from gym import spaces
from tqdm import trange


class CliffWalkingEnv(gym.Env):
    def __init__(self, epsilon=0.4, penalty=-25.0, reward=15.0, max_len=20):
        super().__init__()
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(4, ),
            dtype=np.int32
        ) # this is for maing the action space one-hot
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2, ),
            dtype=np.float32
        )
        self.max_len = 20
        self.epsilon = epsilon
        self.penalty = penalty
        self.reward = reward
        self.board = np.zeros([7, 5])

    def reset(self):
        self._step = 0
        self.already_done = False
        self.cur_pos = [0, 0]
        self.board = np.zeros([7, 5])
        self.board[self.cur_pos[0], self.cur_pos[1]] = 1
        return np.asarray(self.cur_pos)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()

        # if already terminal, absorb it
        if self.already_done:
            reward = 0.0
            info = {"absorb": True}
        else:
            # left: 0, right: 1, up: 2, down: 3
            self.board[self.cur_pos[0], self.cur_pos[1]] = 0
            fall = False

            # action of left and right will become up and down randomly
            if action in {0, 1}:
                eps = np.random.random()
                if eps > 1 - self.epsilon / 2:
                    action = 2
                elif eps > 1 - self.epsilon and eps <= 1 - self.epsilon / 2:
                    action = 3

            # handle the real action
            if action in {2, 3}:
                if action == 2:
                    next_y = self.cur_pos[1] + 1
                elif action == 3:
                    next_y = self.cur_pos[1] - 1
                if next_y not in {0, 1, 2}:
                    fall = True
                    # next_y = np.clip(next_y, 0, 2)
                self.cur_pos[1] = next_y
            elif action in {0, 1}:
                if action == 0:
                    next_x = self.cur_pos[0] - 1
                elif action == 1:
                    next_x = self.cur_pos[0] + 1
                if next_x not in {0, 1, 2, 3, 4}:
                    fall = True
                    # next_x = np.clip(next_x, 0, 4)
                self.cur_pos[0] = next_x

            self.board[self.cur_pos[0], self.cur_pos[1]] = 1

        # self._step += 1

        # done = False
            reward = -1
            info = {
                "success": False,
                "fall": False
            }
            if self.cur_pos[0] == 4 and self.cur_pos[1] == 0:
                reward += self.reward
                # done = True
                info["success"] = True
                self.already_done = True
            if fall:
                reward += self.penalty
                # done = True
                info["fall"] = True
                self.already_done = True

        self._step += 1
        if self._step >= 20:
            done = True
        else:
            done = False
        return np.asarray(self.cur_pos), reward, done, info

    def render(self):
        actual_board = np.concatenate([self.board[-1:, :], self.board[:5, :], self.board[-2:-1, :]], axis=0)
        actual_board = np.concatenate([actual_board[:, -1:], actual_board[:, :3], actual_board[:, -2:-1]], axis=1)
        for x in range(5):
            print("| ", end="")
            for y in actual_board[..., 4 - x]:
                print(f"{y} |", end="")
            print("")
        print("--------------------------")


class BlindWalker():
    def __init__(self, row):
        self.row = row

    def select_action(self, obs):
        col, row = obs[0], obs[1]
        if col == 0:
            if row < self.row:
                return 2
            elif row > self.row:
                return 3
        if col == 4:
            return 3
        return 1


class OptimalWalker():
    def __init__(self, *args, **kwargs):
        pass

    def select_action(self, obs):
        col, row = obs[0], obs[1]
        if col == 4:
            return 3
        if row > 1:
            return 3
        if row < 1:
            return 2
        return 1


class RandomWalker():
    def __init__(self, *args, **kwargs):
        pass

    def select_action(self, obs):
        return np.random.choice([0, 1, 2, 3])


class EpsilonOptimal():
    def __init__(self, eps):
        self.eps = eps
        self.random = RandomWalker()
        self.optimal = OptimalWalker()

    def select_action(self, obs):
        if np.random.random() < self.eps:
            return self.random.select_action(obs)
        else:
            return self.optimal.select_action(obs)


class HumanWalker():
    def __init__(self, *args, **kwargs):
        pass

    def select_action(self):
        c = input()
        return {
            "w": 2,
            "s": 3,
            "a": 0,
            "d": 1
        }.get(c)


def collect_data(env, agent, num_episodes):
    obss = []
    actions = []
    next_obss = []
    rewards = []
    terminals = []
    returns = []
    traj_returns = []
    traj_lengths = []
    for e in trange(num_episodes):
        ep_obss = []
        ep_actions = []
        ep_next_obss =[]
        ep_rewards = []
        ep_terminals = []

        obs = env.reset()
        done = False
        traj_return = 0
        traj_length = 0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            onehot_action = np.zeros(4, dtype=np.float32)
            onehot_action[action] = 1.0
            traj_return += reward
            traj_length += 1
            ep_obss.append(obs)
            ep_actions.append(onehot_action)
            ep_next_obss.append(next_obs)
            ep_rewards.append(reward)
            ep_terminals.append(done)

            obs = next_obs
        ep_returns_to_go = ep_rewards.copy()
        for i in range(len(ep_returns_to_go) - 2, -1, -1):
            ep_returns_to_go[i] = ep_returns_to_go[i] + ep_returns_to_go[i + 1]
        obss.append(np.stack(ep_obss, axis=0))
        actions.append(np.stack(ep_actions, axis=0))
        next_obss.append(np.stack(ep_next_obss, axis=0))
        rewards.append(np.stack(ep_rewards, axis=0))
        terminals.append(np.stack(ep_terminals, axis=0))
        returns.append(np.stack(ep_returns_to_go, axis=0))
        traj_returns.append(traj_return)
        traj_lengths.append(traj_length)

    obss = np.stack(obss, axis=0).astype(np.float32)
    actions = np.stack(actions, axis=0).astype(np.float32)
    next_obss = np.stack(next_obss, axis=0).astype(np.float32)
    rewards = np.asarray(rewards).astype(np.float32)
    terminals = np.asarray(terminals).astype(np.float32)
    returns = np.asarray(returns).astype(np.float32)
    traj_returns = np.asarray(traj_returns).astype(np.float32)
    traj_lengths = np.asarray(traj_lengths).astype(np.float32)
    return obss, actions, next_obss, rewards, terminals, returns, traj_returns, traj_lengths


def get_cliff_dataset(
    config={
        "random": 10000,
        "blind0": 50,
        "optimal": 50
    }
):
    np.random.seed(42)
    env = CliffWalkingEnv()
    agents = {
        "random": RandomWalker(),
        "blind0": BlindWalker(0),
        "blind1": BlindWalker(1),
        "blind2": BlindWalker(2),
        "optimal": OptimalWalker()
    }
    output = {}
    for cls, episode_num in config.items():
        if episode_num == 0:
            continue
        agent = agents[cls]
        obss, actions, next_obss, rewards, terminals, returns, traj_returns, traj_lengths = \
            collect_data(env, agent, episode_num)
        output[cls] = {
            "obs": obss,
            "action": actions,
            "next_obs": next_obss,
            "reward": rewards,
            "terminal": terminals,
            "return": returns,
            "traj_return": traj_returns,
            "traj_length": traj_lengths
        }
    return output


if __name__ == "__main__":
    env = CliffWalkingEnv()
    # agent = HumanWalker()
    # while True:
    #     ll = rr = 0
    #     _, done = env.reset(), False
    #     env.render()
    #     while not done:
    #         action = agent.select_action()
    #         _, r, done, info = env.step(action)
    #         env.render()
    #         ll += 1
    #         rr += r
    #         print(f"reward: {r}, info: {info}")
    #     print("total length: ", ll, "total reward: ", rr)

    # agent = RandomWalker()
    # dataset = get_cliff_dataset(config={
    #     "random": 10000
    # })
    # dataset = dataset["random"]
    agent = EpsilonOptimal(0.5)
    obss, actions, next_obss, rewards, terminals, returns, traj_returns, traj_lengths = collect_data(env, agent, 10000)
    dataset = {
        "obs": obss,
        "action": actions,
        "next_obs": next_obss,
        "reward": rewards,
        "terminal": terminals,
        "return": returns,
        "traj_return": traj_returns,
        "traj_length": traj_lengths
    }
    np.savez("./datasets/cliff/unlabel/eps_optimal_0.5-num10000.npz", **dataset)
