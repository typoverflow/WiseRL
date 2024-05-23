import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from wiserl.module.net.mlp import MLP

device = "cuda"

def generate_demos(dataset, num_trajs, steps=0):
    N = dataset["obs"].shape[0]
    idxs = np.random.choice(N, size=(num_trajs, ), replace=False)
    return dataset["obs"][idxs], dataset["action"][idxs], dataset["reward"][idxs], dataset["traj_return"][idxs]

def create_training_data(obss, actions, returns, rewards, num_trajs):
    training_obss = []
    training_actions = []
    training_labels = []
    training_rewards = []
    # training_returns = []
    num_demos = len(obss)
    for i in range(num_trajs):
        ti = tj = 0
        while (ti == tj):
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        if returns[ti] > returns[tj]:
            label = 0
        else:
            label = 1
        training_obss.append(np.stack([obss[ti], obss[tj]], axis=0))
        training_actions.append(np.stack([actions[ti], actions[tj]], axis=0))
        training_labels.append(label)
        training_rewards.append(np.stack([rewards[ti], rewards[tj]], axis=0))
    return np.stack(training_obss, axis=0), \
        np.stack(training_actions, axis=0), \
        np.stack(training_labels, axis=0), \
        np.stack(training_rewards, axis=0)

def learn_reward(reward_network, optimizer, training_obss, training_actions, training_labels, num_iter, l1_reg):
    loss_criterion = nn.CrossEntropyLoss()

    training_obss = np.asarray(training_obss)
    training_actions = np.asarray(training_actions)
    training_labels = np.asarray(training_labels)
    N = training_obss.shape[0]
    idxs = np.arange(N)
    loss_list = []
    for iter in range(num_iter):
        np.random.shuffle(idxs)
        batch_size = 8
        loss_sub_list = []
        for i in range((N-1)//batch_size+1):
            idx = idxs[i*batch_size: min((i+1)*batch_size, N)]
            obss = torch.from_numpy(training_obss[idx]).to(device)
            actions = torch.from_numpy(training_actions[idx]).to(device)
            labels = torch.from_numpy(training_labels[idx]).to(device)
            obs_1, obs_2 = obss[:, 0], obss[:, 1]
            action_1, action_2 = actions[:, 0], actions[:, 1]
            optimizer.zero_grad()
            logit1 = reward_network(obs_1, action_1).sum(1)
            logit2 = reward_network(obs_2, action_2).sum(1)
            logits = logit2-logit1
            reward_loss = loss_criterion(logits.squeeze(-1), labels.float())
            reg_loss = (logit1.abs() + logit2.abs()).mean()
            loss = reward_loss + l1_reg * reg_loss
            loss.backward()
            optimizer.step()
            loss_sub_list.append(loss.item())
        loss_list.append(np.mean(loss_sub_list))
    return loss_list


class RewardNet(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.mlp = MLP(
            input_dim, 1, hidden_dims
        )

    def forward(self, obs, action):
        x = torch.concat([obs, action], dim=-1)
        x = self.mlp(x)
        return x


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='eps_optimal_0.5-num10000.npz', help='Select the environment name to run, i.e. maze2d-medium-dense-v1')
    parser.add_argument('--initial_pairs', default = 50, type=int, help="initial number of pairs of trajectories used to train the reward models")
    parser.add_argument('--num_snippets', default = 0, type = int, help = "number of short subtrajectories to sample")
    parser.add_argument('--voi', default='dis', help='Choose between infogain, disagreement, or random')
    parser.add_argument('--num_rounds', default = 60, type = int, help = "number of rounds of active querying")
    parser.add_argument('--num_queries', default = 5, type = int, help = "number of queries per round of active querying")
    parser.add_argument('--num_iter', default = 50, type = int, help = "number of iteration of initial data")
    parser.add_argument('--retrain_num_iter', default = 10, type = int, help = "number of training iteration after one round of active querying")
    parser.add_argument('--num_ensembles', default = 7, type = int, help = "number of ensemble of members")
    parser.add_argument('--seed', default = 0, type = int, help = "random seed")
    parser.add_argument('--beta', default = 10, type = int, help = "beta as a measure of confidence for info gain")

    args = parser.parse_args()

    # torch rng
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    env_name = args.env_name
    # data = np.load(os.path.join(f"./datasets/cliff/pref/eps_optimal_0.5-num300.npz"))
    data = np.load(os.path.join(f"./datasets/cliff/unlabel/{args.env_name}"))
    dataset = {}
    for k, v in data.items():
        dataset[k] = np.asarray(v)
    # action = dataset["action"]
    # action_shape = action.shape
    # onehot_action = np.eye(4).astype(np.float32)
    # onehot_action = onehot_action[action.astype(np.int32), :]
    # dataset["action"] = onehot_action
    input_dim = 6

    initial_pairs = args.initial_pairs
    num_snippets = args.num_snippets
    num_iter = args.num_iter
    retrain_num_iter = args.retrain_num_iter
    num_queries = args.num_queries
    voi = args.voi
    num_rounds = args.num_rounds
    num_ensembles = args.num_ensembles
    beta = args.beta

    lr = 0.0005
    weight_decay = 0.0
    l1_reg = 0.0
    stochastic = True

    demo_obss_list = []
    demo_actions_list = []
    returns_list = []
    reward_list = []
    models_list = []
    training_obss_list = []
    training_actions_list = []
    training_labels_list = []
    training_rewards_list = []
    models_list = []

    for seed in range(num_ensembles):
        torch.manual_seed(seed)
        np.random.seed(seed)
        demo_obss, demo_actions, learning_rewards, learning_returns = generate_demos(dataset, initial_pairs)
        demo_obss_list.append(demo_obss)
        demo_actions_list.append(demo_actions)
        returns_list.append(learning_returns)
        reward_list.append(learning_rewards)

        training_obss, training_actions, training_labels, training_rewards = create_training_data(demo_obss, demo_actions, learning_returns, learning_rewards, initial_pairs)

        training_obss_list.append(training_obss)
        training_actions_list.append(training_actions)
        training_labels_list.append(training_labels)
        training_rewards_list.append(training_rewards)

        reward_net = RewardNet(input_dim, [256, 256]).to(device)
        optimizer = optim.Adam(reward_net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_list = learn_reward(reward_net, optimizer, training_obss, training_actions, training_labels, num_iter, l1_reg)
        models_list.append(reward_net)
        print(f"Pretrain, model {seed}: loss {loss_list}")

    large_num_trajs = int(dataset["obs"].shape[0])
    large_num_pairs = large_num_trajs
    large_demo_obss, large_demo_actions, large_demo_rewards, large_demo_returns = generate_demos(dataset, large_num_trajs)
    large_training_obss, large_training_actions, large_training_labels, large_training_rewards = create_training_data(
        large_demo_obss, large_demo_actions, large_demo_returns, large_demo_rewards, large_num_pairs
    )

    with torch.no_grad():
        batch_size = 256
        acc_list = []
        for seed in range(num_ensembles):
            net = models_list[seed]
            pred_list = []
            for i in range((large_num_pairs-1) // batch_size + 1):
                idxs = np.arange(i*batch_size, min((i+1)*batch_size, large_num_pairs))
                obss = large_training_obss[idxs]
                actions = large_training_actions[idxs]
                obs1 = torch.from_numpy(obss[:, 0]).to(device)
                obs2 = torch.from_numpy(obss[:, 1]).to(device)
                action1 = torch.from_numpy(actions[:, 0]).to(device)
                action2 = torch.from_numpy(actions[:, 1]).to(device)
                labels = torch.from_numpy(large_training_labels[idxs]).to(device)

                logit1 = net(obs1, action1).sum(1)
                logit2 = net(obs2, action2).sum(1)
                logits = torch.concat([logit1, logit2], dim=-1)
                pred_prob = torch.softmax(logits, dim=-1)
                pred_label = torch.argmax(logits, dim=-1)
                pred_list.append(pred_label.detach().cpu().numpy())
            pred_list = np.concatenate(pred_list)
            label_list = large_training_labels
            acc_list.append((pred_list == label_list).mean())
        print(f"Round {-1}: acc {acc_list}, labeled size: {training_obss_list[0].shape[0]}")

    # query and retrain
    for round in range(num_rounds):
        # calculate the information gain
        batch_size = 256
        var_list = []
        infogain_list = []
        with torch.no_grad():
            for i in range((large_num_pairs-1) // batch_size + 1):
                idxs = np.arange(i*batch_size, min((i+1)*batch_size, large_num_pairs))
                obss = large_training_obss[idxs]
                actions = large_training_actions[idxs]
                obs1 = torch.from_numpy(obss[:, 0]).to(device)
                obs2 = torch.from_numpy(obss[:, 1]).to(device)
                action1 = torch.from_numpy(actions[:, 0]).to(device)
                action2 = torch.from_numpy(actions[:, 1]).to(device)
                labels = torch.from_numpy(large_training_labels[idxs]).to(device)

                pred_prob_list = []
                pred_label_list = []
                for net in models_list:
                    logit1 = net(obs1, action1).sum(1)
                    logit2 = net(obs2, action2).sum(1)
                    logits = torch.concat([logit1, logit2], dim=-1)
                    pred_prob = torch.softmax(logits, dim=-1)
                    pred_label = torch.argmax(logits, dim=-1)
                    pred_prob_list.append(pred_prob)
                    pred_label_list.append(pred_label)
                pred_prob_list = torch.stack(pred_prob_list, dim=1)
                pred_label_list = torch.stack(pred_label_list, dim=1)

                var = pred_label_list.float().mean(dim=1) * (1. - pred_label_list.float().mean(dim=1))
                entropy = - torch.sum(pred_prob_list * torch.log2(pred_prob_list), dim=-1)
                ave_ps = torch.mean(pred_prob_list, dim=1)
                H1 = - torch.sum(ave_ps * torch.log2(ave_ps), dim=-1)
                H2 = torch.mean(entropy, dim=1)
                infogain = H1 - H2

                var_list.append(var.detach().cpu().numpy())
                infogain_list.append(infogain.detach().cpu().numpy())

        var_list = np.concatenate(var_list)
        infogain_list = np.concatenate(infogain_list)
        if voi == "dis":
            query_idx = var_list.argsort()[-num_queries:][::-1]
        elif voi == "info":
            query_idx = infogain_list.argsort()[-num_queries][::-1]
        else:
            query_idx = np.random.choice(len(large_training_obss), (num_queries, ))

        for i in range(len(training_obss_list)):
            for idx in query_idx:
                training_obss_list[i] = np.concatenate([training_obss_list[i], large_training_obss[idx][None, ...]], axis=0)
                training_actions_list[i] = np.concatenate([training_actions_list[i], large_training_actions[idx][None, ...]], axis=0)
                training_labels_list[i] = np.concatenate([training_labels_list[i], large_training_labels[idx][None, ...]], axis=0)
                training_rewards_list[i] = np.concatenate([training_rewards_list[i], large_training_rewards[idx][None, ...]], axis=0)

        # retrain the models
        for seed in range(num_ensembles):
            torch.manual_seed(seed)
            np.random.seed(seed)
            training_obss, training_actions, training_labels = training_obss_list[seed], training_actions_list[seed], training_labels_list[seed]
            reward_net = models_list[seed]
            optimizer = optim.Adam(reward_net.parameters(), lr=lr, weight_decay=weight_decay)
            loss_list = learn_reward(reward_net, optimizer, training_obss, training_actions, training_labels, retrain_num_iter, l1_reg)
            print(f"Train Round {round}, model {seed}: loss {loss_list}")

        with torch.no_grad():
            acc_list = []
            for seed in range(num_ensembles):
                net = models_list[seed]
                pred_list = []
                for i in range((large_num_pairs-1) // batch_size + 1):
                    idxs = np.arange(i*batch_size, min((i+1)*batch_size, large_num_pairs))
                    obss = large_training_obss[idxs]
                    actions = large_training_actions[idxs]
                    obs1 = torch.from_numpy(obss[:, 0]).to(device)
                    obs2 = torch.from_numpy(obss[:, 1]).to(device)
                    action1 = torch.from_numpy(actions[:, 0]).to(device)
                    action2 = torch.from_numpy(actions[:, 1]).to(device)
                    labels = torch.from_numpy(large_training_labels[idxs]).to(device)

                    logit1 = net(obs1, action1).sum(1)
                    logit2 = net(obs2, action2).sum(1)
                    logits = torch.concat([logit1, logit2], dim=-1)
                    pred_prob = torch.softmax(logits, dim=-1)
                    pred_label = torch.argmax(logits, dim=-1)
                    pred_list.append(pred_label.detach().cpu().numpy())
                pred_list = np.concatenate(pred_list)
                label_list = large_training_labels
                acc_list.append((pred_list == label_list).mean())
            print(f"Round {round}: acc {acc_list}, labeled size: {training_obss_list[0].shape}")

    save_obss = training_obss_list[0][-300:]
    save_actions = training_actions_list[0][-300:]
    save_labels = training_labels_list[0][-300:]
    save_rewards = training_rewards_list[0][-300:]
    np.savez(f"./datasets/cliff/pref/eps_optimal_0.5-num300.npz",
             obs_1=save_obss[:, 0],
             obs_2=save_obss[:, 1],
             action_1=save_actions[:, 0],
             action_2=save_actions[:, 1],
             label=save_labels,
             reward_1=save_rewards[:, 0],
             reward_2=save_rewards[:, 1])
