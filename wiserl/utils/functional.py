import torch


def expectile_regression(pred, target, expectile):
    diff = target - pred
    return torch.where(diff > 0, expectile, 1-expectile) * (diff**2)

def discounted_cum_sum(seq, discount):
    seq = seq.copy()
    for t in reversed(range(len(seq)-1)):
        seq[t] += discount * seq[t+1]
    return seq
