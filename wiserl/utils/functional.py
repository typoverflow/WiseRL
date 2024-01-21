import torch


def expectile_regression(pred, target, expectile):
    diff = target - pred
    return torch.where(diff > 0, expectile, 1-expectile) * (diff**2)

def discounted_cum_sum(seq, discount):
    seq = seq.copy()
    for t in reversed(range(len(seq)-1)):
        seq[t] += discount * seq[t+1]
    return seq

def biased_bce_with_logits(adv1, adv2, label, bias=1.0):
    # Apply the log-sum-exp trick.
    # y = 1 if we prefer x2 to x1
    # We need to implement the numerical stability trick.

    logit21 = adv2 - bias * adv1
    logit12 = adv1 - bias * adv2
    max21 = torch.clamp(-logit21, min=0, max=None)
    max12 = torch.clamp(-logit12, min=0, max=None)
    nlp21 = torch.log(torch.exp(-max21) + torch.exp(-logit21 - max21)) + max21
    nlp12 = torch.log(torch.exp(-max12) + torch.exp(-logit12 - max12)) + max12
    loss = label * nlp21 + (1 - label) * nlp12
    return loss
