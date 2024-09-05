import math
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch.distributions import Normal, OneHotCategorical
from torch.distributions.utils import clamp_probs

class TanhNormal(Normal):
    def __init__(self,
                 loc: torch.Tensor,
                 scale: torch.Tensor,
                 ):
        super().__init__(loc, scale)
        self.epsilon = np.finfo(np.float32).eps.item()

    def log_prob(self,
                 value: torch.Tensor,
                 pre_tanh_value: bool=False,
                 ):
        if not pre_tanh_value:
            pre_value = torch.clip(value, -1.0+self.epsilon, 1.0-self.epsilon)
            pre_value = 0.5 * (pre_value.log1p() - (-pre_value).log1p())
        else:
            pre_value = value
            value = torch.tanh(pre_value)
        return super().log_prob(pre_value) - 2*(math.log(2.0) - pre_value - torch.nn.functional.softplus(-2 * pre_value))

    def sample(self, sample_shape: Union[Sequence[int], int]=torch.Size([]), return_raw: bool=False):
        z = super().sample(sample_shape)
        return (torch.tanh(z), z) if return_raw else torch.tanh(z)

    def rsample(self, sample_shape: Union[Sequence[int], int]=torch.Size([]), return_raw: bool=False):
        z = super().rsample(sample_shape)
        return (torch.tanh(z), z) if return_raw else torch.tanh(z)

    def entropy(self):
        return super().entropy()

    @property
    def tanh_mean(self):
        return torch.tanh(self.mean)


class OneHotCategoricalSTGumbelSoftmax(OneHotCategorical):
    r"""
    Creates a reparameterizable :class:`OneHotCategorical` distribution based on the straight-
    through gumbel-softmax estimator from [1].

    [1] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al, 2017)
    """
    has_rsample = True
    
    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        super().__init__(probs, logits, validate_args=validate_args)
        self.temperature = temperature
    
    def gumbel_softmax_sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        uniforms = clamp_probs(
            torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)
        )
        gumbels = -((-(uniforms.log())).log())
        y = self.logits + gumbels
        return torch.nn.functional.softmax(y / self.temperature, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        samples = self.sample(sample_shape)
        gumbel_softmax_samples = self.gumbel_softmax_sample(sample_shape)
        return samples + (gumbel_softmax_samples - gumbel_softmax_samples.detach())
