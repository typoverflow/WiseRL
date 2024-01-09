from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

import torch
import torch.nn as nn

from src.module.net.mlp import MLP, EnsembleMLP

ModuleType = Type[nn.Module]

class Critic(nn.Module):
    """
    A vanilla critic module, which can be used as Q(s, a) or V(s).

    Parameters
    ----------
    input_dim :  The dimensions of input.
    output_dim :  The dimension of critic's output.
    device :  The device which the model runs on. Default is cpu.
    ***(any args of MLP or EnsembleMLP)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int=1,
        device: Union[str, int, torch.device] = "cpu",
        *,
        ensemble_size: int=1,
        hidden_dims: Sequence[int] = [],
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        dropout: Optional[Union[float, Sequence[float]]] = None,
        share_hidden_layer: Union[Sequence[bool], bool] = False,
    ) -> None:
        super().__init__()
        self.critic_type = "Critic"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.ensemble_size = ensemble_size

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if ensemble_size == 1:
            self.output_layer = MLP(
                input_dim = input_dim,
                output_dim = output_dim,
                hidden_dims = hidden_dims,
                norm_layer = norm_layer,
                activation = activation,
                dropout = dropout,
                device = device
            )
        elif isinstance(ensemble_size, int) and ensemble_size > 1:
            self.output_layer = EnsembleMLP(
                input_dim = input_dim,
                output_dim = output_dim,
                hidden_dims = hidden_dims,
                norm_layer = norm_layer,
                activation = activation,
                dropout = dropout,
                device = device,
                ensemble_size = ensemble_size,
                share_hidden_layer = share_hidden_layer
            )
        else:
            raise ValueError(f"ensemble size should be int >= 1.")

    def forward(self, obs: torch.Tensor, action: Optional[torch.Tensor]=None, *args, **kwargs) -> torch.Tensor:
        """Compute the Q-value (when action is given) or V-value (when action is None).

        Parameters
        ----------
        obs :  The observation, should be torch.Tensor.
        action :  The action, should be torch.Tensor.

        Returns
        -------
        torch.Tensor :  Q(s, a) or V(s).
        """
        if action is not None:
            obs = torch.cat([obs, action], dim=-1)
        return self.output_layer(obs)


class DoubleCritic(nn.Module):
    """
    Double Critic module, which consists of two (or more) independent Critic modules, can be used to implement the popular Double-Q trick.

    Notes
    -----
    1. Except for DoubleCritic. As we are handling ensemble explicitly with `critic_num`, you should not
      specify `ensemble_size` or `share_hidden_layer` for this module any more.

    Parameters
    ----------
    input_dim :  The dimensions of input.
    output_dim :  The dimension of critic's output.
    critic_num :  The num of critics. Default is 2.
    reduce :  A unary function which specifies how to aggregate the output values. Default is torch.min along the 0 dimension.
    device :  The device which the model runs on. Default is cpu.
    ***(any args of MLP)
    """
    _reduce_fn_ = {
        "min": lambda x: torch.min(x, dim=0)[0],
        "max": lambda x: torch.max(x, dim=0)[0],
        "mean": lambda x: torch.mean(x, dim=0)
    }
    def __init__(
        self,
        input_dim: int,
        output_dim: int=1,
        critic_num: int=2,
        reduce: Union[str, Callable]="min",
        device: Union[str, int, torch.device]="cpu",
        *,
        hidden_dims: Sequence[int] = [],
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        dropout: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        super().__init__()
        self.critic_type = "DoubleCritic"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.critic_num = critic_num

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.output_layer = EnsembleMLP(
            input_dim = input_dim,
            output_dim = output_dim,
            ensemble_size = critic_num,
            hidden_dims = hidden_dims,
            norm_layer = norm_layer,
            activation = activation,
            dropout = dropout,
            share_hidden_layer = False,
            device = device
        )

        if isinstance(reduce, str):
            self.reduce = self._reduce_fn_[reduce]
        else:
            self.reduce = reduce

    def forward(self, obs: torch.Tensor, action: Optional[torch.Tensor]=None, reduce: bool=True, *args, **kwargs) -> torch.Tensor:
        """Compute the Q-value (when action is given) or V-value (when action is None), and aggregate them with the pre-defined reduce method.
        If `reduce` is False, then no aggregation will be performed.

        Parameters
        ----------
        obs :  The observation, should be torch.Tensor.
        action :  The action, should be torch.Tensor.
        reduce :  Whether to aggregate the outputs or not. Default is True.

        Returns
        -------
        torch.Tensor :  Q(s, a) or V(s).
        """
        if action is not None:
            obs = torch.cat([obs, action], dim=-1)
        output = self.output_layer(obs)
        if reduce:
            return self.reduce(output)
        else:
            return output
