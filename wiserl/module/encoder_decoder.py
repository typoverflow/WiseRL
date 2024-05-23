from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from wiserl.module.net.mlp import MLP

ModuleType = Type[nn.Module]


class MLPEncDec(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = [],
        deterministic: bool = True,
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        logstd_min: float = -20.0,
        logstd_max: float = 2.0,
    ):
        self.deterministic = deterministic
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=output_dim if self.deterministic else 2*output_dim,
            hidden_dims=hidden_dims,
            norm_layer=norm_layer
        )
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        if self.deterministic:
            return x
        else:
            mean, logstd = torch.chunk(x, 2, dim=-1)
            logstd = torch.clip(logstd, min=self.logstd_min, max=self.logstd_max)
            return mean, logstd

    def sample(self, x: torch.Tensor, deterministic: bool=False, return_mean_logstd: bool=False):
        if self.deterministic:
            return self.forward(x)
        else:
            mean, logstd = self.forward(x)
            dist = Normal(mean, logstd.exp())
            if deterministic:
                s, logprob = dist.mean(), None
            else:
                s = dist.rsample()
                logprob = dist.log_prob(s).sum(-1, keepdim=True)
            if return_mean_logstd:
                return s, logprob, mean, logstd
            else:
                return s, logprob


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask
        self.tokeys = nn.Linear(emb, emb*heads, bias=False)
        self.toqueries = nn.Linear(emb, emb*heads, bias=False)
        self.tovalues = nn.Linear(emb, emb*heads, bias=False)
        self.unifyheads = nn.Linear(heads*emb, emb)

    def forward(self, x, mask=None):
        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)
        keys = keys.transpose(1, 2).contiguous().view(b*h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b*h, t, e)
        values = values.transpose(1, 2).contiguous().view(b*h, t, e)
        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t, t)

        dot = torch.bmm(dot, values).view(b, h, t, e)
        dot = dot.transpose(1, 2).contiguous().view(b, t, h*e)
        return self.unifyheads(dot)


class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult*emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult*emb, emb)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        embed_dim,
        num_layers,
        num_heads,
        deterministic: bool = True,
        logstd_min: float = -20.0,
        logstd_max: float = 2.0,
    ):
        super().__init__()
        self.deterministic = deterministic
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max
        self.token_embedding = nn.Linear(input_dim, embed_dim)
        self.cls_embedding = nn.Parameter(torch.randn([embed_dim, ]), requires_grad=True)
        tblocks = []
        for i in range(num_layers):
            tblocks.append(TransformerBlock(emb=embed_dim, heads=num_heads))
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(embed_dim, 2*output_dim if not deterministic else output_dim)

    def forward(self, x: torch.Tensor):
        if tokens is not None:
            tokens = self.token_embedding(tokens)
            tokens = torch.concat([self.cls_embedding.repeat(tokens.shape[0], 1, 1), tokens], dim=1)
            b, t, e = tokens.size()
        else:
            tokens = self.cls_embedding.repeat(1, 1, 1)
            b, t, e = tokens.size()
        x = self.tblocks(x)[:, 0]
        x = self.toprobs(x)
        if self.deterministic:
            return x
        else:
            mean, logstd = torch

    def sample(self, x: torch.Tensor, deterministic: bool=False):
        if self.deterministic:
            return self.forward(x)
        else:
            mean, logstd = self.forward(x)
            dist = Normal(mean, logstd.exp())
            if deterministic:
                s, logprob = dist.mean(), None
            else:
                s = dist.rsample()
                logprob = dist.log_prob(s).sum(-1, keepdim=True)
            return s, logprob
