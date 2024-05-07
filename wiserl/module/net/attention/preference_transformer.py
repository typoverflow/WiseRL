from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from wiserl.module.net.attention.base import BaseTransformer
from wiserl.module.net.attention.gpt2 import GPT2
from wiserl.module.net.attention.positional_encoding import get_pos_encoding


class PreferenceTransformer(BaseTransformer):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int,
        pref_embed_dim: int,
        num_layers: int,
        seq_len: int,
        num_heads: int=1,
        attention_dropout: Optional[float]=0.1,
        residual_dropout: Optional[float]=0.1,
        embed_dropout: Optional[float]=0.1,
        pos_encoding: str="embed",
        use_weighted_sum: bool=True
    ) -> None:
        super().__init__()
        self.backbone = GPT2(
            input_dim=embed_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            causal=True,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            embed_dropout=embed_dropout,
            pos_encoding="none",
            seq_len=0
        )
        self.pos_encoding = get_pos_encoding(pos_encoding, embed_dim, seq_len)
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.act_embed = nn.Linear(action_dim, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)
        self.use_weighted_sum = use_weighted_sum
        # additional layers
        self.pref_embed_dim = pref_embed_dim
        if use_weighted_sum:
            self.to_kqv = nn.Linear(embed_dim, 2*pref_embed_dim+1, bias=False)
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(embed_dim, pref_embed_dim),
                nn.GELU(),
                nn.Linear(pref_embed_dim, 1)
            )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        key_padding_mask: Optional[torch.Tensor]=None
    ):
        B, L, *_ = states.shape
        state_embedding = self.pos_encoding(self.obs_embed(states), timesteps)
        action_embedding = self.pos_encoding(self.act_embed(actions), timesteps)
        stacked_input = torch.stack([state_embedding, action_embedding], dim=2).reshape(B, 2*L, state_embedding.shape[-1])
        stacked_input = self.embed_ln(stacked_input)
        if key_padding_mask is not None:
            key_padding_mask = torch.stack([key_padding_mask, key_padding_mask], dim=2).reshape(B, 2*L)
        out = self.backbone(
            inputs=stacked_input,
            timesteps=None,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            do_embedding=False
        )
        out = out[:, 1::2] # select the action token output

        if self.use_weighted_sum:
            out = self.to_kqv(out)
            query, key, value = out.split([self.pref_embed_dim, self.pref_embed_dim, 1], dim=2)
            query = query / (self.pref_embed_dim ** 0.25)
            key = key / (self.pref_embed_dim ** 0.25)
            attention_weights = torch.bmm(query, key.transpose(1, 2))
            attention_weights = torch.softmax(attention_weights, dim=2)
            out = torch.bmm(attention_weights, value)
            return value, out
        else:
            value = self.output_layer(out)
            return value, value
