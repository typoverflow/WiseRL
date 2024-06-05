from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from wiserl.module.net.attention.base import BaseTransformer
from wiserl.module.net.attention.positional_encoding import get_pos_encoding
from wiserl.module.net.attention.bert import BERTBlock


class EncoderTransformer(BaseTransformer):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int,
        z_dim: int,
        num_layers: int,
        num_heads: int,
        attention_dropout: Optional[float]=0.1,
        residual_dropout: Optional[float]=0.1,
        embed_dropout: Optional[float]=0.1,
        seq_len: Optional[int]=1024
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.embed_timestep = nn.Embedding(seq_len, embed_dim)
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Linear(action_dim, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)


        self.blocks = nn.ModuleList([
            BERTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout
            ) for _ in range(num_layers)
        ])

        self.to_z = nn.Linear(embed_dim, z_dim)


    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: Optional[torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        key_padding_mask: Optional[torch.Tensor]=None,
    ):
        device = states.device
        B, L, *_ = states.shape

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        stacked_inputs = (
            torch.stack((state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(B, 2*L, self.embed_dim)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_mask = torch.stack((key_padding_mask, key_padding_mask), dim=1).permute(0, 2, 1).reshape(B, 2*L)

        for i, block in enumerate(self.blocks):
            stacked_inputs = block(stacked_inputs, key_padding_mask=stacked_mask.to(torch.bool))
        
        stacked_inputs = stacked_inputs.reshape(B,L,2,self.embed_dim,).permute(0, 2, 1, 3)
        stacked_inputs = stacked_inputs.sum(dim=2).sum(dim=1)

        z = self.to_z(stacked_inputs)
        return z
