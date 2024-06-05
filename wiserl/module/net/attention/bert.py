from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from wiserl.module.net.attention.base import BaseTransformer
from wiserl.module.net.attention.positional_encoding import get_pos_encoding

# the bert block from huggingface version
# different from the original bert block
class BERTBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        backbone_dim: Optional[int]=None,
        pre_norm: bool=False,
        attention_dropout: Optional[float]=None,
        residual_dropout: Optional[float]=None
    ) -> None:
        super().__init__()
        if backbone_dim is None:
            backbone_dim = 4 * embed_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.ff1 =  nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(residual_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff2 = nn.Sequential(
            nn.Linear(embed_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(residual_dropout)
        self.pre_norm = pre_norm

    def forward(
        self,
        input: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        key_padding_mask: Optional[torch.Tensor]=None,
    ):
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(torch.bool)

        residual = input
        if self.pre_norm:
            residual = residual + self._sa_block(self.norm1(input), attention_mask, key_padding_mask)
            residual = residual + self._ff_block(self.norm2(residual))
        else:
            residual = self.norm1(residual + self._sa_block(input, attention_mask, key_padding_mask))
            residual = self.norm2(residual + self._ff_block(residual))
        return residual

    def _sa_block(self, inputs, attention_mask, key_padding_mask):
        inputs = self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            need_weights=False,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask
        )[0]
        return self.dropout1(self.ff1(inputs))

    def _ff_block(self, input):
        return self.dropout2(self.ff2(input))

