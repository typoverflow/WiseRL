from math import e
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from wiserl.module.net.attention.base import BaseTransformer
from wiserl.module.net.attention.gpt2 import GPT2
from wiserl.module.net.attention.positional_encoding import get_pos_encoding


class PreferenceDiscriminator(BaseTransformer):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int,
        num_layers: int,
        seq_len: int,
        num_heads: int=1,
        attention_dropout: Optional[float]=0.1,
        residual_dropout: Optional[float]=0.1,
        embed_dropout: Optional[float]=0.1,
        pos_encoding: str="embed",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
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
        self.pos_encoding = get_pos_encoding(pos_encoding, embed_dim, 2*(seq_len+1))
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.act_embed = nn.Linear(action_dim, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)
        self.obs_head = nn.Linear(embed_dim, obs_dim)
        self.act_head = nn.Linear(embed_dim, action_dim)
        self.discriminator = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def _forward(
        self,
        observations_1: torch.Tensor,
        actions_1: torch.Tensor,
        observations_2: torch.Tensor,
        actions_2: torch.Tensor,
        timesteps: torch.Tensor=None,
        attention_mask: Optional[torch.Tensor]=None,
        key_padding_mask: Optional[torch.Tensor]=None,
    ):
        observations = torch.cat([observations_1, observations_2], dim=1)
        actions = torch.cat([actions_1, actions_2], dim=1)
        B, L, *_ = observations.shape
        state_embedding = self.pos_encoding(self.obs_embed(observations), timesteps)
        action_embedding = self.pos_encoding(self.act_embed(actions), timesteps)
        stacked_input = torch.stack([state_embedding, action_embedding], dim=2).reshape(B, 2*L, state_embedding.shape[-1])
        stacked_input1, stacked_input2 = torch.split(stacked_input, 2*observations_1.shape[1], dim=1)
        # insert sep between two sequences
        sep_embedding = torch.zeros(self.embed_dim, device=observations_1.device).repeat(B, 2, 1)
        stacked_input = torch.cat([stacked_input1, sep_embedding, stacked_input2, sep_embedding], dim=1)
        stacked_input = self.embed_ln(stacked_input)
        if key_padding_mask is not None:
            key_padding_mask = torch.stack([key_padding_mask, key_padding_mask], dim=2).reshape(B, 2*L)
        out, attentions = self.backbone(
            inputs=stacked_input,
            timesteps=None,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            do_embedding=False,
            output_attentions=True,
        )
        return out, attentions


    def forward(
        self,
        observations_1: torch.Tensor,
        actions_1: torch.Tensor,
        observations_2: torch.Tensor,
        actions_2: torch.Tensor,
        timesteps: torch.Tensor=None,
        attention_mask: Optional[torch.Tensor]=None,
        key_padding_mask: Optional[torch.Tensor]=None,
    ):
        out, attentions = self._forward(observations_1, actions_1, observations_2, actions_2, timesteps, attention_mask, key_padding_mask)
        pred_act = self.act_head(out[:, ::2])
        pred_obs = self.obs_head(out[:, 1::2])  # o[:t] + a[:t] -> s[t+1]
        
        pred_act1, pred_act2 = torch.split(pred_act, observations_1.shape[1] + 1, dim=1)
        pred_act1, pred_act2 = pred_act1[:, :-1], pred_act2[:, :-1]
        pred_obs1, pred_obs2 = torch.split(pred_obs, observations_1.shape[1] + 1, dim=1)
        pred_obs1, pred_obs2 = pred_obs1[:, :-1], pred_obs2[:, :-1]
        
        pred_label = self.discriminator(out[:, -1])
        
        return pred_obs1, pred_act1, pred_obs2, pred_act2, pred_label, attentions
