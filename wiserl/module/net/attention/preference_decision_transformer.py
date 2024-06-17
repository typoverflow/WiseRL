from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from wiserl.module.net.attention.base import BaseTransformer
from wiserl.module.net.attention.gpt2 import GPTBlock
from wiserl.module.net.attention.positional_encoding import get_pos_encoding

class PreferenceDecisionTransformer(BaseTransformer):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int,
        z_dim: int,
        num_layers: int,
        num_heads: int,
        action_tanh: bool=True,
        causal: bool=True,
        attention_dropout: Optional[float]=0.1,
        residual_dropout: Optional[float]=0.1,
        embed_dropout: Optional[float]=0.1,
        seq_len: Optional[int]=1024,
        max_ep_len: Optional[int]=1024,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.embed_timestep = nn.Embedding(max_ep_len, embed_dim)
        self.embed_z = nn.Linear(z_dim, embed_dim)
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Linear(action_dim, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)


        self.blocks = nn.ModuleList([
            GPTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout
            ) for _ in range(num_layers)
        ])
        # OPPO only predicts action

        self.predict_state = nn.Linear(embed_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(embed_dim, action_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(embed_dim, 1)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.causal = causal
        self.seq_len = seq_len
        self.max_ep_len = max_ep_len


    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        zs: torch.Tensor,
        timesteps: Optional[torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        key_padding_mask: Optional[torch.Tensor]=None,
    ):
        device = states.device
        B, L, *_ = states.shape

        if self.causal:
            mask = ~torch.tril(torch.ones([3*L, 3*L])).to(torch.bool).to(device)
        else:
            mask = torch.zeros([3*L, 3*L]).to(torch.bool).to(device)
        if attention_mask is not None:
            mask = torch.bitwise_or(attention_mask.to(torch.bool), mask)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        z_embeddings = self.embed_z(zs)
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        z_embeddings = z_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (z_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3*L, self.embed_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_mask = torch.stack((key_padding_mask, key_padding_mask, key_padding_mask), dim=1).permute(0, 2, 1).reshape(B, 3*L)

        for i, block in enumerate(self.blocks):
            stacked_inputs = block(stacked_inputs, attention_mask=mask, key_padding_mask=stacked_mask.to(torch.bool))
        stacked_inputs = stacked_inputs.reshape(B, L, 3, self.embed_dim).permute(0, 2, 1, 3)
        
        # get predictions
        return_preds = self.predict_return(stacked_inputs[:,2])
        state_preds = self.predict_state(stacked_inputs[:,0]) 
        action_preds = self.predict_action(stacked_inputs[:,1])
        return state_preds, action_preds, return_preds
    
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        zs: torch.Tensor,
        timesteps: Optional[torch.Tensor]=None,
        mask: Optional[torch.Tensor]=None
    ):
        device = states.device

        zs = zs = zs.reshape(1, 1, self.z_dim)
        B, L, _ = states.shape
        zs = torch.cat([zs, torch.zeros((1, self.seq_len-1, self.z_dim), device=device)],dim=1)
        
        state_preds, action_preds, return_preds = self.forward(states, actions, zs, timesteps, key_padding_mask=mask)
        return action_preds[0,-1].cpu().numpy()



