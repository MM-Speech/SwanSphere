"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding

from modules.tts.dit.f5_modules import (
    TimestepEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
)

class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim=1024,
        depth=22,
        heads=16,
        dim_head=64,
        dropout=0.0,
        ff_mult=4,
        mel_dim=24,
        conv_layers=0,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embed.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.time_mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x, time, mask):  
        '''
            x: float["b n d"],  # nosied input audio
            time: float["b"] | float[""],  # time step
            mask: bool["b n"] | None = None, # attention mask
        '''
        batch, seq_len = x.shape[0], x.shape[1]
        t = self.time_embed(time)
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output