import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class AmbisonicsEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.direction_embedding = nn.Sequential(
            nn.Linear(3, hparams.d_model),
            nn.GELU(),
            nn.Linear(hparams.d_model, hparams.d_model),
        )
        self.energy_map_projection = nn.Sequential(
            nn.Linear(49, hparams.d_model),
            nn.GELU(),
            nn.Linear(hparams.d_model, hparams.d_model),
        )
        
        embed_dim = hparams.d_model
        
        self.clip_projection = nn.Linear(hparams.d_clip, embed_dim)

        self.embed_positions = nn.Embedding(
            hparams.max_source_length,
            embed_dim,
        )
        # self.embed_spatial = nn.Embedding(4, embed_dim)
        
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.gradient_checkpointing = False
    
    def forward(
        self,
        direction: Optional[torch.FloatTensor] = None,
        energy_map: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        # retrieve inputs_embeds
        if inputs_embeds is None:
            raise ValueError("You have to specify inputs_embeds")
        # print(f'{inputs_embeds.shape =}')[17, 20, 512]
        bsz, seq, dim = inputs_embeds.shape
        
        # Prepare position ids
        position_ids = torch.arange(seq, device=inputs_embeds.device)
        # Prepare input embeddings
        inputs_embeds = self.clip_projection(inputs_embeds)
        if energy_map is not None:
            inputs_embeds = inputs_embeds + self.energy_map_projection(energy_map)
        
        embed_pos = self.embed_positions(position_ids)
        inputs_embeds = inputs_embeds + embed_pos
        
        if direction is not None:
            inputs_embeds = inputs_embeds + self.direction_embedding(
                direction
            ).unsqueeze(1)
        return inputs_embeds