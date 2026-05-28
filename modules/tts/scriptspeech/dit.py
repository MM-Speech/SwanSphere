import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import attrdict
import torchdiffeq

from modules.tts.ar_dur.commons.align_ops import expand_states
from modules.tts.llama_dit.llama import LLaMa
from utils.nn.seq_utils import sequence_mask, add_prefix_nd, remove_prefix

@dataclass
class ModelArgs:
    # text
    vocab_size: int = None
    text_dim: int = 1024
    text_inject_method: str = 'left-prefill'     # left-prefill | expand-prefill | concat

    # audio
    audio_vocab_size: int = None
    audio_tokenizer: str = 'glm4v'
    
    # llama
    encoder_dim: int = 1024
    encoder_n_layers: int = 24
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = None
    mlp_extend: float = None
    max_seq_len: int = 16384
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = 4
    use_causal_attn: bool = False

    in_channels: int = 16
    out_channels: int = 16

    # diffusion
    cfg_mask_text_token: int = None

    # trainging
    do_checkpoint: bool = False


class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp

        self.encoder = LLaMa(hp)

        self.prenet = nn.Linear(hp.in_channels, hp.encoder_dim)
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels)

        self.ctx_mask_proj = nn.Linear(1, hp.encoder_dim)
        self.ctx_proj = nn.Linear(hp.in_channels, hp.encoder_dim)
        self.audio_token_proj = nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=3, padding='same')
        
        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)

        from modules.flow_matching.llama import LLaMa as LLaMaSmall,  ModelArgs as ModelArgsSmall
        self.text_embedder = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        self.text_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.text_dim,
            n_layers=6, n_heads=8
        ))

        if hp.audio_tokenizer == 'glm4v':
            self.audio_token_embed = nn.Embedding(hp.audio_vocab_size, hp.encoder_dim)
            self.audio_token_upsampler = nn.Upsample(scale_factor=2, mode='nearest')
    
        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.hp.encoder_n_layers)
            )
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.hp.encoder_n_layers)
            )
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.f5_time_embed.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.f5_time_embed.time_mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.encoder.layers:
            nn.init.constant_(block.attention_norm.linear.weight, 0)
            nn.init.constant_(block.attention_norm.linear.bias, 0)
        
        # Zero-out output layers:
        nn.init.constant_(self.encoder.norm.linear.weight, 0)
        nn.init.constant_(self.encoder.norm.linear.bias, 0)
        nn.init.constant_(self.encoder.out_proj.weight, 0)
        nn.init.constant_(self.encoder.out_proj.bias, 0)
        
    def forward(self, inputs, sigmas=None, x_noisy=None):
        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask
        bsz, device = ctx_feature.size(0), ctx_feature.device

        x = inputs['lat']
        x_mask = sequence_mask(inputs['lat_lens'])
    
        # Here, x is x1 in CFM
        x0 = torch.randn_like(x)
        t = self.flow_matcher.time_sampler.sample([x0.shape[0]], x0.device).type_as(x0)
        xt = t[:, None, None] * x + (1 - t[:, None, None]) * x0
        ut = x - x0
        # t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x)
        
        # define noisy_input and target
        t = t.bfloat16()
        x_noisy = (xt * (1 - ctx_mask)).bfloat16()
        target = ut

        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        x_txt = torch.full((bsz, x.shape[1]), self.hp.cfg_mask_text_token).to(txt_tokens)
        if self.hp.text_inject_method == 'left-prefill':
            x_txt[sequence_mask(txt_mask.sum(1), x.shape[1])] = txt_tokens[txt_mask]
            x_txt = self.text_encoder(x=self.text_embedder(x_txt), attn_mask=x_mask)   # [B, T, C]
        elif self.hp.text_inject_method == 'expand-prefill':
            txt_tokens[~txt_mask] = self.hp.cfg_mask_text_token
            expand_mult = x.shape[1] // txt_tokens.shape[1]
            x_txt[:, :txt_tokens.shape[1]*expand_mult:expand_mult] = txt_tokens
            x_txt = self.text_encoder(x=self.text_embedder(x_txt), attn_mask=x_mask)   # [B, T, C]
        elif self.hp.text_inject_method == 'concat':
            txt_tokens[~txt_mask] = self.hp.cfg_mask_text_token
            x_txt = txt_tokens
            x_txt = self.text_encoder(x=self.text_embedder(x_txt), attn_mask=txt_mask)   # [B, T, C]

        ctx_cond = self.ctx_mask_proj(ctx_mask) + self.ctx_proj(ctx_feature)   # B, T, C

        audio_tokens = self.audio_token_embed(inputs['semantic_tokens'])
        audio_tokens = self.audio_token_upsampler(audio_tokens.transpose(1, 2)).transpose(1, 2) # [B, T, C]
        audio_tokens = self.audio_token_proj(audio_tokens.transpose(1, 2)).transpose(1, 2)

        if self.hp.text_inject_method == 'concat':
            x_noisy = self.prenet(x_noisy) + ctx_cond + audio_tokens
            x_noisy = add_prefix_nd(x_txt, txt_mask.sum(1), x_noisy, x_mask.sum(1))
            x_mask = sequence_mask(txt_mask.sum(1) + x_mask.sum(1))
        else:
            x_noisy = self.prenet(x_noisy) + ctx_cond + x_txt + audio_tokens
        encoder_out = self.encoder(x_noisy, self.f5_time_embed(t), attn_mask=x_mask, do_checkpoint=self.hp.do_checkpoint)
        if self.hp.text_inject_method == 'concat':
            encoder_out = remove_prefix(encoder_out, txt_mask.sum(1), x_mask.sum(1) - txt_mask.sum(1))
        pred = self.postnet(encoder_out)

        return pred, target

    def _forward(self, x, cond, other_cond, timesteps, ctx_mask, attn_mask, seq_cfg_w=[1.0, 1.5, 1.5, 1.5]):
        """ When we use torchdiffeq, we need to include the CFG process inside _forward() """
        x = x * (1 - ctx_mask)
        x = self.prenet(x) + cond
        if self.hp.text_inject_method == 'concat':
            x_txt = other_cond['x_txt']
            txt_len = x_txt.shape[1]
            x = torch.cat([x_txt, x], dim=1)
            attn_mask = torch.ones_like(x[..., 0])
        pred_v = self.encoder(x, self.f5_time_embed(timesteps), attn_mask=attn_mask)
        if self.hp.text_inject_method == 'concat':
            pred_v = pred_v[:, txt_len:]
        pred = self.postnet(pred_v)

        """ Perform CFG """
        ## old MegaTTS3 CFG:
        # uncond + w0 * cond_txt - w0 * uncond + w1 * cond_spk_txt - w1 * cond_txt
        # (1 - w0) * uncond + (w0 - w1) * cond_txt + w1 * cond_spk_txt
        
        ## new CFG:
        # uncond + s1 * (cond_txt - uncond) + s2 * (cond_spk - uncond) + s3 * (cond_spk_txt - uncond)
        # (1 - s1 - s2 - s3) * uncond + s1 * cond_txt + s2 * cond_spk + s3 * cond_spk_txt

        ## equation:
        # 1 - w0 = 1 - s1 - s2 - s3
        # w0 - w1 = s1
        # s2 = 0
        # w1 = s3
        ## answer:
        # w1 = s3
        # w0 = s1 + s3

        cond_all, cond_txt, cond_semantic, cond_spk, uncond = pred.chunk(5)

        pred = (
            uncond +
            seq_cfg_w[0] * (cond_all - uncond) + 
            seq_cfg_w[1] * (cond_txt - uncond) + 
            seq_cfg_w[2] * (cond_semantic - uncond) + 
            seq_cfg_w[3] * (cond_spk - uncond)
        )

        return pred

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, seq_cfg_w=[1.0, 1.5, 1.5, 1.5], **kwargs):
        audio_tokens = self.audio_token_embed(inputs['semantic_tokens'])
        audio_tokens = self.audio_token_upsampler(audio_tokens.transpose(1, 2)).transpose(1, 2) # [B, T, C]
        audio_tokens = self.audio_token_proj(audio_tokens.transpose(1, 2)).transpose(1, 2)  # [B, T, C]
        x_mask = torch.ones_like(audio_tokens[:, :, 0])

        (bsz, tgt_len, _), device = audio_tokens.shape, audio_tokens.device

        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask
        ctx_cond = self.ctx_mask_proj(ctx_mask) + self.ctx_proj(ctx_feature)

        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        x_txt = torch.full((bsz, tgt_len), self.hp.cfg_mask_text_token).to(txt_tokens)
        if self.hp.text_inject_method == 'left-prefill':
            x_txt[sequence_mask(txt_mask.sum(1), tgt_len)] = txt_tokens[txt_mask]
            x_txt = self.text_encoder(x=self.text_embedder(x_txt), attn_mask=x_mask)   # [B, T, C]
        elif self.hp.text_inject_method == 'expand-prefill':
            txt_tokens[~txt_mask] = self.hp.cfg_mask_text_token
            expand_mult = tgt_len // txt_tokens.shape[1]
            x_txt[:, :txt_tokens.shape[1]*expand_mult:expand_mult] = txt_tokens
            x_txt = self.text_encoder(x=self.text_embedder(x_txt), attn_mask=x_mask)   # [B, T, C]
        elif self.hp.text_inject_method == 'concat':
            x_txt = txt_tokens
            x_txt = self.text_encoder(x=self.text_embedder(x_txt), attn_mask=txt_mask)   # [B, T, C]

        if self.hp.text_inject_method == 'concat':
            cond = ctx_cond + audio_tokens
            other_cond = {
                'x_txt': x_txt
            }
        else:
            cond = ctx_cond + x_txt + audio_tokens
            other_cond = {}

        ''' Euler ODE solver '''
        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1).to(cond)
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

        traj = torchdiffeq.odeint(
            lambda t, x: self._forward(
                torch.cat([x] * bsz), cond, other_cond, timesteps=t.unsqueeze(0), ctx_mask=ctx_mask, attn_mask=x_mask, seq_cfg_w=seq_cfg_w),
            torch.randn([1, tgt_len, self.hp.out_channels], device=device),
            t_schedule,
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        x = traj[-1]
        return x
