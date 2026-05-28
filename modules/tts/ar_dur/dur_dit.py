import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import attrdict
import torchdiffeq

from modules.tts.ar_dur.commons.align_ops import expand_states
from modules.tts.llama_dit.llama_ca import LLaMa
from modules.commons.pos_encoding import SinusoidalPositionalEmbedding
from modules.tts.ar_dur.dur_lm import ContinuousTimeEncoderV2

from utils.nn.seq_utils import sequence_mask, remove_prefix, remove_suffix, add_prefix_nd
from utils.nn.generation_utils import stochastic_round
from utils.commons.io import print_once

def build_dur_model(hparams, text_tokenizer=None, init_pretrained=False, vocab_size=None, padding_idx=None):
    if text_tokenizer is not None:
        vocab_size = len(text_tokenizer)
    model_config = ModelArgs(
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        init_pretrained=init_pretrained
    )
    if hparams.get('model_size', 'base') == 'small':
        print_once('| use small model')
        model_config.encoder_n_layers = 12
        model_config.encoder_n_heads = 12
        model_config.encoder_dim = 768 
    elif hparams.get('model_size', 'base') == '1b':
        print_once('| use base 1b model')
        model_config.encoder_n_layers = 28
        model_config.encoder_n_heads = 24
        model_config.encoder_dim = 1536
    model_config.cond_dim = hparams.get('cond_dim', 1024)
    model = Diffusion(model_config)
    return model

@dataclass
class ModelArgs:
    # text
    vocab_size: int = None
    padding_idx: int = None

    encoder_dim: int = 1024
    encoder_n_layers: int = 24
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = None
    crossattn_n_layers: int = 12
    mlp_extend: float = None
    max_seq_len: int = 16384
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = 4
    use_causal_attn: bool = False
    use_qk_norm: bool = False
    use_dynamic_cross_gate: bool = False

    cond_dim: int = None
    
    init_pretrained: bool = True
    do_checkpoint: bool = False


class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp

        self.encoder = LLaMa(hp)
        self.dur_proj = nn.Linear(1, hp.encoder_dim)
        self.ctx_proj = nn.Linear(1, hp.encoder_dim)
        self.txt_embed = nn.Embedding(hp.vocab_size, hp.encoder_dim, hp.padding_idx)
        self.cond_proj = nn.Linear(self.hp.cond_dim, self.hp.encoder_dim, bias=False)
        self.out_proj = nn.Linear(hp.encoder_dim, 1)

        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)

    def forward(self, inputs, sigmas=None, x_noisy=None):
        txt_tokens = inputs['merged_ph_tokens']  # [B, T]
        txt_lens = inputs['merged_ph_tokens_len']
        txt_embeds = self.txt_embed(txt_tokens) # [B, T, C]

        dur_tokens = inputs['dur_tokens']   # [B, T]

        x_mask = sequence_mask(txt_lens)    # [B, T]
        loss_mask = sequence_mask(txt_lens)[..., None]  # [B, T, 1]
        dur_ctx_len_min = torch.clamp_min(txt_lens * 0.01, 1)
        dur_ctx_len_max = torch.clamp_min(txt_lens - dur_ctx_len_min, 1)
        dur_ctx_len = torch.rand(txt_lens.shape[0]).to(txt_lens.device) * (dur_ctx_len_max - dur_ctx_len_min) + dur_ctx_len_min
        dur_ctx_len = dur_ctx_len.long()
        ctx_mask = sequence_mask(dur_ctx_len, loss_mask.shape[1])[..., None]   # [B, T, 1]
        loss_mask[ctx_mask > 0] = 0

        x1 = torch.log1p(dur_tokens.to(txt_embeds))[..., None]  # [B, T, 1]
        x0 = torch.randn_like(x1)
        t = self.flow_matcher.time_sampler.sample([x0.shape[0]], x0.device).type_as(x0)
        xt = t[:, None, None] * x1 + (1 - t[:, None, None]) * x0
        xt = xt.to(txt_embeds)
        ut = x1 - x0

        # define noisy_input and target
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t = self.f5_time_embed(t)
        target = ut

        cond = inputs['condition']
        cond_lens = inputs['condition_lens']
        cond = self.cond_proj(cond)

        x_noisy = self.dur_proj(xt * (1 - ctx_mask.to(xt))) + self.ctx_proj(xt * ctx_mask.to(xt))
        encoder_out = self.encoder.forward(
            x=x_noisy,
            t=t,
            attn_mask=x_mask,
            context=cond,
            context_lens=cond_lens,
            do_checkpoint=self.hp.do_checkpoint
        )
        
        pred = self.out_proj(encoder_out)

        if loss_mask.sum() == 0:
            loss = 0.0
        else:
            loss = F.mse_loss(pred, target, reduction='none')
            loss = (loss * loss_mask).sum() / loss_mask.sum()

        return loss
    
    def _forward(self, x, cond, timesteps, seq_cfg_w=(1.0, 1.5, 1.5, 1.5), timestep_annealing_w=(1.0, 0.0, 1.0)):
        """ When we use torchdiffeq, we need to include the CFG process inside _forward() """
        ctx = cond['ctx']
        ctx_mask = cond['ctx_mask']
        attn_mask = cond['attn_mask']
        x_txt = cond['x_txt']
        
        x = x * (1 - ctx_mask)
        
        x = self.lat_proj(x) + self.ctx_proj(ctx) + self.ctx_mask_proj(ctx_mask)
        x = x + self.lat_offset_encoder(
            positions=torch.arange(x.shape[1]).to(x.device), out_dtype=x.dtype, device=x.device
        ) * 4
        x = self.prenet(torch.cat([x, x_txt], dim=-1))
        
        if 'caption_emb' in cond and cond['caption_emb'] is not None:
            caption_embs = self.caption_proj(cond['caption_emb'])
            if self.hp.use_caption_text_mark:
                caption_text_mark_embed = self.caption_text_mark_embed(cond['caption_text_mark'].long())
                caption_embs = caption_embs + caption_text_mark_embed
        else:
            caption_embs = None
            
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_embed = self.f5_time_embed(timesteps)
        pred_v = self.encoder(x, t_embed, attn_mask=attn_mask, context=caption_embs, context_lens=cond['caption_lens'])
        pred = self.postnet(pred_v)

        a, b, p = timestep_annealing_w
        gamma_t = a + b * torch.pow(1 - timesteps, p)
        seq_cfg_w = [gamma_t * w for w in seq_cfg_w]
        
        if len(seq_cfg_w) == 4:
            cond_all, cond_txt, cond_cap, cond_ref, uncond = pred.chunk(5)

            pred = (
                uncond +
                seq_cfg_w[0] * (cond_all - uncond) + # all
                seq_cfg_w[1] * (cond_txt - uncond) + # txt
                seq_cfg_w[2] * (cond_cap - uncond) + # cap
                seq_cfg_w[3] * (cond_ref - uncond) # ref wav
            )
        
        elif len(seq_cfg_w) == 2:
            cond_all, cond_txt, uncond = pred.chunk(3)
            pred = (
                uncond + 
                seq_cfg_w[0] * (cond_txt - uncond) + 
                seq_cfg_w[1] * (cond_all - cond_txt)
            )

        return pred

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, seq_cfg_w=[1.0, 1.5, 1.5, 1.5], timestep_annealing_w=(1.0, 0.0, 1.0), use_amo_sampler=False, **kwargs):
        if 'dur' in inputs:
            dur = inputs['dur']     # [B, T]
            tgt_len = dur.sum(1) // 4    # [B]
        else:
            tgt_len = inputs['tgt_len']
        x_mask = sequence_mask(tgt_len)
        
        x_txt = self.forward_ling_encoder(inputs, x_mask)

        (bsz, tgt_len, _), device = x_txt.shape, x_txt.device

        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        cond = {
            'ctx': ctx_feature,
            'ctx_mask': ctx_mask,
            'attn_mask': x_mask,
            'x_txt': x_txt,
            'caption_emb': inputs['caption_emb'],
            'caption_lens': inputs['caption_lens'],
        }
        if self.hp.use_caption_text_mark:
            cond['caption_text_mark'] = inputs['caption_text_mark']
        if inputs.get('vad_mask') is not None:
            cond['vad_mask'] = inputs['vad_mask']

        ''' Euler ODE solver '''
        t_schedule = torch.linspace(0, 1, timesteps + 1).to(device)
        
        sway_sampling_coef = -1.0
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)
        else:
            t_schedule = 0.5 * (1 - torch.cos(torch.pi * t_schedule))
            
        if use_amo_sampler:

            def amo_sampling(sample, sigma, sigma_next, pred_v):
                # Upcast to avoid precision issues when computing prev_sample
                t = sigma
                s = sigma_next
                x_t = sample
                c = 3  # 2
                o = min(s + c * (s - t), 1)
                pred_x_o = x_t + (o - t) * pred_v
                a = s / o
                b = (torch.clamp_min((1 - s) ** 2 - (a * (1 - o)) ** 2, 0)) ** 0.5
                noises = torch.randn(size=x_t.shape, device=x_t.device)
                prev_sample = a * pred_x_o + b * noises
                prev_sample = prev_sample.to(pred_v.dtype)
                return prev_sample

            x = torch.randn([1, tgt_len, self.hp.out_channels], device=device)                
            for step_index in range(timesteps):
                sigma = t_schedule[step_index].to(x_txt.dtype)
                sigma_next = t_schedule[step_index + 1]
                model_out = self._forward(torch.cat([x] * bsz), cond, timesteps=sigma.unsqueeze(0), seq_cfg_w=seq_cfg_w, timestep_annealing_w=timestep_annealing_w)
                x = amo_sampling(x, sigma, sigma_next, model_out)
        
        else:

            traj = torchdiffeq.odeint(
                lambda t, x: self._forward(
                    torch.cat([x] * bsz), cond, timesteps=t.unsqueeze(0), seq_cfg_w=seq_cfg_w, timestep_annealing_w=timestep_annealing_w),
                torch.randn([1, tgt_len, self.hp.out_channels], device=device),
                t_schedule,
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
            x = traj[-1]
        
        return x
