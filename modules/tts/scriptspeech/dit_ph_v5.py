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

from utils.nn.seq_utils import sequence_mask, remove_prefix, remove_suffix, add_prefix_nd
from utils.nn.generation_utils import stochastic_round
from utils.commons.io import print_once

@dataclass
class ModelArgs:
    # text
    vocab_size: int = None
    text_dim: int = 1024
    text_inject_method: str = 'left-prefill'     # left-prefill | expand-prefill | concat
    use_sparse_dur: bool = False
    sparse_ph_idx: int = None
    sparse_tone_idx: int = None

    # audio
    add_vad_mask: bool = False
    audio_encoder_dim: int = None

    # caption
    caption_dim: int = 1024
    use_caption_encoder: bool = True
    use_caption_text_mark: bool = False
    crossattn_n_layers: int = 0
    
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
    use_qk_norm: bool = False
    use_dynamic_cross_gate: bool = False

    in_channels: int = 16
    out_channels: int = 16

    # diffusion
    cfg_mask_text_token: int = None
    text_fill_token: int = None
    ph_fill_token: int = None

    # trainging
    do_checkpoint: bool = False


class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp

        self.encoder = LLaMa(hp)
        self.add_vad_mask = hp.add_vad_mask
        
        self.lat_proj = nn.Linear(self.hp.in_channels, self.hp.encoder_dim)
        self.lat_offset_encoder = SinusoidalPositionalEmbedding(hp.encoder_dim)
        self.ctx_proj = nn.Linear(self.hp.in_channels, self.hp.encoder_dim)
        self.ctx_mask_proj = nn.Linear(1, self.hp.encoder_dim)
        self.prenet = nn.Linear(self.hp.encoder_dim * 2 , self.hp.encoder_dim)
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels)
        
        if hp.use_caption_encoder:
            self.caption_proj = nn.Linear(self.hp.caption_dim, self.hp.encoder_dim)
            if hp.use_caption_text_mark:
                self.caption_text_mark_embed = nn.Embedding(2, self.hp.encoder_dim)

        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)

        from modules.asr.llama.llama import LLaMa as LLaMaSmall, ModelArgs as ModelArgsSmall
        from modules.tts.ar_dur.dur_lm import ContinuousTimeEncoderV2
        
        self.dur_decoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=6, n_heads=16,
            use_causal_attn=False, 
            crossattn_n_layers=8,
            use_dynamic_cross_gate=True
        ))
        self.audio_feature_proj = nn.Linear(hp.audio_encoder_dim, hp.encoder_dim, bias=False)
        self.dur_ctx_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=2, n_heads=16,
            use_causal_attn=False, 
            crossattn_n_layers=0,
        ))
        self.dur_head = nn.Linear(hp.encoder_dim, 1)
        
        self.text_embedder = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        self.text_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=4, n_heads=16,
            use_causal_attn=False, 
            crossattn_n_layers=0,
        ))
        self.tone_embed = nn.Embedding(32, hp.encoder_dim, padding_idx=0)
        self.ph_embed = nn.Embedding(302, hp.encoder_dim)
        self.ph_offset_encoder = SinusoidalPositionalEmbedding(hp.encoder_dim)
        self.ph_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=4, n_heads=16,
            use_causal_attn=False, 
            crossattn_n_layers=8,
            use_dynamic_cross_gate=True
        ))
        self.ling_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=4, n_heads=16,
            use_causal_attn=False, 
            crossattn_n_layers=8,
            use_dynamic_cross_gate=True
        ))

        # # init all weights
        self._init_weights()

    def _init_weights(self) -> None:
        # Linear and Embedding layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.hp.encoder_n_layers))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.hp.encoder_n_layers)
                )
        # Time embedding MLP
        nn.init.normal_(self.f5_time_embed.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.f5_time_embed.time_mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks
        for block in self.encoder.layers:
            nn.init.zeros_(block.attention_norm.linear.weight)
            nn.init.zeros_(block.attention_norm.linear.bias)

        # Zero-out output layers
        nn.init.zeros_(self.encoder.norm.linear.weight)
        nn.init.zeros_(self.encoder.norm.linear.bias)
        nn.init.zeros_(self.encoder.out_proj.weight)
        nn.init.zeros_(self.encoder.out_proj.bias)

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
        
        # define noisy_input and target
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t = self.f5_time_embed(t)
        x_noisy = (xt * (1 - ctx_mask)).bfloat16()
        target = ut

        x_txt, dur_loss, dur_total_loss = self.forward_ling_encoder(inputs, x_mask, is_train=True)

        if self.hp.use_caption_encoder and 'caption_emb' in inputs and inputs['caption_emb'] is not None:
            caption_embs = self.caption_proj(inputs['caption_emb'])
            if self.hp.use_caption_text_mark:
                caption_text_mark_embed = self.caption_text_mark_embed(inputs['caption_text_mark'].long())
                caption_embs = caption_embs + caption_text_mark_embed
            caption_lens = inputs['caption_lens']
        else:
            caption_embs, caption_lens = None, None
            
        x_noisy = self.lat_proj(x_noisy) + self.ctx_proj(ctx_feature) + self.ctx_mask_proj(ctx_mask)
        x_noisy = x_noisy + self.lat_offset_encoder(
            positions=torch.arange(x_noisy.shape[1]).to(x_noisy.device), out_dtype=x_noisy.dtype, device=x_noisy.device
        ) * 4
        x_noisy = self.prenet(torch.cat([x_noisy, x_txt], dim=-1))

        encoder_out = self.encoder(x_noisy, t, attn_mask=x_mask, do_checkpoint=self.hp.do_checkpoint, 
                                   context=caption_embs, context_lens=caption_lens)
        
        pred = self.postnet(encoder_out)

        return pred, target, dur_loss, dur_total_loss
    
    def forward_ling_encoder(self, inputs, x_mask=None, is_train=True):
        ph_tokens = inputs["phone"]     # [B, T]
        tone_tokens = inputs["tone"]
        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        bsz = ph_tokens.shape[0]
        
        ph_mask = inputs.get('ph_mask', ph_tokens > 0)
        ph_cfg_mask = (ph_tokens[:, 0] == 301)[:, None]

        ph_embeds = self.ph_embed(ph_tokens)
        tone_embeds = self.tone_embed(tone_tokens)
        ph_embeds = ph_embeds + tone_embeds
        txt_embeds = self.text_encoder(
            x=self.text_embedder(txt_tokens), attn_mask=txt_mask, do_checkpoint=self.hp.do_checkpoint
        )
        x_txt = self.ph_encoder(
            x=ph_embeds,
            attn_mask=ph_mask,
            context=txt_embeds,
            context_lens=txt_mask.sum(1),
            do_checkpoint=self.hp.do_checkpoint
        )

        audio_ctx_feature = inputs['audio_ctx_feature']
        audio_ctx_mask = inputs['audio_ctx_mask']
        audio_ctx_feature = self.audio_feature_proj(audio_ctx_feature)
        audio_ctx_feature = self.dur_ctx_encoder.forward(
            x=audio_ctx_feature,
            attn_mask=audio_ctx_mask,
            do_checkpoint=self.hp.do_checkpoint,
        )
        dur_pred = self.dur_decoder.forward(
            x=x_txt + ph_embeds,
            attn_mask=ph_mask,
            context=audio_ctx_feature,
            context_lens=audio_ctx_mask.sum(1),
            do_checkpoint=self.hp.do_checkpoint,
        )
        dur_pred = self.dur_head(dur_pred)
        
        if "dur" in inputs and is_train:
            dur_cfg_mask = (inputs['dur'].sum(1) == 0)[:, None]
            ph_cfg_mask = 1 - (ph_cfg_mask | dur_cfg_mask).to(dur_pred)
            dur_tokens = inputs['dur'].clamp_min(0).to(x_txt)
            dur_loss = F.mse_loss(dur_pred[..., 0], dur_tokens.log1p(), reduction='none')
            if (ph_mask * ph_cfg_mask).sum() > 0:
                dur_loss = (dur_loss * ph_mask * ph_cfg_mask).sum() / (ph_mask * ph_cfg_mask).sum()
            else:
                dur_loss = 0
            dur_pred = dur_pred[..., 0].expm1()     # [B, T]
            dur_total_pred = dur_pred.sum(1)
            dur_tokens[ph_cfg_mask[:, 0] > 0] = ((dur_tokens - dur_pred).detach() + dur_pred)[ph_cfg_mask[:, 0] > 0].to(dur_tokens)
        else:
            dur_loss = None
            dur_tokens = dur_pred[..., 0].expm1()
            dur_total_pred = dur_tokens.sum(1)

        dur_tokens = stochastic_round(dur_tokens).clamp_min(0).long()

        dur_total_loss = None
        if x_mask is None and not is_train:
            # inference
            dur_tokens = dur_tokens[2:] = 0
            dur_total_pred = dur_tokens.sum(1)
            x_mask = sequence_mask(stochastic_round(dur_total_pred.float() / 4))
        else:
            dur_total_tgt = inputs['dur'].to(x_txt).sum(1)
            dur_total_loss = F.l1_loss(dur_total_pred, dur_total_tgt, reduction='none') / dur_total_tgt.clamp_min(1)
            if (ph_cfg_mask).sum() > 0:
                dur_total_loss = (dur_total_loss * ph_cfg_mask[:, 0]).sum() / (ph_cfg_mask[:, 0]).sum()
            else:
                dur_total_loss = 0
            dur_total_loss_log = F.mse_loss(dur_total_pred.log1p(), dur_total_tgt.log1p(), reduction='none')
            if (ph_cfg_mask).sum() > 0:
                dur_total_loss_log = (dur_total_loss_log * ph_cfg_mask[:, 0]).sum() / (ph_cfg_mask[:, 0]).sum()
            else:
                dur_total_loss_log = 0
            dur_total_loss = dur_total_loss + dur_total_loss_log
        tgt_len = x_mask.shape[1]

        x_ph = torch.full((bsz, tgt_len), self.hp.ph_fill_token).to(ph_tokens)
        x_ph = self.ph_embed(x_ph)
        x_ph_mask = sequence_mask(ph_mask.long().sum(1), tgt_len)
        x_ph[x_ph_mask] = x_txt[ph_mask]
        
        offsets = torch.cumsum(dur_tokens, dim=1).long()   # [B, T]
        dur_embed = self.ph_offset_encoder(positions=offsets, out_dtype=x_ph.dtype, device=offsets.device)
        x_ph[x_ph_mask] = x_ph[x_ph_mask] + dur_embed[ph_mask]

        x_txt = self.ling_encoder.forward(
            x=x_ph,
            attn_mask=x_mask,
            context=txt_embeds,
            context_lens=txt_mask.sum(1),
            do_checkpoint=self.hp.do_checkpoint,
        )

        if is_train:
            return x_txt, dur_loss, dur_total_loss

        return x_txt
    
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
