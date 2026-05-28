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

from utils.commons.hparams import hparams
from modules.tts.ar_dur.commons.align_ops import expand_states
from modules.tts.llama_dit.llama_avgen import LLaMa
from utils.nn.seq_utils import sequence_mask

logger = logging.getLogger(__name__)

@dataclass
class ModelArgs:
    # text
    vocab_size: int = None
    text_dim: int = 1024

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

    caption_dim: int = 3584 + 1 # dim of seedance text encoder + content mask

    in_channels: int = 16
    out_channels: int = 16

    # diffusion
    cfg_mask_text_token: int = None

    # trainging
    do_checkpoint: bool = False
    use_qk_norm: bool = False
    use_dur: bool = False

    cfg_mask_text_token: int = None
    text_fill_token: int = None
    ph_fill_token: int = None
    use_caption_pool_in_adaln: bool = False


class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp

        self.encoder = LLaMa(hp)
        self.add_vad_mask = hparams.get('add_vad_mask', False)
        if self.add_vad_mask:
            print('| use vad mask!')
            self.prenet = nn.Linear(self.hp.in_channels + 2, self.hp.encoder_dim)
        else:
            self.prenet = nn.Linear(self.hp.in_channels + 1, self.hp.encoder_dim)

        self.lat_proj = nn.Linear(self.hp.encoder_dim * 2, self.hp.encoder_dim)
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels)
        self.caption_proj = nn.Linear(self.hp.caption_dim, self.hp.encoder_dim)

        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)

        from modules.flow_matching.llama import LLaMa as LLaMaSmall,  ModelArgs as ModelArgsSmall
        self.text_embedder = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        self.text_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=4, n_heads=16
        ))

        self.ph_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=4, n_heads=16
        ))
        self.tone_embed = nn.Embedding(32, hp.encoder_dim, padding_idx=0)
        self.ph_embed = nn.Embedding(302, hp.encoder_dim)
        if hp.use_dur:
            self.ph_offset_encoder = SinusoidalPositionalEmbedding(hp.encoder_dim)
        
        self.ling_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=4, n_heads=16
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

    def forward_ling_encoder(self, inputs, x_mask):
        ph_tokens = inputs["phone"]
        tone_tokens = inputs["tone"]
        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        bsz = ph_tokens.shape[0]
        tgt_len = x_mask.shape[1]
        
        ph_mask = ph_tokens > 0
        
        ph_tokens_ = torch.full((bsz, tgt_len), self.hp.ph_fill_token).to(ph_tokens)
        x_ph_mask = sequence_mask(ph_mask.long().sum(1), tgt_len)
        ph_tokens_[x_ph_mask] = ph_tokens[ph_mask]
        x_ph = self.ph_embed(ph_tokens_)
        tone_embed = self.tone_embed(tone_tokens)
        x_ph[x_ph_mask] = x_ph[x_ph_mask] + tone_embed[ph_mask]

        if self.hp.use_dur and 'dur' in inputs:
            dur = inputs['dur']
            offsets = torch.cumsum(dur, dim=1).long()   # [B, T]
            dur_embed = self.ph_offset_encoder(positions=offsets, out_dtype=x_ph.dtype, device=offsets.device)
            x_ph[x_ph_mask] = x_ph[x_ph_mask] + dur_embed[ph_mask]

        x_ph = self.ph_encoder(x=x_ph, attn_mask=x_mask)
        
        x_txt = torch.full((bsz, tgt_len), self.hp.text_fill_token).to(txt_tokens)
        x_txt[sequence_mask(txt_mask.sum(1), tgt_len)] = txt_tokens[txt_mask]
        x_txt = self.text_encoder(x=self.text_embedder(x_txt), attn_mask=x_mask)   # [B, T, C]
        
        x_txt = x_txt + x_ph
        x_txt = self.ling_encoder(x=x_txt, attn_mask=x_mask)
        
        return x_txt

    def forward(self, inputs, sigmas=None, x_noisy=None):
        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask
        bsz, device = ctx_feature.size(0), ctx_feature.device

        x = inputs['lat']
        x_mask = sequence_mask(inputs['lat_lens'], maxlen=x.shape[1])
    
        # Here, x is x1 in CFM
        x0 = torch.randn_like(x)
        t = self.flow_matcher.time_sampler.sample([x0.shape[0]], x0.device).type_as(x0)
        xt = t[:, None, None] * x + (1 - t[:, None, None]) * x0
        ut = x - x0
        # t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x)
        
        # define noisy_input and target
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t = self.f5_time_embed(t)
        x_noisy = (xt * (1 - ctx_mask)).bfloat16() + ctx_feature # prefix ref wav
        target = ut

        x_txt = self.forward_ling_encoder(inputs, x_mask)

        if 'caption_emb' in inputs and inputs['caption_emb'] is not None:
            caption_embs = self.caption_proj(inputs['caption_emb'])
        else:
            caption_embs = None
        if self.add_vad_mask:
            x_noisy = self.lat_proj(
                torch.cat([self.prenet(torch.cat([x_noisy, ctx_mask, inputs['vad_mask']], -1)), x_txt], -1)) 
        else:
            x_noisy = self.lat_proj(torch.cat([self.prenet(torch.cat([x_noisy, ctx_mask], -1)), x_txt], -1)) 
        encoder_out = self.encoder(x_noisy, t, attn_mask=x_mask,
                                   do_checkpoint=self.hp.do_checkpoint, context=caption_embs, 
                                   context_lens=inputs['caption_lens'], caption_mark=inputs.get('caption_mark', None)) # TODO context
        pred = self.postnet(encoder_out)

        return pred, target

    def _forward(self, x, cond, timesteps, seq_cfg_w=[1.0, 1.5, 1.5, 1.5]):
        """ When we use torchdiffeq, we need to include the CFG process inside _forward() """
        ctx = cond['ctx']
        ctx_mask = cond['ctx_mask']
        attn_mask = cond['attn_mask']
        x_txt = cond['x_txt']
        x = x * (1 - ctx_mask) + ctx
        if self.add_vad_mask:
            x = self.lat_proj(
                torch.cat([self.prenet(torch.cat([x, ctx_mask, cond['vad_mask']], -1)), x_txt],
                          -1))
        else:
            x = self.lat_proj(torch.cat([self.prenet(torch.cat([x, ctx_mask], -1)), x_txt], -1))
        if 'caption_emb' in cond and cond['caption_emb'] is not None:
            caption_embs = self.caption_proj(cond['caption_emb'])
        else:
            caption_embs = None

        with torch.amp.autocast('cuda', dtype=torch.float32):
            t = self.f5_time_embed(timesteps)
            
        pred_v = self.encoder(x, t, attn_mask=attn_mask, context=caption_embs, context_lens=cond['caption_lens'])
        pred = self.postnet(pred_v)

        cond_all, cond_txt, cond_cap, cond_ref, uncond = pred.chunk(5)

        pred = (
            uncond +
            seq_cfg_w[0] * (cond_all - uncond) + # all
            seq_cfg_w[1] * (cond_txt - uncond) + # txt
            seq_cfg_w[2] * (cond_cap - uncond) + # cap
            seq_cfg_w[3] * (cond_ref - uncond) # ref wav
        )

        return pred

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, seq_cfg_w=[1.0, 1.5, 1.5], **kwargs):

        audio_tokens = torch.zeros(inputs['lat_ctx'].shape[0],
                                    inputs['lat_ctx'].shape[1],
                                    self.hp.encoder_dim).to(inputs['lat_ctx'])
        x_mask = torch.ones_like(audio_tokens[:, :, 0])

        (bsz, tgt_len, _), device = audio_tokens.shape, audio_tokens.device

        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        x_txt = self.forward_ling_encoder(inputs, x_mask)

        cond = {
            'ctx': ctx_feature,
            'ctx_mask': ctx_mask,
            'attn_mask': x_mask,
            'x_txt': x_txt,
            'caption_emb':inputs['caption_emb'] if 'caption_emb' in inputs else None,
            'caption_lens': inputs['caption_lens'] if 'caption_lens' in inputs else None,
            'vad_mask': inputs['vad_mask']
        }

        ''' Euler ODE solver '''
        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1).to(device)
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

        traj = torchdiffeq.odeint(
            lambda t, x: self._forward(
                torch.cat([x] * bsz), cond, timesteps=t.unsqueeze(0), seq_cfg_w=seq_cfg_w),
            torch.randn([1, tgt_len, self.hp.out_channels], device=device),
            t_schedule,
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        x = traj[-1]
        return x
