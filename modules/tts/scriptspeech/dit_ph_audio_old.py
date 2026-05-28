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
from modules.tts.llama_dit.llama_moe import LLaMa

from utils.nn.seq_utils import sequence_mask
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

    # caption
    caption_dim: int = 1024
    use_caption_encoder: bool = True
    crossattn_n_layers: int = 24
    
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
    use_qk_norm: bool = True

    in_channels: int = 16
    out_channels: int = 16

    # diffusion
    cfg_mask_text_token: int = None

    # trainging
    do_checkpoint: bool = False
    use_qk_norm: bool = True


class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp

        self.encoder = LLaMa(hp)
        self.add_vad_mask = hp.add_vad_mask
        if self.add_vad_mask:
            print_once('| use vad mask!')
            self.prenet = nn.Linear(self.hp.in_channels + 2, self.hp.encoder_dim)
        else:
            self.prenet = nn.Linear(self.hp.in_channels + 1, self.hp.encoder_dim)
        self.lat_proj = nn.Linear(self.hp.encoder_dim * 2, self.hp.encoder_dim)
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels)

        # 文本 caption 的投影（保持原逻辑）
        if hp.use_caption_encoder:
            self.caption_proj = nn.Linear(self.hp.caption_dim, self.hp.encoder_dim)
        # 新增：caption_audio 的投影（无论 use_caption_encoder 与否，都可用）
        self.caption_audio_proj = nn.Linear(self.hp.caption_dim, self.hp.encoder_dim)

        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)

        from modules.flow_matching.llama import LLaMa as LLaMaSmall,  ModelArgs as ModelArgsSmall
        self.text_embedder = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        self.text_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=6, n_heads=8
        ))

        self.ph_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=6, n_heads=8
        ))
        self.tone_embed = nn.Embedding(32, hp.encoder_dim, padding_idx=0)
        self.ph_embed = nn.Embedding(302, hp.encoder_dim)
        self.ling_pre_net = torch.nn.Sequential(*[
            torch.nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate([2, 2])
        ])

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

    # ---------- 新增：统一对齐 audio_mask 到 [B, T_lat, 1] ----------
    def _align_time_mask(self, mask: Optional[torch.Tensor], B: int, T_lat: int, device, dtype) -> torch.Tensor:
        if mask is None:
            return torch.zeros((B, T_lat, 1), device=device, dtype=dtype)
        m = mask.to(device=device, dtype=dtype)
        if m.dim() == 2:
            m = m.unsqueeze(-1)
        # 对齐时间长度
        if m.size(1) < T_lat:
            m = F.pad(m, (0, 0, 0, T_lat - m.size(1)), value=0.0)
        elif m.size(1) > T_lat:
            print_once(f"| WARN audio_mask length {m.size(1)} > latent length {T_lat}, clipping to T_lat")
            m = m[:, :T_lat]
        if m.size(-1) != 1:
            m = m[..., :1]
        return m

    # ---------- 新增：构造基于 caption_audio 的时间序列特征 ----------
    def _build_caption_audio_feature(self, inputs, T_lat: int, device, dtype) -> torch.Tensor:
        """
        返回形状 [B, T_lat, C] 的特征：
          cap_audio_emb -> Linear -> 掩码平均到 [B,1,C] -> 按 audio_mask 扩展到 [B,T_lat,C]
        """
        cap_audio = inputs.get('caption_audio_emb', None)           # [B, La, D_cap]
        if cap_audio is None:
            # 没有则返回 0
            B = inputs['lat'].size(0)
            return torch.zeros((B, T_lat, self.hp.encoder_dim), device=device, dtype=dtype)

        cap_audio = cap_audio.to(device)
        cap_audio_proj = self.caption_audio_proj(cap_audio)         # [B, La, C]

        # 掩码平均（有 lens 用 lens，否则普通平均）
        lens = inputs.get('caption_audio_lens', None)
        if lens is not None:
            lens = lens.to(device)
            La = cap_audio_proj.size(1)
            mask = sequence_mask(lens, La).to(device=device, dtype=cap_audio_proj.dtype).unsqueeze(-1)  # [B,La,1]
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            cap_vec = (cap_audio_proj * mask).sum(dim=1, keepdim=True) / denom                          # [B,1,C]
        else:
            if cap_audio_proj.size(1) == 0:  # 罕见异常：空序列兜底
                B = cap_audio_proj.size(0)
                cap_vec = torch.zeros((B, 1, cap_audio_proj.size(-1)), device=device, dtype=cap_audio_proj.dtype)
            else:
                cap_vec = cap_audio_proj.mean(dim=1, keepdim=True)                                      # [B,1,C]

        # 扩展到 T_lat
        cap_feat = cap_vec.expand(-1, T_lat, -1)                                                         # [B,T_lat,C]

        # 乘以 audio_mask
        audio_mask = self._align_time_mask(inputs.get('audio_mask', None),
                                           B=cap_feat.size(0), T_lat=T_lat, device=device, dtype=cap_feat.dtype)
        cap_feat = cap_feat * audio_mask                                                                 # [B,T_lat,C]
        return cap_feat

    def forward(self, inputs, sigmas=None, x_noisy=None):
        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask
        bsz, device = ctx_feature.size(0), ctx_feature.device

        x = inputs['lat']
        x_mask = sequence_mask(inputs['lat_lens'])
        T_lat = x.size(1)

        # CFM 构造
        x0 = torch.randn_like(x)
        t = self.flow_matcher.time_sampler.sample([x0.shape[0]], x0.device).type_as(x0)
        xt = t[:, None, None] * x + (1 - t[:, None, None]) * x0
        ut = x - x0

        with torch.amp.autocast('cuda', dtype=torch.float32):
            t = self.f5_time_embed(t)
        x_noisy = (xt * (1 - ctx_mask)).bfloat16() + ctx_feature  # prefix ref wav
        target = ut

        # ------- 文本与音素条件 -------
        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        x_txt_tokens = torch.full((bsz, x.shape[1]), self.hp.cfg_mask_text_token).to(txt_tokens)
        x_txt_tokens[sequence_mask(txt_mask.sum(1), x.shape[1])] = txt_tokens[txt_mask]
        x_txt = self.text_encoder(x=self.text_embedder(x_txt_tokens), attn_mask=x_mask)   # [B, T, C]

        x_ph = self.forward_ph_encoder(inputs)                                            # [B, T, C]

        # ------- caption_audio 特征 -------
        cap_audio_feat = self._build_caption_audio_feature(inputs, T_lat=T_lat, device=device, dtype=x_txt.dtype)
        x_txt = x_txt + x_ph + cap_audio_feat

        # ------- cross-attn 上下文 -------
        if 'caption_emb' in inputs and inputs['caption_emb'] is not None:
            caption_embs = self.caption_proj(inputs['caption_emb'])
        else:
            caption_embs = None

        # ------- 与编码器前的投影 -------
        if self.add_vad_mask:
            x_noisy = self.lat_proj(torch.cat([self.prenet(torch.cat([x_noisy, ctx_mask, inputs['vad_mask']], -1)), x_txt], -1))
        else:
            x_noisy = self.lat_proj(torch.cat([self.prenet(torch.cat([x_noisy, ctx_mask], -1)), x_txt], -1))

        # === 关键新增：对齐并传入 audio_mask，用于 FFN 分路 ===
        audio_mask_lat = self._align_time_mask(
            inputs.get('audio_mask', None),
            B=bsz, T_lat=T_lat, device=device, dtype=x_noisy.dtype
        )

        encoder_out = self.encoder(
            x_noisy, t, attn_mask=x_mask,
            context=caption_embs, context_lens=inputs['caption_lens'],
            audio_mask=audio_mask_lat,     # <--- 新增参数
        )

        pred = self.postnet(encoder_out)
        return pred, target


    def _forward_ph_encoder(self, ph_tokens, tone_tokens):
        ph_mask = ph_tokens > 0
        ph_embed = self.ph_embed(ph_tokens)
        tone_embed = self.tone_embed(tone_tokens)
        x_ph = ph_embed + tone_embed
        x_ph = self.ph_encoder(x_ph, ph_mask)
        return x_ph

    def forward_ph_encoder(self, inputs):
        if not self.hp.use_sparse_dur:
            x_ph = self._forward_ph_encoder(inputs["phone"], inputs["tone"])
            x_ph = self.ling_pre_net(expand_states(x_ph, inputs['mel2ph']).transpose(1, 2)).transpose(1, 2)
        else:
            phone, tone = inputs["phone"], inputs["tone"]
            phone = torch.cat([torch.full_like(phone[:, 0:1], self.hp.sparse_ph_idx), phone], dim=1)
            tone = torch.cat([torch.full_like(tone[:, 0:1], self.hp.sparse_tone_idx), tone], dim=1)
            mel2ph_sparse = inputs['mel2ph_sparse']
            mel2ph = inputs['mel2ph']
            mel2ph_sparse[mel2ph > 0] += 1
            x_ph = self._forward_ph_encoder(phone, tone)
            x_ph = self.ling_pre_net(expand_states(x_ph, mel2ph_sparse).transpose(1, 2)).transpose(1, 2)

        return x_ph

    def _forward(self, x, cond, timesteps, seq_cfg_w=(1.0, 1.5, 1.5, 1.5), timestep_annealing_w=(1.0, 0.0, 1.0)):
        ctx = cond['ctx']
        ctx_mask = cond['ctx_mask']
        attn_mask = cond['attn_mask']
        x_txt = cond['x_txt']  # 已包含 txt + ph + cap_audio
        x = x * (1 - ctx_mask) + ctx

        if self.add_vad_mask:
            x = self.lat_proj(torch.cat([self.prenet(torch.cat([x, ctx_mask, cond['vad_mask']], -1)), x_txt], -1))
        else:
            x = self.lat_proj(torch.cat([self.prenet(torch.cat([x, ctx_mask], -1)), x_txt], -1))

        if 'caption_emb' in cond and cond['caption_emb'] is not None:
            caption_embs = self.caption_proj(cond['caption_emb'])
        else:
            caption_embs = None

        pred_v = self.encoder(
            x, self.f5_time_embed(timesteps),
            attn_mask=attn_mask,
            context=caption_embs, context_lens=cond['caption_lens'],
            audio_mask=cond.get('audio_mask', None),   # <--- 新增
        )
        pred = self.postnet(pred_v)

        cond_all, cond_txt, cond_cap, cond_ref, uncond = pred.chunk(5)

        a, b, p = timestep_annealing_w
        gamma_t = a + b * torch.pow(1 - timesteps, p)
        seq_cfg_w = [gamma_t * w for w in seq_cfg_w]

        pred = (
            uncond +
            seq_cfg_w[0] * (cond_all - uncond) +
            seq_cfg_w[1] * (cond_txt - uncond) +
            seq_cfg_w[2] * (cond_cap - uncond) +
            seq_cfg_w[3] * (cond_ref - uncond)
        )
        return pred


    @torch.no_grad()
    def inference(self, inputs, timesteps=20, seq_cfg_w=[1.0, 1.5, 1.5, 1.5], timestep_annealing_w=(1.0, 0.0, 1.0), **kwargs):
        x_ph = self.forward_ph_encoder(inputs)
        x_mask = torch.ones_like(x_ph[:, :, 0])

        (bsz, tgt_len, _), device = x_ph.shape, x_ph.device

        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        # 文本特征
        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        x_txt_tokens = torch.full((bsz, tgt_len), self.hp.cfg_mask_text_token).to(txt_tokens)
        x_txt_tokens[sequence_mask(txt_mask.sum(1), tgt_len)] = txt_tokens[txt_mask]
        x_txt = self.text_encoder(x=self.text_embedder(x_txt_tokens), attn_mask=x_mask)

        # caption_audio 特征
        cap_audio_feat = self._build_caption_audio_feature(inputs, T_lat=tgt_len, device=device, dtype=x_txt.dtype)
        x_txt = x_txt + x_ph + cap_audio_feat

        # === 关键新增：推理也对齐并携带 audio_mask ===
        audio_mask_lat = self._align_time_mask(
            inputs.get('audio_mask', None),
            B=bsz, T_lat=tgt_len, device=device, dtype=x_txt.dtype
        )

        cond = {
            'ctx': ctx_feature,
            'ctx_mask': ctx_mask,
            'attn_mask': x_mask,
            'x_txt': x_txt,
            'caption_emb': inputs['caption_emb'],
            'caption_lens': inputs['caption_lens'],
            'audio_mask': audio_mask_lat,   # <--- 新增
        }
        if self.add_vad_mask:
            cond['vad_mask'] = inputs.get('vad_mask', torch.zeros_like(ctx_mask))

        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1).to(device)
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

        traj = torchdiffeq.odeint(
            lambda t, x: self._forward(
                torch.cat([x] * bsz), cond, timesteps=t.unsqueeze(0),
                seq_cfg_w=seq_cfg_w, timestep_annealing_w=timestep_annealing_w),
            torch.randn([1, tgt_len, self.hp.out_channels], device=device),
            t_schedule,
            atol=1e-4, rtol=1e-4, method="euler",
        )
        x = traj[-1]
        return x
