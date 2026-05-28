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
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = 4
    use_causal_attn: bool = False

    in_channels: int = 16
    out_channels: int = 16

    # diffusion
    cfg_mask_text_token: int = None

    # training
    do_checkpoint: bool = False
    use_qk_norm: bool = True
    use_caption_pool_in_adaln: bool = False

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
        if hp.use_caption_encoder:
            self.caption_proj = nn.Linear(self.hp.caption_dim, self.hp.encoder_dim)

        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)

        from modules.flow_matching.llama import LLaMa as LLaMaSmall,  ModelArgs as ModelArgsSmall
        # text
        self.text_embedder = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        self.text_encoder = LLaMaSmall(ModelArgsSmall(dim=hp.encoder_dim, n_layers=6, n_heads=8))
        # phone/tone
        self.ph_encoder = LLaMaSmall(ModelArgsSmall(dim=hp.encoder_dim, n_layers=6, n_heads=8))
        self.tone_embed = nn.Embedding(32, hp.encoder_dim, padding_idx=0)
        self.ph_embed = nn.Embedding(302, hp.encoder_dim)
        self.ling_pre_net = torch.nn.Sequential(*[
            torch.nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate([2, 2])
        ])
        # ===== NEW: speaker role（按 ph 路径单独扩展，不 sparse）=====
        self.spk_embed = nn.Embedding(3, hp.encoder_dim, padding_idx=0)
        self.spk_encoder = LLaMaSmall(ModelArgsSmall(dim=hp.encoder_dim, n_layers=6, n_heads=8))
        self.spk_pre_net = torch.nn.Sequential(*[
            torch.nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate([2, 2])
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.hp.encoder_n_layers))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.hp.encoder_n_layers))
        nn.init.normal_(self.f5_time_embed.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.f5_time_embed.time_mlp[2].weight, std=0.02)
        for block in self.encoder.layers:
            nn.init.zeros_(block.attention_norm.linear.weight)
            nn.init.zeros_(block.attention_norm.linear.bias)
        nn.init.zeros_(self.encoder.norm.linear.weight)
        nn.init.zeros_(self.encoder.norm.linear.bias)
        nn.init.zeros_(self.encoder.out_proj.weight)
        nn.init.zeros_(self.encoder.out_proj.bias)

    # --------- NEW: speaker 分支（仿 ph 路径，非 sparse）---------
    def _forward_spk_encoder(self, spk_ids):
        """
        spk_ids: [B, Tph]，phone-level 说话人 id（0-pad, 1/2/... 角色）
        输出：编码后的 [B, Tph, C]
        """
        spk_mask = spk_ids > 0
        spk_embed = self.spk_embed(spk_ids)               # [B, Tph, C]
        x_spk = self.spk_encoder(spk_embed, spk_mask)     # [B, Tph, C]
        return x_spk

    def forward_spk_encoder(self, inputs):
        """
        expand 到 mel 时轴，再通过 2×stride=2 的 conv 下采样到 latent 时轴
        """
        spk_ids = inputs.get('spk_ids', None)
        if spk_ids is None:
            spk_ids = inputs.get('spk_mask', None)
        assert spk_ids is not None, "inputs 需包含 phone-level 的 spk_ids（或 spk_mask）"
        x_spk = self._forward_spk_encoder(spk_ids)                                        # [B, Tph, C]
        x_spk = self.spk_pre_net(expand_states(x_spk, inputs['mel2ph']).transpose(1, 2))  # [B, C, Tlat]
        x_spk = x_spk.transpose(1, 2)                                                     # [B, Tlat, C]
        return x_spk

    # -------------------------------------------------------------

    def forward(self, inputs, sigmas=None, x_noisy=None):
        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask
        bsz, device = ctx_feature.size(0), ctx_feature.device

        x = inputs['lat']
        x_mask = sequence_mask(inputs['lat_lens'])  # [B, T]

        # CFM: x1=x, t 采样
        x0 = torch.randn_like(x)
        t = self.flow_matcher.time_sampler.sample([x0.shape[0]], x0.device).type_as(x0)
        xt = t[:, None, None] * x + (1 - t[:, None, None]) * x0
        ut = x - x0

        with torch.amp.autocast('cuda', dtype=torch.float32):
            t = self.f5_time_embed(t)

        x_noisy = (xt * (1 - ctx_mask)).bfloat16() + ctx_feature
        target = ut

        # ===== 文本（按 T=latent 长度铺展）=====
        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        T = x.shape[1]
        x_txt_ids = torch.full((bsz, T), self.hp.cfg_mask_text_token,
                               dtype=txt_tokens.dtype, device=txt_tokens.device)
        valid_T = sequence_mask(txt_mask.long().sum(1), T)
        x_txt_ids[valid_T] = txt_tokens[txt_mask]
        x_txt = self.text_encoder(x=self.text_embedder(x_txt_ids), attn_mask=x_mask)   # [B, T, C]

        # ===== phone 分支 =====
        x_ph = self.forward_ph_encoder(inputs)  # [B, T, C]

        # ===== speaker 分支（仿 ph，非 sparse）=====
        x_spk = self.forward_spk_encoder(inputs)  # [B, T, C]

        # 融合
        x_txt = x_txt + x_ph + x_spk

        # caption
        if 'caption_emb' in inputs and inputs['caption_emb'] is not None:
            caption_embs = self.caption_proj(inputs['caption_emb'])
        else:
            caption_embs = None

        # 主干
        if self.add_vad_mask:
            x_noisy = self.lat_proj(
                torch.cat([self.prenet(torch.cat([x_noisy, ctx_mask, inputs['vad_mask']], -1)), x_txt], -1))
        else:
            x_noisy = self.lat_proj(torch.cat([self.prenet(torch.cat([x_noisy, ctx_mask], -1)), x_txt], -1))

        encoder_out = self.encoder(
            x_noisy, t, attn_mask=x_mask, do_checkpoint=self.hp.do_checkpoint,
            context=caption_embs, context_lens=inputs['caption_lens']
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

    def _forward(self, x, cond, timesteps,
                # 2档阶梯: [w_txt, w_all]；4档: [w_all, w_txt, w_cap, w_ref]；老版: [w_all, w_txt, w_cap, w_ref, w_spk]
                seq_cfg_w=(1.0, 1.5),
                timestep_annealing_w=(1.0, 0.0, 1.0)):
        ctx = cond['ctx']; ctx_mask = cond['ctx_mask']; attn_mask = cond['attn_mask']
        x_txt = cond['x_txt']
        if self.add_vad_mask:
            x = x * (1 - ctx_mask) + ctx
            x = self.lat_proj(torch.cat([self.prenet(torch.cat([x, ctx_mask, cond['vad_mask']], -1)), x_txt], -1))
        else:
            x = x * (1 - ctx_mask) + ctx
            x = self.lat_proj(torch.cat([self.prenet(torch.cat([x, ctx_mask], -1)), x_txt], -1))

        caption_embs = self.caption_proj(cond['caption_emb']) if ('caption_emb' in cond and cond['caption_emb'] is not None) else None
        pred_v = self.encoder(x, self.f5_time_embed(timesteps), attn_mask=attn_mask,
                            context=caption_embs, context_lens=cond['caption_lens'])
        pred = self.postnet(pred_v)

        # 阶梯权重随时间退火（与你现有一致）
        a, b, p = timestep_annealing_w
        gamma_t = a + b * torch.pow(1 - timesteps, p)
        seq_cfg_w = [gamma_t * w for w in seq_cfg_w]

        # 2档（3路）：[all, txt, uncond]
        if len(seq_cfg_w) == 2:
            cond_all, cond_txt, uncond = pred.chunk(3)
            pred = uncond + seq_cfg_w[0] * (cond_txt - uncond) + seq_cfg_w[1] * (cond_all - cond_txt)
            return pred

        # 4档（5路）：[all, txt, cap, ref, uncond]
        if len(seq_cfg_w) == 4:
            cond_all, cond_txt, cond_cap, cond_ref, uncond = pred.chunk(5)
            pred = (uncond
                    + seq_cfg_w[0] * (cond_all - uncond)
                    + seq_cfg_w[1] * (cond_txt - uncond)
                    + seq_cfg_w[2] * (cond_cap - uncond)
                    + seq_cfg_w[3] * (cond_ref - uncond))
            return pred

        # 兼容老 6路（含 spk）
        cond_all, cond_txt, cond_cap, cond_ref, cond_spk, uncond = pred.chunk(6)
        pred = (uncond
                + seq_cfg_w[0] * (cond_all - uncond)
                + seq_cfg_w[1] * (cond_txt - uncond)
                + seq_cfg_w[2] * (cond_cap - uncond)
                + seq_cfg_w[3] * (cond_ref - uncond)
                + seq_cfg_w[4] * (cond_spk - uncond))
        return pred


    @torch.no_grad()
    def inference(self, inputs, timesteps=20,
                seq_cfg_w=(1.0, 1.5), timestep_annealing_w=(1.0, 0.0, 1.0), **kwargs):
        x_ph  = self.forward_ph_encoder(inputs)
        # 若保留说话人编码：对话版才有
        x_spk = self.forward_spk_encoder(inputs) if hasattr(self, 'forward_spk_encoder') else 0
        x_mask = torch.ones_like(x_ph[:, :, 0])

        bsz, tgt_len, _ = x_ph.shape
        device = x_ph.device

        ctx_mask    = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        # 文本特征（按 latent 长度铺展）
        txt_tokens, txt_mask = inputs["txt_tokens"], inputs["txt_mask"]
        x_txt_ids = torch.full((bsz, tgt_len), self.hp.cfg_mask_text_token,
                            dtype=txt_tokens.dtype, device=txt_tokens.device)
        from utils.nn.seq_utils import sequence_mask
        valid_T = sequence_mask(txt_mask.long().sum(1), tgt_len)
        x_txt_ids[valid_T] = txt_tokens[txt_mask]
        x_txt = self.text_encoder(x=self.text_embedder(x_txt_ids), attn_mask=x_mask)
        x_txt = x_txt + x_ph + (x_spk if isinstance(x_spk, torch.Tensor) else 0)

        cond = {
            'ctx': ctx_feature, 'ctx_mask': ctx_mask, 'attn_mask': x_mask,
            'x_txt': x_txt, 'caption_emb': inputs.get('caption_emb', None),
            'caption_lens': inputs.get('caption_lens', None),
        }
        if self.add_vad_mask and ('vad_mask' in inputs):
            cond['vad_mask'] = inputs['vad_mask']

        # ODE 时间表（保留你的 sway）
        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1, device=device)
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

        traj = torchdiffeq.odeint(
            lambda t, x: self._forward(torch.cat([x] * bsz), cond,
                                    timesteps=t.unsqueeze(0),
                                    seq_cfg_w=seq_cfg_w,
                                    timestep_annealing_w=timestep_annealing_w),
            torch.randn([1, tgt_len, self.hp.out_channels], device=device),
            t_schedule, atol=1e-4, rtol=1e-4, method="euler",
        )
        return traj[-1]
