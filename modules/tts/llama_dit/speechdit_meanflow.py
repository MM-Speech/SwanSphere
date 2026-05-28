import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from modules.tts.ar_dur.commons.align_ops import expand_states
from modules.tts.llama_dit.llama import LLaMa
from modules.tts.llama_dit.time_embedding import CFMTimeEmbedding
import torchdiffeq

logger = logging.getLogger(__name__)

@dataclass
class ModelArgs:
    # frontend
    phone_embed_dim: int = 512
    tone_embed_dim: int = 128
    bpe_embed_dim: int = 384

    n_phone: int = 300 + 2
    n_tone: int = 30 + 2
    n_bpe: int = 32008
    bpe_pad: int = 32005

    local_cond_dim: int = 512
    time_embed_dim: int = 256
    ctx_mask_dim: int = 16

    local_cond_project_type: str = "linear"  # conv
    local_cond_conv_kernel: int = 9
    local_cond_conv_padding: int = 4

    # llama
    encoder_dim: int = 1024
    encoder_n_layers: int = 24
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = None
    mlp_extend: float = None
    max_seq_len: int = 16384
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    dropout: float = 0.0
    ffn_dim_multiplier: Optional[float] = 2
    use_causal_attn: bool = False

    causal: bool = False
    use_window_mask: bool = False
    window_size: list = field(default_factory=lambda: [-1, -1])
    window_type: str = "elemwise"  # elemwise, blockwise
    llama_provider: str = "ctiga"

    # speaker encoder
    spk_e_dim: int = 1024
    spk_embed_dim: int = 512

    # postnet
    postnet_type: str = "linear"  # conv
    postnet_kernel: int = 3

    target: str = "bn"
    prompt_feature: str = "bn"
    in_channels: int = 16
    out_channels: int = 16
    use_textprefix: bool = True

    target_type: str = "vector_field"

    # for uniform t
    lognormal_mean: float = 0.0
    lognormal_std: float = 1.0

    use_seg_embed: bool = True
    use_bn_eos_bos: bool = True

    max_phone_len: int = 2000
    max_bn_len: int = 4000

    flashattn_version: str = "2.3"

    use_expand_ph: bool = False
    zero_xt_prompt: bool = False

    use_causal_attn: bool = False
    
    use_cache: bool = False
    use_logitnorm_time: bool = False
    max_cache_batch_size: int = 10

    do_checkpoint: bool = False


class Diffusion(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        # text.
        # self.ph_proj = nn.Sequential(
        #     nn.Embedding(hp.n_phone, hp.phone_embed_dim, padding_idx=0),
        #     nn.Linear(hp.phone_embed_dim, hp.encoder_dim)
        # )
        # self.tone_proj = nn.Sequential(
        #     nn.Embedding(hp.n_tone, hp.tone_embed_dim, padding_idx=0),
        #     nn.Linear(hp.tone_embed_dim, hp.encoder_dim)
        # )

        # time-embedding
        # self.time_embedding = CFMTimeEmbedding(hp.time_embed_dim)

        self.ctx_mask_proj = nn.Linear(1, hp.ctx_mask_dim)
        
        local_cond_in_channels = hp.out_channels + hp.ctx_mask_dim

        self.local_cond_project = nn.Linear(
            local_cond_in_channels, hp.local_cond_dim)

        if not hasattr(hp, "window_size"):
            hp.window_size = [-1, -1]
        
        self.encoder = LLaMa(hp)

        self.x_prenet = nn.Linear(hp.in_channels, hp.encoder_dim)
        self.prenet = nn.Linear(hp.local_cond_dim, hp.encoder_dim)
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels)
  
        self.use_seg_embed = hp.use_seg_embed
        if hp.use_seg_embed:
            self.seg_embed = nn.Embedding(3, hp.encoder_dim, padding_idx=0)
            nn.init.trunc_normal_(self.seg_embed.weight, std=0.02, a=-0.04, b=0.04)

        if hp.use_bn_eos_bos:
            self.bn_eos_bos = nn.Parameter(torch.randn(2, hp.encoder_dim))  
        
        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)
        self.time_embed_merger = nn.Linear(2 * hp.encoder_dim, hp.encoder_dim)

        # text encoder
        from modules.tts.ar_dur.commons.layers import Embedding
        from modules.tts.ar_dur.commons.nar_tts_modules import PosEmb
        from modules.tts.ar_dur.commons.rel_transformer import RelTransformerEncoder
        self.ph_encoder = RelTransformerEncoder(
            302, hp.encoder_dim, hp.encoder_dim,
            hp.encoder_dim * 2, 4, 6,
            3, 0.0, prenet=True, pre_ln=True)
        self.tone_embed = Embedding(32, hp.encoder_dim, padding_idx=0)
        self.ph_pos_embed = PosEmb(hp.encoder_dim)
        self.ling_pre_net = torch.nn.Sequential(*[
            torch.nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate([2, 2])
        ])
    
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
        B, device = ctx_feature.size(0), ctx_feature.device

        """ local conditioning (prompt_latent + spk_embed) """
        ctx_mask_emb = self.ctx_mask_proj(ctx_mask)
        # ctx_feature = ctx_feature * (1 - inputs["spk_cfg_mask"][:, :, None])
        local_cond = torch.cat([ctx_feature, ctx_mask_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)

        """ diffusion target latent """
        x = inputs['lat']
    
        # Here, x is x1 in CFM
        x0 = torch.randn_like(x)
        # t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x)
        tr = self.flow_matcher.time_sampler.sample([x0.shape[0], 2], x0.device).type_as(x0)
        t = torch.maximum(tr[:, 0], tr[:, 1])
        r = torch.minimum(tr[:, 0], tr[:, 1])
        xt = x * t[:, None, None] + x0 * (1 - t[:, None, None])
        ut = x - x0

        def model_func(xt, t, r):
            # concat condition.
            t = t.bfloat16()
            r = r.bfloat16()
            x_noisy = (xt * (1 - ctx_mask)).bfloat16()
            x_ling = self.forward_ling_encoder(inputs["phone"], inputs["tone"])
            x_ling = self.ling_pre_net(expand_states(x_ling, inputs['mel2ph']).transpose(1, 2)).transpose(1, 2)
            x_noisy = self.x_prenet(x_noisy) + self.prenet(local_cond) + x_ling
            time_embed = self.time_embed_merger(torch.cat([self.f5_time_embed(t), self.f5_time_embed(r)], dim=-1))

            # encode
            encoder_out = self.encoder(x_noisy, time_embed, attn_mask=inputs["text_mel_mask"], do_checkpoint=self.hp.do_checkpoint)
            u_ = self.postnet(encoder_out)
            return u_
        
        # pred, dudt = torch.func.jvp(
        #     model_func,
        #     (xt, t, r),
        #     (ut, torch.ones_like(t), torch.zeros_like(r))
        # )
        pred, dudt = torch.autograd.functional.jvp(
            model_func,
            (xt, t, r),
            (ut, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True
        )
        target = ut - (t - r)[:, None, None] * dudt
        target = target.detach()

        return pred, target
    
    def forward_ling_encoder(self, txt_tokens, tone_tokens):
        ph_tokens = txt_tokens
        ph_nonpadding = (ph_tokens > 0).float()[:, :, None]  # [B, T_phone, 1]

        # enc_ph
        ph_enc_oembed = self.tone_embed(tone_tokens)
        ph_enc_oembed = ph_enc_oembed + self.ph_pos_embed(
            torch.arange(0, ph_tokens.shape[1])[None,].to(ph_tokens.device))
        ph_enc_oembed = ph_enc_oembed
        ph_enc_oembed = ph_enc_oembed * ph_nonpadding
        x_ling = self.ph_encoder(ph_tokens, other_embeds=ph_enc_oembed) * ph_nonpadding
        return x_ling

    def _forward(self, x, local_cond, x_ling, timesteps, ctx_mask, dur=None, seq_cfg_w=[1.0,1.0]):
        """ When we use torchdiffeq, we need to include the CFG process inside _forward() """
        x = x * (1 - ctx_mask)
        x = self.x_prenet(x) + self.prenet(local_cond) + x_ling
        pred_v = self.encoder(x, self.f5_time_embed(timesteps), attn_mask=torch.ones((x.size(0), x.size(1)), device=x.device))
        pred = self.postnet(pred_v)

        """ Perform CFG """
        cond_spk_txt, cond_txt, uncond = pred.chunk(3)
        pred = uncond + seq_cfg_w[0] * (cond_txt - uncond) + seq_cfg_w[1] * (cond_spk_txt - cond_txt)
        # pred = uncond + 3.0 * (0.05*(cond_txt-uncond) + 0.95*(cond_spk_txt-cond_txt))
        return pred

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, seq_cfg_w=[1.0, 1.0], **kwargs):
        # txt embedding
        # text_embed = self.ph_proj(inputs["phone"]) + self.tone_proj(inputs["tone"])  # [B, T, 1024]
        x_ling = self.forward_ling_encoder(inputs["phone"], inputs["tone"])
        x_ling = self.ling_pre_net(expand_states(x_ling, inputs['dur']).transpose(1, 2)).transpose(1, 2)

        # speaker embedding
        ctx_feature = inputs['lat_ctx']
        ctx_feature[1:, :, :] = 0 # prefix spk cfg
        ctx_mask_emb = self.ctx_mask_proj(inputs['ctx_mask'])

        # local conditioning.
        local_cond = torch.cat([ctx_feature, ctx_mask_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)
        
        ''' Euler ODE solver '''
        bsz, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
        time_schedule = torch.from_numpy(np.linspace(0, 1, timesteps+1, endpoint=True)).to(device)
        # x = torch.randn([1, frm_len, self.hp.out_channels], device=device)
        # for step_index in range(timesteps):
        #     # Upcast to avoid precision issues when computing prev_sample
        #     x = x.to(torch.float32)
        #     sigma = time_schedule[step_index].to(x_ling.dtype)
        #     sigma_next = time_schedule[step_index + 1]
        #     model_output = self._forward(torch.cat([x] * bsz), local_cond, x_ling, timesteps=sigma.unsqueeze(0), ctx_mask=inputs['ctx_mask'], dur=inputs['dur'], seq_cfg_w=seq_cfg_w)
        #     x = x + (sigma_next - sigma) * model_output
        #     # Cast sample back to model compatible dtype
        #     x = x.to(model_output.dtype)
        
        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1, device=device, dtype=x_ling.dtype)
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)
        
        traj = torchdiffeq.odeint(
            lambda t, x: self._forward(
                torch.cat([x] * bsz), local_cond, x_ling, timesteps=t.unsqueeze(0), ctx_mask=inputs['ctx_mask'], dur=inputs['dur'], seq_cfg_w=seq_cfg_w),
            torch.randn([1, frm_len, self.hp.out_channels], device=device),
            t_schedule,
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        x = traj[-1]
        return x
