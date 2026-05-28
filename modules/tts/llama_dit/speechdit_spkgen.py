import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional
import traceback

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from modules.tts.ar_dur.commons.align_ops import expand_states
from modules.tts.llama_dit.llama import LLaMa
from modules.tts.llama_dit.time_embedding import CFMTimeEmbedding
from utils.nn.seq_utils import sequence_mask
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

    # image condition
    visual_model_name: str = 'simple_encoder'
    visual_model_path: str = ''
    use_img_cfg_mask_token: bool = False
    img_patch_size: int = 14
    apply_img_ids: bool = True
    fix_num_patches: bool = True
    num_patches: int = 3600
    use_visual_pooler: bool = False

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


def build_dit_model(hparams):
    config = ModelArgs()
    config.target_type = 'epsilon' if hparams.get('use_ddpm', False) else 'velocity'
    config.target_type = 'vector_field' if hparams.get('use_vpcfm', False) else config.target_type
    config.use_expand_ph = hparams.get('use_expand_ph', False)
    config.zero_xt_prompt = hparams.get('zero_xt_prompt', False)
    config.do_checkpoint = hparams.get('do_checkpoint', False)
    if hparams.get('use_small_model', False):
        config.encoder_n_layers = 12
        config.encoder_n_heads = 12
        config.encoder_dim = 768 
    if hparams.get('use_dit_1b', False):
        config.encoder_n_layers = 28
        config.encoder_n_heads = 16
        config.encoder_dim = 1536 
    config.max_seq_len = 16384
    config.in_channels = config.out_channels = hparams['latent_dim']
    
    config.visual_model_name = hparams.get('visual_model_name', 'simple_encoder')
    config.visual_model_path = hparams.get('visual_model_name', '')
    config.use_img_cfg_mask_token = hparams.get('use_img_cfg_mask_token', False)
    config.img_patch_size = hparams.get('patch_size', 14)
    config.apply_img_ids = hparams.get('apply_img_ids', True)
    config.fix_num_patches = hparams.get('fix_num_patches', True)
    config.num_patches = hparams.get('num_patches', 3600)
    config.use_visual_pooler = hparams.get('use_visual_pooler', False)

    dit = Diffusion(config)

    return dit


class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
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
        self.time_embedding = CFMTimeEmbedding(hp.time_embed_dim)

        self.ctx_mask_proj = nn.Linear(1, hp.ctx_mask_dim)
        
        # local_cond_in_channels = hp.out_channels + hp.ctx_mask_dim
        # self.local_cond_project = nn.Linear(
        #     local_cond_in_channels, hp.local_cond_dim)
        self.cond_prenet = nn.Linear(hp.encoder_dim + hp.ctx_mask_dim, hp.encoder_dim)

        if not hasattr(hp, "window_size"):
            hp.window_size = [-1, -1]
        
        self.encoder = LLaMa(hp)

        self.x_prenet = nn.Linear(hp.in_channels, hp.encoder_dim)
        # self.prenet = nn.Linear(hp.local_cond_dim, hp.encoder_dim)
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

        # image encoder
        if hp.visual_model_name == 'simple_encoder':
            from modules.tts.image_encoder.encoders import SimpleEncoder
            patch_size = hp.img_patch_size
            self.image_encoder = SimpleEncoder(
                in_channels=patch_size*patch_size*3,
                out_channels=hp.encoder_dim
            )
            if hp.apply_img_ids:
                self.image_posemb = nn.Linear(2, hp.encoder_dim)
        elif hp.visual_model_name == 'google/siglip-so400m-patch14-384':
            from modules.tts.image_encoder.encoders import SiglipEncoder
            self.image_encoder = SiglipEncoder(
                out_channels=hp.encoder_dim,
                pretrained_model_name_or_path=hp.visual_model_path,
                use_pooler_output=hp.use_visual_pooler
            )
        if hp.use_img_cfg_mask_token:
            self.img_cfg_mask_token = nn.Parameter(torch.zeros((1, 1, hp.encoder_dim)))

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
        
    # def forward(self, inputs, sigmas=None, x_noisy=None):
    #     image_patches = inputs['image_patches']
    #     lat_lens = inputs['lat_lens']
    #     x = inputs['lat']
    #     bsz, device = x.shape[0], x.device
    #     attn_mask = sequence_mask(lat_lens + self.hp.num_patches)

    #     ctx_feature = self.image_encoder(image_patches)     # [B, 3600, C]
    #     if self.hp.use_visual_pooler:
    #         ctx_feature = ctx_feature.unsqueeze(1)
    #     if 'img_ids' in inputs and inputs['img_ids'] is not None and self.image_posemb is not None:
    #         ctx_feature = ctx_feature + self.image_posemb(inputs['img_ids'])
    #     # if 'lat_cfg_mask' in inputs and inputs['lat_cfg_mask'] is not None:
    #     if self.hp.use_img_cfg_mask_token:
    #         lat_cfg_mask = inputs['lat_cfg_mask']
    #         ctx_feature = ctx_feature * (1 - lat_cfg_mask) + self.img_cfg_mask_token * lat_cfg_mask
    #     else:
    #         ctx_feature = ctx_feature * (1 - lat_cfg_mask)
    #     ctx_feature = torch.cat([ctx_feature, torch.zeros((bsz, x.shape[1], ctx_feature.shape[2]), device=device)], dim=1)  # [B, 3600+T, C]
        
    #     ctx_mask = torch.zeros((bsz, x.shape[1] + self.hp.num_patches, 1), device=device)
    #     ctx_mask[:, :self.hp.num_patches] = 1
    #     ctx_mask_emb = self.ctx_mask_proj(ctx_mask)
    #     local_cond = torch.cat([ctx_feature, ctx_mask_emb], dim=-1)
    #     local_cond = self.cond_prenet(local_cond)
    
    #     # Here, x is x1 in CFM
    #     x0 = torch.randn_like(x)
    #     t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x)
        
    #     # define noisy_input and target
    #     t = t.bfloat16()
    #     x_noisy = torch.cat([torch.zeros((bsz, self.hp.num_patches, xt.shape[2]), device=device), xt], dim=1).bfloat16()
    #     target = ut

    #     # concat condition.
    #     x_ling = self.forward_ling_encoder(inputs["phone"], inputs["tone"])
    #     x_ling = self.ling_pre_net(expand_states(x_ling, inputs['mel2ph']).transpose(1, 2)).transpose(1, 2)
    #     x_ling = torch.cat([torch.zeros((bsz, self.hp.num_patches, x_ling.shape[2]), device=device), x_ling], dim=1)
    #     x_noisy = self.x_prenet(x_noisy)
    #     # assert x_noisy.shape == local_cond.shape == x_ling.shape, f"x_noisy.shape {x_noisy.shape}\nlocal_cond.shape {local_cond.shape}\nx_ling.shape {x_ling.shape}\ninputs['mel2ph'].shape {inputs['mel2ph'].shape}\nx.shape {x.shape}\nwavs.shape {inputs['wavs'].shape}mel.shape {inputs['mel'].shape}"
    #     assert x_noisy.shape == local_cond.shape == x_ling.shape, f"x_noisy.shape {x_noisy.shape}\nlocal_cond.shape {local_cond.shape}\nx_ling.shape {x_ling.shape}\ninputs['mel2ph'].shape {inputs['mel2ph'].shape}\nx.shape {x.shape}"
    #     x_noisy = x_noisy + local_cond + x_ling
    #     encoder_out = self.encoder(x_noisy, self.f5_time_embed(t), attn_mask=attn_mask, do_checkpoint=self.hp.do_checkpoint)
    #     pred = self.postnet(encoder_out)
    #     pred = pred[:, self.hp.num_patches:]
    #     # x_noisy.shape torch.Size([12, 4062, 1024])
    #     # local_cond.shape torch.Size([12, 4062, 1024])
    #     # x_ling.shape torch.Size([12, 4064, 1024])
    #     # inputs['mel2ph'].shape torch.Size([12, 1856])
    #     # x.shape torch.Size([12, 462, 32])

    #     return pred, target

    def forward_img_encoder(self, image_patches, lat_len, img_ids=None, lat_cfg_mask=None):
        bsz, device = image_patches.shape[0], image_patches.device
        ctx_feature = self.image_encoder(image_patches)     # [B, T_img, C]
        if self.hp.use_visual_pooler:
            ctx_feature = ctx_feature.unsqueeze(1)
        if img_ids is not None and self.image_posemb is not None:
            ctx_feature = ctx_feature + self.image_posemb(img_ids)
        if lat_cfg_mask is not None:
            if self.hp.use_img_cfg_mask_token:
                ctx_feature = ctx_feature * (1 - lat_cfg_mask) + self.img_cfg_mask_token * lat_cfg_mask
            else:
                ctx_feature = ctx_feature * (1 - lat_cfg_mask)
        ctx_feature = torch.cat([ctx_feature, torch.zeros((bsz, lat_len, ctx_feature.shape[2]), device=device)], dim=1)  # [B, 3600+T, C]
        
        ctx_mask = torch.zeros((bsz, lat_len + self.hp.num_patches, 1), device=device)
        ctx_mask[:, :self.hp.num_patches] = 1
        ctx_mask_emb = self.ctx_mask_proj(ctx_mask)
        local_cond = torch.cat([ctx_feature, ctx_mask_emb], dim=-1)
        local_cond = self.cond_prenet(local_cond)
        return local_cond
    
    def forward(self, inputs, sigmas=None, x_noisy=None):
        image_patches = inputs['image_patches']
        lat_lens = inputs['lat_lens']
        x = inputs['lat']
        bsz, device = x.shape[0], x.device
        attn_mask = sequence_mask(lat_lens + self.hp.num_patches)

        local_cond = self.forward_img_encoder(
            image_patches,
            x.shape[1],
            inputs['img_ids'] if 'img_ids' in inputs else None,
            inputs['lat_cfg_mask']
        )   # [B, T_img+T, C]
    
        # Here, x is x1 in CFM
        x0 = torch.randn_like(x)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x)
        
        # define noisy_input and target
        t = t.bfloat16()
        x_noisy = torch.cat([torch.zeros((bsz, self.hp.num_patches, xt.shape[2]), device=device), xt], dim=1).bfloat16()
        target = ut

        # concat condition.
        x_ling = self.forward_ling_encoder(inputs["phone"], inputs["tone"])
        x_ling = self.ling_pre_net(expand_states(x_ling, inputs['mel2ph']).transpose(1, 2)).transpose(1, 2)
        x_ling = torch.cat([torch.zeros((bsz, self.hp.num_patches, x_ling.shape[2]), device=device), x_ling], dim=1)
        x_noisy = self.x_prenet(x_noisy)
        # assert x_noisy.shape == local_cond.shape == x_ling.shape, f"x_noisy.shape {x_noisy.shape}\nlocal_cond.shape {local_cond.shape}\nx_ling.shape {x_ling.shape}\ninputs['mel2ph'].shape {inputs['mel2ph'].shape}\nx.shape {x.shape}\nwavs.shape {inputs['wavs'].shape}mel.shape {inputs['mel'].shape}"
        assert x_noisy.shape == local_cond.shape == x_ling.shape, f"x_noisy.shape {x_noisy.shape}\nlocal_cond.shape {local_cond.shape}\nx_ling.shape {x_ling.shape}\ninputs['mel2ph'].shape {inputs['mel2ph'].shape}\nx.shape {x.shape}"
        x_noisy = x_noisy + local_cond + x_ling
        encoder_out = self.encoder(x_noisy, self.f5_time_embed(t), attn_mask=attn_mask, do_checkpoint=self.hp.do_checkpoint)
        pred = self.postnet(encoder_out)
        pred = pred[:, self.hp.num_patches:]

        return pred, target

    # def forward(self, inputs, sigmas=None, x_noisy=None):
    #     lat_lens = inputs['lat_lens']
    #     x = inputs['lat']
    #     bsz, device = x.shape[0], x.device
    #     attn_mask = sequence_mask(lat_lens)
    
    #     # Here, x is x1 in CFM
    #     x0 = torch.randn_like(x)
    #     t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x)
        
    #     # define noisy_input and target
    #     t = t.bfloat16()
    #     x_noisy = xt.bfloat16()
    #     target = ut

    #     # concat condition.
    #     x_ling = self.forward_ling_encoder(inputs["phone"], inputs["tone"])
    #     x_ling = self.ling_pre_net(expand_states(x_ling, inputs['mel2ph']).transpose(1, 2)).transpose(1, 2)
    #     x_noisy = self.x_prenet(x_noisy)
    #     x_noisy = x_noisy + x_ling
    #     encoder_out = self.encoder(x_noisy, self.f5_time_embed(t), attn_mask=attn_mask, do_checkpoint=self.hp.do_checkpoint)
    #     pred = self.postnet(encoder_out)

    #     return pred, target
    
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
        x = self.x_prenet(x) + local_cond + x_ling
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
        bsz, tgt_len, _ = x_ling.shape
        device = x_ling.device
        x_ling = torch.cat([torch.zeros((bsz, self.hp.num_patches, x_ling.shape[2]), device=device), x_ling], dim=1)

        lat_cfg_mask = torch.zeros((bsz, 1, 1), device=device)
        lat_cfg_mask[1:] = 1    # prefix spk cfg
        local_cond = self.forward_img_encoder(
            inputs['image_patches'],
            tgt_len,
            inputs['img_ids'] if 'img_ids' in inputs else None,
            lat_cfg_mask
        )
        ctx_mask = torch.zeros((bsz, tgt_len + self.hp.num_patches, 1), device=device)
        ctx_mask[:, :self.hp.num_patches] = 1

        ''' Euler ODE solver '''
        bsz, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1, device=device, dtype=x_ling.dtype)
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

        x0 = inputs.get('x0', None)
        if x0 is not None:
            assert x0.shape[1] >= frm_len
            x0 = x0[0:1, :frm_len]
        else:
            x0 = torch.randn([1, frm_len, self.hp.out_channels], device=device)
        
        traj = torchdiffeq.odeint(
            lambda t, x: self._forward(
                torch.cat([x] * bsz), local_cond, x_ling, timesteps=t.unsqueeze(0), ctx_mask=ctx_mask, dur=inputs['dur'], seq_cfg_w=seq_cfg_w),
            x0,
            t_schedule,
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        x = traj[-1]
        return x


    # @torch.no_grad()
    # def inference(self, inputs, timesteps=20, seq_cfg_w=[1.0, 1.0], **kwargs):
    #     # txt embedding
    #     # text_embed = self.ph_proj(inputs["phone"]) + self.tone_proj(inputs["tone"])  # [B, T, 1024]
    #     x_ling = self.forward_ling_encoder(inputs["phone"], inputs["tone"])
    #     x_ling = self.ling_pre_net(expand_states(x_ling, inputs['dur']).transpose(1, 2)).transpose(1, 2)
    #     bsz, tgt_len, _ = x_ling.shape
    #     device = x_ling.device

    #     ''' Euler ODE solver '''
    #     bsz, device, frm_len = (x_ling.size(0), x_ling.device, x_ling.size(1))
    #     sway_sampling_coef = -1.0
    #     t_schedule = torch.linspace(0, 1, timesteps + 1, device=device, dtype=x_ling.dtype)
    #     if sway_sampling_coef is not None:
    #         t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)
        
    #     traj = torchdiffeq.odeint(
    #         lambda t, x: self._forward(
    #             torch.cat([x] * bsz), 0, x_ling, timesteps=t.unsqueeze(0), ctx_mask=0, dur=inputs['dur'], seq_cfg_w=seq_cfg_w),
    #         torch.randn([1, frm_len, self.hp.out_channels], device=device),
    #         t_schedule,
    #         atol=1e-4,
    #         rtol=1e-4,
    #         method="euler",
    #     )
    #     x = traj[-1]
    #     return x
