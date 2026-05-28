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
from einops import rearrange, repeat
import torchdiffeq

from modules.tts.ar_dur.commons.align_ops import expand_states
from modules.tts.llama_dit.llama_ca import LLaMa

from utils.nn.seq_utils import sequence_mask, add_prefix, add_prefix_nd, remove_prefix, last_token_mask
from utils.nn.generation_utils import amo_sampling
from utils.losses.focal_loss import sigmoid_focal_loss
from utils.commons.io import print_once
from utils.nn.seq_utils import get_incremental_state, set_incremental_state, softmax, make_positions

@dataclass
class ModelArgs:    
    # text
    text_vocab_size: int = None
    phone_vocab_size: int = 302
    tone_vocab_size: int = 32
    text_dim: int = 1024

    # audio
    patch_size: int = 4
    ctx_n_patches: int = 2
    add_vad_mask: bool = False

    # caption
    caption_dim: int = 1024
    use_caption_encoder: bool = True
    crossattn_n_layers: int = 24
    
    # encoder
    encoder_dim: int = 1024
    encoder_n_layers: int = 6
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = None
    
    # lm
    lm_dim: int = 1024
    lm_enc_n_layers: int = 2
    lm_dec_n_layers: int = 36
    lm_n_heads: int = 16
    
    # stop clf
    focal_loss_alpha: float = 0.99
    focal_loss_gamma: float = 1.5
    
    # decoder
    decoder_dim: int = 1024
    decoder_n_layers: int = 6
    decoder_n_heads: int = 16
    decoder_n_kv_heads: int = None
    training_patch_keep_ratio: float = -1.0
    decoder_use_caption: bool = True
    
    # latent
    in_channels: int = 16
    out_channels: int = 16

    # trainging
    do_checkpoint: bool = False
    
    
class DurationEncoder(nn.Module):
    def __init__(self, dim, K=128, scales=(0.1, 0.2, 0.2, 0.3), min_value=0.0, max_value=128.0):
        super().__init__()
        assert K % (2 * len(scales)) == 0 or K % len(scales) == 0
        K_per = K // len(scales)
        
        Bs = []
        for s in scales:
            B = torch.randn(K_per) * s
            B = B.clamp(min=-3.0*s, max=3.0*s)
            Bs.append(B)
        B = torch.cat(Bs, dim=0)  # [K]
        self.register_buffer("B", B)
        
        self.fourier_proj = nn.Linear(2 * K, dim)
        self.linear_proj = nn.Linear(1, dim)
        self.discrete_proj = nn.Embedding(int(max_value + 10), dim)
        self.out_proj = nn.Linear(dim, dim)
        self.min_value = min_value
        self.max_value = max_value
    
    def forward(self, x):
        # x [B, T]
        x = x.clamp(self.min_value, self.max_value).float()
        # RFF
        z = x[..., None] * self.B   # [B, T, K]
        x1 = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        x1 = x1 / math.sqrt(self.B.shape[-1])
        x1 = self.fourier_proj(x1)
        
        x_lin = (x / self.max_value)[..., None]
        x = x1 + self.linear_proj(x_lin) + self.discrete_proj(x.clamp(0, self.max_value).long())
        x = self.out_proj(F.silu(x))
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, proj=True, init_size=1024, store_dtype=torch.float32):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError(f"embedding_dim must be >= 2, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.weights = self.get_embedding(
            num_embeddings=init_size,
            embedding_dim=embedding_dim,
            device=None,
            dtype=store_dtype,
        )
        if proj:
            self.proj = nn.Linear(embedding_dim, embedding_dim)
        else:
            self.proj = nn.Identity()
        
    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, device=None, dtype=torch.float32):
        half_dim = embedding_dim // 2
        if half_dim > 1:
            div_term = torch.exp(
                torch.arange(half_dim, device=device, dtype=dtype)
                * (-math.log(10000.0) / (half_dim - 1))
            )
        elif half_dim == 1:
            div_term = torch.ones(1, device=device, dtype=dtype)
        else:
            raise ValueError("Invalid embedding_dim leading to half_dim == 0")

        positions = torch.arange(num_embeddings, device=device, dtype=dtype).unsqueeze(1)  # [L, 1]
        angles = positions * div_term.unsqueeze(0)  # [L, half_dim]

        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [L, 2*half_dim]
        if embedding_dim % 2 == 1:
            pad = torch.zeros(num_embeddings, 1, device=device, dtype=dtype)
            emb = torch.cat([emb, pad], dim=1)  # [L, D]
        return emb

    @torch.no_grad()
    def _maybe_grow(self, max_pos_needed):
        if max_pos_needed > self.weights.size(0):
            self.weights = self.get_embedding(
                num_embeddings=max_pos_needed,
                embedding_dim=self.embedding_dim,
                device=self.weights.device,
                dtype=self.weights.dtype,
            )

    def forward(self, timestep=None, positions=None, offset=0, out_dtype=None, device=None):
        if (timestep is None) and (positions is None):
            raise ValueError("Either `timestep` or `positions` must be provided.")

        dev = self.weights.device if device is None else device
        self.weights = self.weights.to(device=dev)
        store_dtype = self.weights.dtype

        max_index = 0
        if timestep is not None:
            if torch.is_tensor(timestep):
                if timestep.numel() == 0:
                    raise ValueError("Empty `timestep` tensor.")
                tmax = int(timestep.max().item())
            else:
                tmax = int(timestep)
            max_index = max(max_index, tmax + offset)

        if positions is not None:
            if not torch.is_tensor(positions):
                raise TypeError("`positions` must be a torch.Tensor.")
            if positions.numel() == 0:
                raise ValueError("Empty `positions` tensor.")
            pmax = int(positions.max().item())
            max_index = max(max_index, pmax + offset)

        if max_index < 0:
            raise ValueError("All (index + offset) must be non-negative.")

        max_pos_needed = max_index + 1
        self._maybe_grow(max_pos_needed)

        out_dtype = out_dtype or store_dtype

        if positions is not None:
            idx = positions.to(device=dev, dtype=torch.long) + offset
            if (idx < 0).any():
                raise ValueError("positions + offset contains negative indices.")
            emb = self.weights.index_select(0, idx.reshape(-1))
            emb = emb.reshape(*positions.shape, -1)
            return self.proj(emb.to(dtype=out_dtype))

        if torch.is_tensor(timestep):
            idx = timestep.to(device=dev, dtype=torch.long) + offset
            if (idx < 0).any():
                raise ValueError("timestep + offset contains negative indices.")
            emb = self.weights.index_select(0, idx.reshape(-1)).reshape(*idx.shape, -1)
            return self.proj(emb.to(dtype=out_dtype))
        else:
            idx = int(timestep) + offset
            if idx < 0:
                raise ValueError("timestep + offset is negative.")
            return self.proj(self.weights[idx].to(dtype=out_dtype))  # (D,)


class StopClf(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_size=256):
        super().__init__()
        self.linear1 = nn.Linear(input_channels, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_channels)
    
    def forward(self, x):
        x = F.silu(self.linear1(x))
        x = self.linear2(x)
        return x


class DiTARModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = self.hp = config
        
        # linguistic
        from modules.flow_matching.llama import LLaMa as LLaMaSmall,  ModelArgs as ModelArgsSmall
        self.text_embedder = nn.Embedding(config.text_vocab_size, config.text_dim)
        self.text_encoder = LLaMaSmall(ModelArgsSmall(
            dim=config.text_dim,
            n_layers=4, n_heads=16
        ))
        self.ph_encoder = LLaMaSmall(ModelArgsSmall(
            dim=config.text_dim,
            n_layers=4, n_heads=16
        ))
        self.tone_embed = nn.Embedding(32, config.text_dim, padding_idx=0)
        self.ph_embed = nn.Embedding(302, config.text_dim)
        
        self.duration_encoder = DurationEncoder(config.text_dim)
        self.ph_offset_encoder = SinusoidalPositionalEmbedding(config.text_dim, proj=True)
        
        self.ling_postnet = nn.Linear(config.text_dim, config.lm_dim, bias=False)
        
        # semantic
        self.caption_lm_proj = nn.Linear(config.caption_dim, config.lm_dim)
        self.caption_decoder_proj = nn.Linear(config.caption_dim, config.decoder_dim)

        # encoder
        from modules.flow_matching.llama import LLaMa as LLaMaSmall,  ModelArgs as ModelArgsSmall
        self.encoder_prenet = nn.Linear(config.in_channels, config.encoder_dim)
        self.encoder = LLaMaSmall(ModelArgsSmall(
            dim=config.encoder_dim,
            n_layers=config.encoder_n_layers, 
            n_heads=config.encoder_n_heads,
            use_qk_norm=True
        ))
        self.encoder_cls_token = nn.Parameter(torch.randn((1, 1, config.encoder_dim)))
        self.encoder_postnet = nn.Linear(config.encoder_dim, config.lm_dim, bias=False)
        
        # lm
        from modules.asr.llama.llama_seq2seq import Seq2SeqLLaMA, ModelArgs as Seq2SeqLLaMAModelArgs
        self.lm_offset_encoder = SinusoidalPositionalEmbedding(config.lm_dim, proj=True)
        self.lm = Seq2SeqLLaMA(Seq2SeqLLaMAModelArgs(
            dim=config.lm_dim,
            n_heads=config.lm_n_heads,
            enc_n_layers=config.lm_enc_n_layers,
            dec_n_layers=config.lm_dec_n_layers,
            use_qk_norm=True,
        ))
        self.speech_start_token = nn.Parameter(torch.randn((1, 1, config.lm_dim)))
        self.lm_postnet = nn.Linear(config.lm_dim, config.decoder_dim, bias=False)
        self.stop_clf = StopClf(config.lm_dim, 1)
        
        # decoder
        if config.decoder_use_caption:
            from modules.tts.llama_dit.llama_ca import LLaMa as LLaMaDiTCA, ModelArgs as LLaMaDiTCAModelArgs
            self.decoder = LLaMaDiTCA(LLaMaDiTCAModelArgs(
                encoder_dim=config.decoder_dim,
                encoder_n_layers=config.decoder_n_layers,
                crossattn_n_layers=config.decoder_n_layers,
                encoder_n_heads=config.decoder_n_heads,
                ffn_dim_multiplier=4,
                use_qk_norm=True
            ))
        else:
            from modules.tts.llama_dit.llama import LLaMa as LLaMaDiT, ModelArgs as LLaMaDiTModelArgs
            self.decoder = LLaMaDiT(LLaMaDiTModelArgs(
                encoder_dim=config.decoder_dim,
                encoder_n_layers=config.decoder_n_layers,
                encoder_n_heads=config.decoder_n_heads,
                ffn_dim_multiplier=4,
                use_qk_norm=True
            ))
        self.decoder_prenet = nn.Linear(config.in_channels, config.decoder_dim)
        self.decoder_offset_encoder = SinusoidalPositionalEmbedding(config.decoder_dim, proj=True)
        self.ctx_mask_embed = nn.Embedding(2, config.decoder_dim)
        self.decoder_postnet = nn.Linear(config.decoder_dim, config.out_channels)
        
        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(config.decoder_dim)
        
    def forward(self, inputs):
        x: torch.Tensor = inputs['lat']
        bsz, device = x.size(0), x.device
        float_type = autocast_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
        x_lens = inputs['lat_lens']
        x_mask = sequence_mask(inputs['lat_lens'], maxlen=x.shape[1]).long()
        
        x_ling, x_ling_mask = self.forward_ling_encoder(inputs)
        h, h_mask = self.forward_encoder(x, x_mask)
        
        ######
        # lm #
        ######
        h_offset = self.lm_offset_encoder(
            positions=torch.arange(h.shape[1], device=device) * (4 * self.config.patch_size), 
            out_dtype=h.dtype, device=device
        )[None, ...].expand(h.shape[0], -1, -1)
        h_offset = add_prefix_nd(
            torch.zeros_like(x_ling), x_ling_mask.sum(1), h_offset, torch.LongTensor([h_offset.shape[1]] * h_offset.shape[0]).to(device)
        )
        h_offset = torch.cat([h_offset, torch.zeros_like(h_offset[:, :1])], dim=1)
        
        x_ling = add_prefix_nd(
            x_ling, x_ling_mask.sum(1), self.speech_start_token.to(x_ling).repeat(x_ling.shape[0], 1, 1), 
            torch.ones(x_ling.shape[0], dtype=torch.long, device=device)
        )
        x_ling_mask = sequence_mask(x_ling_mask.sum(1) + 1, maxlen=x_ling.shape[1])
        h = add_prefix_nd(x_ling, x_ling_mask.sum(1), h, h_mask.sum(1))
        h_mask = sequence_mask(x_ling_mask.sum(1) + h_mask.sum(1), maxlen=h.shape[1])
        
        h = h + h_offset
        h = self.lm(
            encoder_x=self.caption_lm_proj(inputs['caption_emb']),
            encoder_padding_mask=sequence_mask(inputs['caption_lens'], maxlen=inputs['caption_emb'].shape[1]),
            decoder_x=h,
            decoder_padding_mask=h_mask,
            do_checkpoint=self.config.do_checkpoint
        )
        h = remove_prefix(h, prefix_lens=x_ling_mask.sum(1) - 1, output_lens=h_mask.sum(1) - x_ling_mask.sum(1))
        h_mask = sequence_mask(h_mask.sum(1) - x_ling_mask.sum(1), maxlen=h.shape[1])
        
        stop_logits = self.stop_clf(h)[..., 0]
        stop_labels = last_token_mask(h_mask).to(stop_logits.dtype)
        # stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_labels, reduction='none')
        stop_loss = sigmoid_focal_loss(
            stop_logits, stop_labels, alpha=self.config.focal_loss_alpha, 
            gamma=self.config.focal_loss_gamma, reduction='none'
        )
        stop_loss = (stop_loss * h_mask).sum() / (h_mask).sum()
        
        h = self.lm_postnet(h)  # [B, T, C]
        
        #######
        # dit #
        #######
        h = h + self.decoder_offset_encoder(
            positions=torch.arange(h.shape[1], device=device), out_dtype=h.dtype, device=device
        )[None, ...]
        h = rearrange(h, 'b t c -> (b t) c')[:, None, :]    # [BxT, 1, C]
        ctx_n_patches, patch_size = self.config.ctx_n_patches, self.config.patch_size
        if ctx_n_patches > 0:
            z = torch.cat([x.new_full((x.shape[0], ctx_n_patches * patch_size, x.shape[2]), 0.0), x], dim=1).unfold(
                dimension=1, 
                size=(ctx_n_patches + 1) * patch_size,
                step=patch_size
            )   # [B, T, C, P]
            z_mask = torch.cat([x_mask.new_full((x_mask.shape[0], ctx_n_patches * patch_size), 1.0), x_mask], dim=1).unfold(
                dimension=1, 
                size=(ctx_n_patches + 1) * patch_size,
                step=patch_size
            )   # [B, T, P]
        else:
            z = x.unfold(dimension=1, size=patch_size, step=patch_size)   # [B, T, C, P]
            z_mask = x_mask.unfold(dimension=1, size=patch_size, step=patch_size)   # [B, T, P]
        z = rearrange(z, 'b t c p -> (b t) p c')    # [BxT, P, C]
        z_mask = rearrange(z_mask, 'b t p -> (b t) p')  # [BxT, P]
        
        ctx_mask = torch.zeros_like(z_mask)
        ctx_mask[:, :ctx_n_patches * patch_size] = 1
        
        # CFG
        ctx_cfg_mask = torch.rand_like(ctx_mask[:, 0].to(float_type))[:, None, None]  # [BxT, 1, 1]
        ctx_cfg_mask = (ctx_cfg_mask < 0.15).to(float_type)
        z = z * ctx_mask[..., None] * (1 - ctx_cfg_mask) + z * (1 - ctx_mask[..., None])
        h_cfg_mask = torch.rand_like(ctx_mask[:, 0].to(float_type))[:, None, None]    # [BxT, 1, 1]
        h_cfg_mask = (h_cfg_mask < 0.15).to(float_type)
        h = h * (1 - h_cfg_mask)
        
        z0 = torch.randn_like(z)
        t = self.flow_matcher.time_sampler.sample([z0.shape[0]], z0.device).type_as(z0)
        zt = t[:, None, None] * z + (1 - t[:, None, None]) * z0
        ut = z - z0
        
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t = self.f5_time_embed(t)
        z_noisy = zt * (1 - ctx_mask[..., None]) + z * ctx_mask[..., None]
        target = ut
        
        z_noisy = self.decoder_prenet(z_noisy)
        
        z_noisy = torch.cat([h.to(z_noisy.dtype), z_noisy], dim=1)    # [BxT, 1+P, C]
        z_mask = torch.cat([torch.ones_like(z_mask)[:, 0:1], z_mask], dim=1)
        ctx_mask = torch.cat([torch.ones_like(ctx_mask)[:, 0:1], ctx_mask], dim=1)
        
        z_noisy = z_noisy + self.ctx_mask_embed(ctx_mask.long()).to(z_noisy.dtype)
        
        if self.config.decoder_use_caption:
            
            # caption_emb_decoder = self.caption_decoder_proj(inputs['caption_emb']).to(z_noisy.dtype)
            # # caption_emb_decoder: torch.Tensor = caption_emb_decoder.repeat(z_noisy.shape[0] // caption_emb_decoder.shape[0], 1, 1)
            # caption_emb_decoder = caption_emb_decoder.unsqueeze(1).expand(-1, z_noisy.shape[0] // caption_emb_decoder.shape[0], -1, -1).flatten(0, 1)
            # caption_lens_decoder = inputs['caption_lens']
            # caption_lens_decoder = caption_lens_decoder.repeat(z_noisy.shape[0] // caption_lens_decoder.shape[0])
            
            factor = z_noisy.shape[0] // inputs['caption_emb'].shape[0]
            assert z_noisy.shape[0] % inputs['caption_emb'].shape[0] == 0, "z_noisy batch must be multiple of caption batch"
            caption_emb_decoder = self.caption_decoder_proj(inputs['caption_emb']).to(z_noisy.dtype)
            caption_emb_decoder = repeat(caption_emb_decoder, 'b l c -> (b r) l c', r=factor).contiguous()
            caption_lens_decoder = repeat(inputs['caption_lens'], 'b -> (b r)', r=factor).contiguous()
            
            cap_cfg_mask = (torch.rand(caption_lens_decoder.shape[0], device=caption_lens_decoder.device) < 0.15).to(float_type)[:, None, None]
            caption_emb_decoder = caption_emb_decoder * (1.0 - cap_cfg_mask)

        if 0 < self.config.training_patch_keep_ratio < 1:
            n_patches_keep = max(1, int(z.shape[0] * self.config.training_patch_keep_ratio))
            indices = torch.randperm(z.shape[0], device=z.device)[:n_patches_keep]
            z_noisy = z_noisy[indices]
            t = t[indices]
            z_mask = z_mask[indices]
            ctx_mask = ctx_mask[indices]
            target = target[indices]
            if self.config.decoder_use_caption:
                caption_emb_decoder = caption_emb_decoder[indices]
                caption_lens_decoder = caption_lens_decoder[indices]
        
        if self.config.decoder_use_caption:
            pred = self.decoder(
                z_noisy, t.to(z_noisy.dtype), z_mask, 
                context=caption_emb_decoder,
                context_lens=caption_lens_decoder,
                do_checkpoint=self.config.do_checkpoint
            )
        else:
            pred = self.decoder(z_noisy, t.to(z_noisy.dtype), z_mask, do_checkpoint=self.config.do_checkpoint)
        
        pred = self.decoder_postnet(pred)
        pred = pred[:, 1:]
        
        loss_mask = z_mask * (1 - ctx_mask)
        loss_mask = loss_mask[:, 1:, None]
        diff_loss = F.mse_loss(pred.float(), target.float(), reduction='none')
        diff_loss = (diff_loss * loss_mask).sum() / loss_mask.sum() / target.shape[-1]
        
        ret = {
            'stop_loss': stop_loss,
            'diff_loss': diff_loss,
            'pred': pred,
            'target': target,
            'loss_mask': loss_mask,
            'ctx_mask': ctx_mask,
            'stop_labels': stop_labels
        }
        
        return ret
        
    def forward_ling_encoder(self, inputs):
        # phone tone
        ph_mask = inputs["phone"] > 0
        ph_embed = self.ph_embed(inputs["phone"])
        tone_embed = self.tone_embed(inputs["tone"])
        x_ph = ph_embed + tone_embed
        x_ph = self.ph_encoder(x_ph, ph_mask)
        
        # duration
        x_dur = self.duration_encoder(inputs['dur'])
        ph_offset = self.ph_offset_encoder(
            positions=torch.cumsum(inputs['dur'].clamp_min(0), dim=1).long(), 
            out_dtype=x_ph.dtype, device=x_ph.device
        )
        x_ph = x_ph + x_dur + ph_offset
        
        # text
        txt_tokens = inputs["txt_tokens"]
        txt_embed = self.text_embedder(txt_tokens)
        txt_mask = inputs["txt_mask"]
        x_txt = self.text_encoder(txt_embed, attn_mask=txt_mask)
        
        x_ling = add_prefix_nd(x_txt, txt_mask.sum(1), x_ph, ph_mask.sum(1))
        x_mask = sequence_mask(txt_mask.sum(1) + ph_mask.sum(1), maxlen=x_ling.shape[1])
        
        x_ling = self.ling_postnet(x_ling)
                
        return x_ling, x_mask
    
    def forward_encoder(self, x, x_mask):        
        x = self.encoder_prenet(x)
        bsz = x.shape[0]

        x = rearrange(x, "b (t p) c -> (b t) p c", p=self.config.patch_size)
        x_mask = rearrange(x_mask, "b (t p) -> (b t) p", p=self.config.patch_size)
        
        x = torch.cat([self.encoder_cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x_mask = torch.cat([torch.ones((x.shape[0], 1)).to(x_mask), x_mask], dim=1)
        
        x = self.encoder(x, x_mask)
        
        x = x[:, 0:1, :]
        x_mask = x_mask[:, 0:1]
        
        x = rearrange(x, "(b t) p c -> b (t p) c", b=bsz)
        x_mask = rearrange(x_mask, "(b t) p -> b (t p)", b=bsz)
        
        x = self.encoder_postnet(x)
        
        return x, x_mask
        
    def inference(self, inputs, timesteps=20, seq_cfg_w=(1.4, 3), timestep_annealing_w=(1.0, 0.0, 1.0), 
                  start_pos=0, use_amo_sampler=True, use_tqdm=False, max_new_patches=1024):
        ref_x = inputs['ref_lat']
        bsz, device = ref_x.size(0), ref_x.device
        float_type = autocast_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else ref_x.dtype
        ref_x_lens = inputs['ref_lat_lens']
        ref_x_mask = sequence_mask(ref_x_lens, maxlen=ref_x.shape[1]).long()
        
        ref_txt_tokens = inputs['ref_txt_tokens']
        ref_txt_mask = inputs['ref_txt_mask']
        ref_phone = inputs['ref_phone']
        ref_tone = inputs['ref_tone']
        ref_dur = inputs['ref_dur']
        ref_phone_mask = inputs['ref_phone'] > 0
        
        txt_tokens = inputs['txt_tokens']
        txt_mask = inputs['txt_mask']
        phone = inputs['phone']
        tone = inputs['tone']
        dur = inputs['dur']
        phone_mask = inputs['phone'] > 0
        
        caption_emb = inputs['caption_emb']
        caption_lens = inputs['caption_lens']
        caption_mask = sequence_mask(caption_lens, maxlen=caption_emb.shape[1])
        
        txt_tokens = add_prefix(ref_txt_tokens, ref_txt_mask.sum(1), txt_tokens, txt_mask.sum(1))
        txt_mask = sequence_mask(ref_txt_mask.sum(1) + txt_mask.sum(1), maxlen=txt_tokens.shape[1])
        phone = add_prefix(ref_phone, ref_phone_mask.sum(1), phone, phone_mask.sum(1))
        tone = add_prefix(ref_tone, ref_phone_mask.sum(1), tone, phone_mask.sum(1))
        dur = add_prefix(ref_dur, ref_phone_mask.sum(1), dur, phone_mask.sum(1))
        phone_mask = sequence_mask(ref_phone_mask.sum(1) + phone_mask.sum(1), maxlen=phone.shape[1])
        
        x_ling, x_ling_mask = self.forward_ling_encoder({
            "phone": phone,
            "tone": tone,
            "dur": dur,
            "txt_tokens": txt_tokens,
            "txt_mask": txt_mask
        })
        ref_h, ref_h_mask = self.forward_encoder(ref_x, ref_x_mask)

        lm_enc_out = self.lm.encode(self.caption_lm_proj(caption_emb), caption_mask)
        
        h_offset = self.lm_offset_encoder(
            positions=torch.arange(ref_h.shape[1], device=device) * (4 * self.config.patch_size), 
            out_dtype=ref_h.dtype, device=device
        )[None, ...].expand(ref_h.shape[0], -1, -1)
        h_offset = add_prefix_nd(
            torch.zeros_like(x_ling), x_ling_mask.sum(1), h_offset, torch.LongTensor([h_offset.shape[1]] * h_offset.shape[0]).to(device)
        )
        h_offset = torch.cat([h_offset, torch.zeros_like(h_offset[:, :1])], dim=1)
        step_offset = ref_h.shape[1] - 1
        
        x_ling = add_prefix_nd(
            x_ling, x_ling_mask.sum(1), self.speech_start_token.to(x_ling).repeat(x_ling.shape[0], 1, 1), 
            torch.ones(x_ling.shape[0], dtype=torch.long, device=device)
        )
        x_ling_mask = sequence_mask(x_ling_mask.sum(1) + 1, maxlen=x_ling.shape[1])
        ref_h = add_prefix_nd(x_ling, x_ling_mask.sum(1), ref_h, ref_h_mask.sum(1))
        ref_h_mask = sequence_mask(x_ling_mask.sum(1) + ref_h_mask.sum(1), maxlen=ref_h.shape[1])
        
        # prefill
        ref_h = ref_h + h_offset
        _ = self.lm.decode(
            decoder_x=ref_h[:, :-1], decoder_padding_mask=None, enc_out=lm_enc_out,
            encoder_padding_mask=caption_mask, start_pos=start_pos, use_cache=True
        )
        start_pos = start_pos + ref_h.shape[1] - 1
        lm_input = ref_h[:, -1:]
        
        # dit ctx
        ctx_n_patches, patch_size = self.config.ctx_n_patches, self.config.patch_size
        if ctx_n_patches > 0:
            z_ctx = ref_x[:, -ctx_n_patches * patch_size:]
        if self.config.decoder_use_caption:
            caption_emb_decoder = self.caption_decoder_proj(caption_emb).to(float_type)
        
        def forward_step(lm_input, step, start_pos, z_ctx=None, caption_emb=None, caption_lens=None):
            lm_offset = self.lm_offset_encoder(
                timestep=step * (4 * patch_size), out_dtype=lm_input.dtype, device=lm_input.device
            )[None, None, ...]
            lm_input = lm_input + lm_offset     # [B, 1, C]
            lm_output = self.lm.decode(
                decoder_x=lm_input, decoder_padding_mask=None, enc_out=lm_enc_out, 
                encoder_padding_mask=caption_mask, start_pos=start_pos, use_cache=True
            )
            stop_logits = self.stop_clf(lm_output)[..., 0]  # [B, 1]
            stop = torch.sigmoid(stop_logits) > 0.9
            
            lm_output = self.lm_postnet(lm_output)  # [B, 1, C]
            
            decoder_offset = self.decoder_offset_encoder.forward(
                timestep=step, out_dtype=lm_output.dtype, device=lm_output.device
            )[None, None, ...]
            lm_output = lm_output + decoder_offset
            
            h = torch.cat([
                lm_output,
                lm_output,
                torch.zeros_like(lm_output)
            ], dim=0).to(float_type)
            
            if z_ctx is not None:
                z_ctx = torch.cat([
                    z_ctx,
                    torch.zeros_like(z_ctx),
                    torch.zeros_like(z_ctx),
                ], dim=0)
            
            t_schedule = torch.linspace(0, 1, timesteps + 1).to(device)
            t_schedule = 0.5 * (1 - torch.cos(torch.pi * t_schedule))
            tgt_len = patch_size
            
            cond = {
                'z_ctx': z_ctx,
                'h': h,
            }
            if self.config.decoder_use_caption:
                caption_emb = torch.cat([
                    caption_emb,
                    torch.zeros_like(caption_emb),
                    torch.zeros_like(caption_emb)
                ], dim=0)
                caption_lens = torch.cat([caption_lens] * 3)
                cond['caption_emb'] = caption_emb
                cond['caption_lens'] = caption_lens
            
            if use_amo_sampler:
                x = torch.randn([bsz, tgt_len, self.hp.out_channels], device=device)                
                for step_index in range(timesteps):
                    sigma = t_schedule[step_index].to(float_type)
                    sigma_next = t_schedule[step_index + 1]
                    model_out = forward_dit_step(
                        torch.cat([x] * 3), cond, t=sigma.unsqueeze(0), 
                        seq_cfg_w=seq_cfg_w, timestep_annealing_w=timestep_annealing_w
                    )
                    x = amo_sampling(x, sigma, sigma_next, model_out)
            
            else:
                traj = torchdiffeq.odeint(
                    lambda t, x: forward_dit_step(
                        torch.cat([x] * 3), cond, t=t.unsqueeze(0), 
                        seq_cfg_w=seq_cfg_w, timestep_annealing_w=timestep_annealing_w
                    ),
                    torch.randn([bsz, tgt_len, self.hp.out_channels], device=device),
                    t_schedule,
                    atol=1e-4,
                    rtol=1e-4,
                    method="euler",
                )
                x = traj[-1]
            
            return x, stop, torch.sigmoid(stop_logits)
            
        def forward_dit_step(x, cond, t, seq_cfg_w=(1.4, 3), timestep_annealing_w=(1.0, 0.0, 1.0)):
            z_ctx = cond['z_ctx']
            h = cond['h']
            if self.config.decoder_use_caption:
                caption_emb = cond['caption_emb']
                caption_lens = cond['caption_lens']

            if z_ctx is not None:
                x = torch.cat([z_ctx, x], dim=1)
            x = self.decoder_prenet(x)
            x = torch.cat([h, x], dim=1)
            x_mask = torch.ones_like(x[..., 0]).long()
            ctx_mask = torch.zeros_like(x_mask)
            if z_ctx is not None:
                ctx_mask[:, :1 + z_ctx.shape[1]] = 1
            else:
                ctx_mask[:, :1] = 1
            
            x = x + self.ctx_mask_embed(ctx_mask.long()).to(x.dtype)
            
            with torch.amp.autocast('cuda', dtype=torch.float32):
                t_embed = self.f5_time_embed(t).expand(x.shape[0], -1)
                
            if self.config.decoder_use_caption:
                pred = self.decoder(
                    x, t_embed.to(x.dtype), x_mask,
                    context=caption_emb, context_lens=caption_lens
                )
            else:
                pred = self.decoder(x, t_embed.to(x.dtype), x_mask)
                
            if z_ctx is not None:
                pred = pred[:, 1 + z_ctx.shape[1]:]
            else:
                pred = pred[:, 1:]
            pred = self.decoder_postnet(pred)
            
            a, b, p = timestep_annealing_w
            gamma_t = a + b * torch.pow(1 - t, p)
            seq_cfg_w = [gamma_t * w for w in seq_cfg_w]
            
            cond_all, cond_txt, uncond = pred.chunk(3)
            pred = (
                uncond + 
                seq_cfg_w[0] * (cond_txt - uncond) + 
                seq_cfg_w[1] * (cond_all - cond_txt)
            )
                        
            return pred
        
        if use_tqdm:
            from tqdm import tqdm
            it = tqdm(range(max_new_patches), desc='| Generating Speech')
        else:
            it = range(max_new_patches)
            
        stop_logits_lst = []
        res_patches = []
        for step in it:
            if self.config.decoder_use_caption:
                x_patch, stop, stop_logits = forward_step(
                    lm_input, step + step_offset, start_pos, 
                    z_ctx[:, -ctx_n_patches * patch_size:] if ctx_n_patches > 0 else None, 
                    caption_emb_decoder, caption_lens
                )   # [B, T, C]
            else:
                x_patch, stop, stop_logits = forward_step(
                    lm_input, step + step_offset, start_pos, 
                    z_ctx[:, -ctx_n_patches * patch_size:] if ctx_n_patches > 0 else None
                )   # [B, T, C]
            h_patch, h_patch_mask = self.forward_encoder(x_patch, torch.ones_like(x_patch[..., 0]).long())
            lm_input = h_patch
            start_pos = start_pos + 1
            stop_logits_lst.append(stop_logits.item())
            
            if ctx_n_patches > 0:
                z_ctx = torch.cat([z_ctx, x_patch], dim=1)
            res_patches.append(x_patch)
            
            if stop.all():
                break
            
        res_lat = torch.cat(res_patches, dim=1)
        
        return res_lat, stop_logits_lst
            
        