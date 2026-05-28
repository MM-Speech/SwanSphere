import logging
import math
import random
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# import attrdict
from einops import rearrange, repeat
import torchdiffeq

from modules.tts.ar_dur.commons.align_ops import expand_states
from modules.commons.hf.transformer import TransformerEncoderModel, TransformerDecoderModel
from modules.commons.hf.transformer_config import TransformerConfig
from modules.commons.hf.transformer_dit import TransformerDiTModel
from modules.commons.hf.transformer_dit_config import TransformerDiTConfig

from utils.nn.seq_utils import sequence_mask, add_prefix, add_prefix_nd, remove_prefix, last_token_mask, build_last_k_soft_labels
from utils.nn.generation_utils import amo_sampling
from utils.losses.focal_loss import sigmoid_focal_loss
from utils.commons.io import print_once
from utils.nn.seq_utils import get_incremental_state, set_incremental_state, softmax, make_positions


"""
加入 policy latent; 替换为 transformers
"""

@dataclass
class ModelArgs:    
    # text -> video token
    text_vocab_size: int = None
    text_dim: int = 768 # 1024
    video_input_dim: int = 512

    # audio
    patch_size: int = 4
    ctx_n_patches: int = 2
    add_vad_mask: bool = False

    # caption
    caption_dim: int = 1024
    use_caption_encoder: bool = False
    crossattn_n_layers: int = 24
    
    # encoder
    encoder_dim: int = 768 # 1024
    encoder_n_layers: int = 4 # 6
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = 8
    
    # lm
    lm_dim: int = 768 # 1024
    lm_enc_n_layers: int = 2
    lm_dec_n_layers: int = 24 # 36
    lm_n_heads: int = 16
    lm_n_kv_heads: int = 8
    
    # stop clf
    focal_loss_alpha: float = 0.99
    focal_loss_gamma: float = 1.5
    
    # decoder
    decoder_dim: int = 768 # 1024
    decoder_n_layers: int = 4 # 6
    decoder_n_heads: int = 16
    decoder_n_kv_heads: int = 8
    training_patch_keep_ratio: float = -1.0
    
    # latent
    in_channels: int = 128
    out_channels: int = 128

    # trainging
    do_checkpoint: bool = False
    warm_up_lm: bool = False
    attn_implementation: str = 'flash_attention_2'
    instruction_finetuning: bool = False
    

class StopClf(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_size=256):
        super().__init__()
        self.linear1 = nn.Linear(input_channels, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear3 = nn.Linear(hidden_size, output_channels, bias=False)
    
    def forward(self, x):
        x = F.silu(self.linear1(x))
        x = F.silu(self.linear2(x))
        x = self.linear3(x)
        return x

    
class DiTARModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = self.hp = config
        
        # linguistic
        # self.text_embedder = nn.Embedding(config.text_vocab_size, config.text_dim)
        self.video_proj = nn.Linear(config.video_input_dim, config.lm_dim)
        self.video_encoder = TransformerEncoderModel(TransformerConfig(
            vocab_size=0, hidden_size=config.lm_dim, 
            intermediate_size=config.lm_dim * 4, num_hidden_layers=4, 
            num_attention_heads=16, num_key_value_heads=8, head_dim=config.lm_dim // 16,
            attn_implementation=config.attn_implementation
        ))
        
        self.video_postnet = nn.Linear(config.text_dim, config.lm_dim, bias=False)
        
        
        # semantic
        if config.use_caption_encoder:
            self.caption_lm_proj = nn.Linear(config.caption_dim, config.lm_dim)
            self.caption_decoder_proj = nn.Linear(config.caption_dim, config.decoder_dim)
            self.caption_encoder = TransformerEncoderModel(TransformerConfig(
                vocab_size=0, hidden_size=config.lm_dim, 
                intermediate_size=config.lm_dim * 4, num_hidden_layers=config.lm_enc_n_layers, 
                num_attention_heads=16, num_key_value_heads=8, head_dim=config.lm_dim // config.lm_n_heads,
                attn_implementation=config.attn_implementation
            ))
        
        # encoder 编码latent，得到初步的h
        self.encoder_prenet = nn.Linear(config.in_channels, config.encoder_dim)
        self.encoder = TransformerEncoderModel(TransformerConfig(
            vocab_size=0, hidden_size=config.encoder_dim, 
            intermediate_size=config.encoder_dim * 4, num_hidden_layers=config.encoder_n_layers, 
            num_attention_heads=config.encoder_n_heads, num_key_value_heads=config.encoder_n_kv_heads, 
            head_dim=config.encoder_dim // config.encoder_n_heads,
            attn_implementation=config.attn_implementation
        ))
        self.encoder_cls_token = nn.Parameter(torch.randn((1, 1, config.encoder_dim)))
        self.encoder_postnet = nn.Linear(config.encoder_dim, config.lm_dim, bias=False)
        
        # lm 进一步处理 h
        self.lm = TransformerDecoderModel(TransformerConfig(
            vocab_size=0, hidden_size=config.lm_dim, 
            intermediate_size=config.lm_dim * 4, num_hidden_layers=config.lm_dec_n_layers, 
            num_attention_heads=config.lm_n_heads, num_key_value_heads=config.lm_n_kv_heads, 
            head_dim=config.lm_dim // config.lm_n_heads, 
            use_dynamic_cross_gate=True, use_gated_attention=True,
            num_cross_attention_layers=config.lm_dec_n_layers if config.use_caption_encoder else 0,
            attn_implementation=config.attn_implementation
        ))
        self.lm.checkpoint_activations = config.do_checkpoint
        self.speech_start_token = nn.Parameter(torch.randn((1, 1, config.lm_dim)))
        self.lm_head = nn.Linear(config.lm_dim, config.lm_dim * 2)
        self.lm_postnet = nn.Linear(config.lm_dim, config.decoder_dim, bias=False)
        self.stop_clf = StopClf(config.lm_dim, 1)
        if config.warm_up_lm:
            self.lm_input = nn.Linear(config.in_channels, config.lm_dim, bias=False)
            self.lm_output = nn.Linear(config.lm_dim, config.out_channels, bias=False)
            
        # decoder  dit部分
        self.decoder = TransformerDiTModel(TransformerDiTConfig(
            hidden_size=config.decoder_dim,
            intermediate_size=config.decoder_dim * 4, num_hidden_layers=config.decoder_n_layers,
            num_attention_heads=config.decoder_n_heads, num_key_value_heads=config.decoder_n_kv_heads,
            head_dim=config.decoder_dim // config.decoder_n_heads,
            use_dynamic_cross_gate=True,
            num_cross_attention_layers=config.decoder_n_layers if config.use_caption_encoder else 0,
            attn_implementation=config.attn_implementation, is_decoder=True
        ))
        self.decoder.gradient_checkpointing = config.do_checkpoint
        self.decoder_prenet = nn.Linear(config.in_channels, config.decoder_dim)
        self.ctx_mask_embed = nn.Embedding(2, config.decoder_dim)
        self.decoder_postnet = nn.Linear(config.decoder_dim, config.out_channels)
        
        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(config.decoder_dim)
        
    def forward(self, inputs):
        if self.config.warm_up_lm:
            raise NotImplementedError
        
        x: torch.Tensor = inputs['lat']
        bsz, device = x.size(0), x.device
        float_type = autocast_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
        x_lens = inputs['lat_lens']
        x_mask = sequence_mask(inputs['lat_lens'], maxlen=x.shape[1]).long() # 这个是attn mask
        
        x_vid, x_vid_mask = self.forward_ling_encoder(inputs)
        h, h_mask = self.forward_encoder(x, x_mask)
        
        if self.config.instruction_finetuning:
            raise NotImplementedError
        
        ######
        # lm #
        ######
        # [对齐x_vid和h]
        T_vid = x_vid.shape[1]
        T_aud = h.shape[1]  # 这是目标长度 (Patch级)
        step_indices = torch.arange(T_aud, device=device).float()
        
        vid_indices = (step_indices * T_vid / T_aud).floor().long()
        vid_indices = vid_indices.clamp(max=T_vid - 1)
        x_vid = x_vid[:, vid_indices, :]
        
        
        # x_vid = add_prefix_nd(
        #     x_vid, x_vid_mask.sum(1), self.speech_start_token.to(x_vid).repeat(x_vid.shape[0], 1, 1), 
        #     torch.ones(x_vid.shape[0], dtype=torch.long, device=device)
        # )
        # x_vid_mask = sequence_mask(x_vid_mask.sum(1) + 1, maxlen=x_vid.shape[1])
        # h = add_prefix_nd(x_vid, x_vid_mask.sum(1), h, h_mask.sum(1))
        # h_mask = sequence_mask(x_vid_mask.sum(1) + h_mask.sum(1), maxlen=h.shape[1])
        start_token = self.speech_start_token.expand(bsz, 1, -1)
        h_shifted = torch.cat([start_token, h[:, :-1, :]], dim=1)
        start_mask = torch.ones((bsz, 1), device=device, dtype=h_mask.dtype)
        h_shifted_mask = torch.cat([start_mask, h_mask[:, :-1]], dim=1)
        h = h_shifted + x_vid
        # h = h + x_vid
        
        caption_emb, caption_mask, context_lens = None, None, None
        if self.config.use_caption_encoder and inputs.get('caption_emb') is not None:
            caption_emb = self.caption_lm_proj(inputs['caption_emb'])
            caption_mask = sequence_mask(inputs['caption_lens'], maxlen=inputs['caption_emb'].shape[1])
            caption_emb = self.caption_encoder(inputs_embeds=caption_emb, attention_mask=caption_mask).last_hidden_state
            context_lens=caption_mask.sum(1)
        
        h = self.lm(
            inputs_embeds=h, attention_mask=h_mask,
            encoder_hidden_states=caption_emb,
            encoder_attention_mask=caption_mask
        ).last_hidden_state
        # h = remove_prefix(h, prefix_lens=x_vid_mask.sum(1) - 1, output_lens=h_mask.sum(1) - x_vid_mask.sum(1))
        # h_mask = sequence_mask(h_mask.sum(1) - x_vid_mask.sum(1), maxlen=h.shape[1])
        
        h = self.lm_head(h)  # [B, T, 2 * C]
        h_mu, h_log_sigma = torch.chunk(h, 2, dim=-1)
        h_sigma = torch.exp(h_log_sigma.clamp(-20, 20))     # [B, T, C]
        eps = torch.randn_like(h_mu)
        h = h_mu + h_sigma * eps
        log_prob = -0.5 * (((h - h_mu) / h_sigma)**2 + 2*h_log_sigma + math.log(2*math.pi)).sum(-1)

        stop_logits = self.stop_clf(h)[..., 0]
        stop_labels = build_last_k_soft_labels(h_mask, K=4, gamma=2).to(stop_logits.dtype)
        # stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_labels, reduction='none')
        # if self.config.instruction_finetuning:
        #     stop_loss = (stop_loss * h_mask * (1 - prefix_mask)).sum() / (h_mask * (1 - prefix_mask)).sum()
        # else:
        #     stop_loss = (stop_loss * h_mask).sum() / (h_mask).sum()
        stop_loss = 0.0

        h = self.lm_postnet(h)  # [B, T, C]
        
        #######
        # dit #
        #######
        h = rearrange(h, 'b t c -> (b t) c')[:, None, :]    # [BxT, 1, C]
        if self.config.instruction_finetuning:
            prefix_mask = rearrange(prefix_mask, 'b t -> (b t) 1')  # [BxT, 1]
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
        
        if 0 < self.config.training_patch_keep_ratio < 1:
            n_patches_keep = max(1, int(z.shape[0] * self.config.training_patch_keep_ratio))
            indices = torch.randperm(z.shape[0], device=z.device)[:n_patches_keep]
            z_noisy = z_noisy[indices]
            t = t[indices]
            z_mask = z_mask[indices]
            ctx_mask = ctx_mask[indices]
            target = target[indices]
        
        pred = self.decoder(
            inputs_embeds=z_noisy, 
            time_step=t.to(z_noisy.dtype), 
            attention_mask=z_mask
        ).last_hidden_state
        
        pred = self.decoder_postnet(pred)
        pred = pred[:, 1:]
        
        loss_mask = z_mask * (1 - ctx_mask)
        loss_mask = loss_mask[:, 1:, None]
        if self.config.instruction_finetuning:
            loss_mask = loss_mask * (1 - prefix_mask[..., None])   # [BxT, P, 1] * [BxT, 1, 1]
        diff_loss = F.mse_loss(pred.float(), target.float(), reduction='none')
        diff_loss = (diff_loss * loss_mask).sum() / loss_mask.sum() / target.shape[-1]
        
        ret = {
            'stop_loss': stop_loss,
            'diff_loss': diff_loss,
            'pred': pred,
            'target': target,
            'loss_mask': loss_mask,
            'ctx_mask': ctx_mask,
            'stop_labels': stop_labels,
            'h_sigma': h_sigma.mean().detach(),
        }
        
        return ret
        
    def forward_ling_encoder(self, inputs):
        video_feats = inputs["video_feats"]
        # global_video_feats = inputs["global_clip_embedding"]
        # video_feats = torch.cat([video_feats, global_video_feats], dim=-1)
        v_embed = self.video_proj(video_feats)
        v_mask = inputs["v_mask"]
        x_vid = self.video_encoder(inputs_embeds=v_embed, attention_mask=v_mask).last_hidden_state
        x_vid = self.video_postnet(x_vid)       
        return x_vid, v_mask
    
    def forward_encoder(self, x, x_mask):        
        x = self.encoder_prenet(x)
        bsz = x.shape[0]

        x = rearrange(x, "b (t p) c -> (b t) p c", p=self.config.patch_size)
        x_mask = rearrange(x_mask, "b (t p) -> (b t) p", p=self.config.patch_size)
        
        x = torch.cat([self.encoder_cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x_mask = torch.cat([torch.ones((x.shape[0], 1)).to(x_mask), x_mask], dim=1)
        
        x = self.encoder(inputs_embeds=x, attention_mask=x_mask).last_hidden_state
        
        x = x[:, 0:1, :]
        x_mask = x_mask[:, 0:1]
        
        x = rearrange(x, "(b t) p c -> b (t p) c", b=bsz)
        x_mask = rearrange(x_mask, "(b t) p -> b (t p)", b=bsz)
        
        x = self.encoder_postnet(x)
        
        return x, x_mask
    
    def inference(self, inputs, timesteps=20, seq_cfg_w=(1.4, 3), timestep_annealing_w=(1.0, 0.0, 1.0), 
                  past_key_values=None, use_tqdm=False, max_new_patches=None, temperature=0.0):
        video_feats = inputs['video_feats']
        # 确保 mask 类型为 long
        v_mask = inputs.get('v_mask', torch.ones(video_feats.shape[:2], device=video_feats.device).long())
        bsz, device = video_feats.size(0), video_feats.device
        
        global_video_feats = inputs["global_clip_embedding"]
        
        # 编码视频
        x_vid, _ = self.forward_ling_encoder({"video_feats": video_feats, "v_mask": v_mask, "global_clip_embedding": global_video_feats})
        T_vid_raw = x_vid.shape[1]

        # 计算目标音频长度
        if max_new_patches is not None:
            T_aud = max_new_patches
        else:
            scale_factor = getattr(self.config, 'patches_per_frame', 21.53 / (4.0 * self.config.patch_size))
            T_aud = max(1, math.ceil(T_vid_raw * scale_factor))

        # 显式对齐
        step_indices = torch.arange(T_aud, device=device).float()
        vid_indices = (step_indices * T_vid_raw / T_aud).floor().long().clamp(max=T_vid_raw - 1)
        x_vid_aligned = x_vid[:, vid_indices, :] # [B, T_aud, C]

        # ===========================
        # 2. 初始化状态
        # ===========================
        # A. LM 初始输入: Start + Video[0]
        current_token = self.speech_start_token.expand(bsz, 1, -1)
        lm_input = current_token + x_vid_aligned[:, 0:1, :]
        
        # B. DiT 上下文
        ctx_n_patches, patch_size = self.config.ctx_n_patches, self.config.patch_size
        z_ctx = torch.zeros(bsz, ctx_n_patches * patch_size, self.config.out_channels, device=device)
        
        # C. KV Cache
        past_key_values = None

        # ===========================
        # 3. 定义内部函数 (闭包)
        # ===========================

        # --- 函数 A: LM Step (找回了这个函数) ---
        def forward_step(lm_input, past_key_values):
            # 1. 构造 Mask
            att_mask = torch.ones(lm_input.shape[:2], device=device, dtype=torch.long)
            
            # 2. LM Forward
            lm_output = self.lm(
                inputs_embeds=lm_input, 
                attention_mask=att_mask,
                past_key_values=past_key_values, 
                use_cache=True, 
                encoder_hidden_states=None, encoder_attention_mask=None
            )
            
            h = lm_output.last_hidden_state
            new_past_key_values = lm_output.past_key_values

            # 3. Heads & Sampling
            h = self.lm_head(h)
            h_mu, h_log_sigma = torch.chunk(h, 2, dim=-1)
            h_sigma = torch.exp(h_log_sigma.clamp(-20, 20))
            
            if temperature > 1e-5:
                eps = torch.randn_like(h_mu)
                h = h_mu + h_sigma * eps * temperature
            else:
                h = h_mu
                
            # 4. Project to DiT Dimension
            h_cond = self.lm_postnet(h)
            
            return h_cond, new_past_key_values

        # --- 函数 B: DiT Derivative (ODE Solver 需要) ---
        def forward_dit_step(x, cond, t, seq_cfg_w=(1.4, 3), timestep_annealing_w=(1.0, 0.0, 1.0)):
            z_ctx_inner = cond['z_ctx']
            h_inner = cond['h']

            # 拼接 Context
            if z_ctx_inner is not None:
                # 扩充为 3 倍 Batch 以匹配输入 x (CFG)
                z_ctx_expanded = torch.cat([z_ctx_inner] * 3, dim=0)
                x = torch.cat([z_ctx_expanded, x], dim=1)
            
            x = self.decoder_prenet(x)
            
            # 拼接 Condition
            h_expanded = torch.cat([h_inner] * 3, dim=0)
            x = torch.cat([h_expanded, x], dim=1)
            
            # Masking
            x_mask = torch.ones_like(x[..., 0]).long()
            ctx_mask = torch.zeros_like(x_mask)
            if z_ctx_inner is not None:
                ctx_mask[:, :1 + z_ctx_inner.shape[1]] = 1
            else:
                ctx_mask[:, :1] = 1 
            x = x + self.ctx_mask_embed(ctx_mask.long()).to(x.dtype)
            
            with torch.amp.autocast('cuda', dtype=torch.float32):
                t_embed = self.f5_time_embed(t).expand(x.shape[0], -1)
            
            # DiT Forward
            pred = self.decoder(
                inputs_embeds=x, time_step=t_embed.to(x.dtype), attention_mask=x_mask
            ).last_hidden_state
            
            # Slice
            if z_ctx_inner is not None:
                pred = pred[:, 1 + z_ctx_inner.shape[1]:]
            else:
                pred = pred[:, 1:]
            
            pred = self.decoder_postnet(pred)
            
            # CFG
            cond_all, cond_txt, uncond = pred.chunk(3)
            a, b, p = timestep_annealing_w
            gamma_t = a + b * torch.pow(1 - t, p)
            seq_cfg_w_t = [gamma_t * w for w in seq_cfg_w]
            
            grad = uncond + seq_cfg_w_t[0] * (cond_txt - uncond) + seq_cfg_w_t[1] * (cond_all - cond_txt)
            return grad

        # ===========================
        # 4. 主生成循环
        # ===========================
        iterator = range(T_aud)
        if use_tqdm:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc='| Generating Spatial Audio')
            
        res_patches = []
        
        for step in iterator:
            # === Step A: 调用 forward_step (LM 推理) ===
            h, past_key_values = forward_step(lm_input, past_key_values)
            
            # === Step B: 调用 ODE Solver (DiT 生成) ===
            cond = {'z_ctx': z_ctx, 'h': h}
            
            t_schedule = torch.linspace(0, 1, timesteps + 1, device=device)
            t_schedule = 0.5 * (1 - torch.cos(torch.pi * t_schedule))
            
            z_init = torch.randn([bsz, patch_size, self.config.out_channels], device=device)
            
            traj = torchdiffeq.odeint(
                lambda t, x: forward_dit_step(
                    torch.cat([x] * 3), cond, t=t.unsqueeze(0), 
                    seq_cfg_w=seq_cfg_w, timestep_annealing_w=timestep_annealing_w
                ),
                z_init, 
                t_schedule, 
                atol=1e-4, rtol=1e-4, method="euler",
            )
            x_patch = traj[-1]
            res_patches.append(x_patch)
            
            # === Step C: 状态更新 ===
            if step < T_aud - 1:
                # 1. Encode Audio History
                h_patch, _ = self.forward_encoder(x_patch, torch.ones_like(x_patch[..., 0]).long())
                
                # 2. Get Next Video Feature
                next_vid_feat = x_vid_aligned[:, step+1 : step+2, :]
                
                # 3. Prepare Next Input (Audio[t] + Video[t+1])
                lm_input = h_patch + next_vid_feat
                
                # import pdb; pdb.set_trace()
                
                # 4. Update Context
                if ctx_n_patches > 0:
                    z_ctx = torch.cat([z_ctx, x_patch], dim=1)
                    z_ctx = z_ctx[:, -ctx_n_patches * patch_size:]
        
        res_lat = torch.cat(res_patches, dim=1)
        return res_lat, []
    
    def inference_stream(self, inputs, timesteps=20, seq_cfg_w=(1.4, 3), timestep_annealing_w=(1.0, 0.0, 1.0), 
                  past_key_values=None, use_tqdm=False, max_new_patches=None, temperature=0.0):
        video_feats = inputs['video_feats']
        # 确保 mask 类型为 long
        v_mask = inputs.get('v_mask', torch.ones(video_feats.shape[:2], device=video_feats.device).long())
        bsz, device = video_feats.size(0), video_feats.device
        
        global_video_feats = inputs["global_clip_embedding"]
        
        # 编码视频
        x_vid, _ = self.forward_ling_encoder({"video_feats": video_feats, "v_mask": v_mask, "global_clip_embedding": global_video_feats})
        T_vid_raw = x_vid.shape[1]

        # 计算目标音频长度
        if max_new_patches is not None:
            T_aud = max_new_patches
        else:
            scale_factor = getattr(self.config, 'patches_per_frame', 21.53 / (4.0 * self.config.patch_size))
            T_aud = max(1, math.ceil(T_vid_raw * scale_factor))

        # 显式对齐
        step_indices = torch.arange(T_aud, device=device).float()
        vid_indices = (step_indices * T_vid_raw / T_aud).floor().long().clamp(max=T_vid_raw - 1)
        x_vid_aligned = x_vid[:, vid_indices, :] # [B, T_aud, C]

        # ===========================
        # 2. 初始化状态
        # ===========================
        # A. LM 初始输入: Start + Video[0]
        current_token = self.speech_start_token.expand(bsz, 1, -1)
        lm_input = current_token + x_vid_aligned[:, 0:1, :]
        
        # B. DiT 上下文
        ctx_n_patches, patch_size = self.config.ctx_n_patches, self.config.patch_size
        z_ctx = torch.zeros(bsz, ctx_n_patches * patch_size, self.config.out_channels, device=device)
        
        # C. KV Cache
        past_key_values = None

        # ===========================
        # 3. 定义内部函数 (闭包)
        # ===========================

        # --- 函数 A: LM Step (找回了这个函数) ---
        def forward_step(lm_input, past_key_values):
            # 1. 构造 Mask
            att_mask = torch.ones(lm_input.shape[:2], device=device, dtype=torch.long)
            
            # 2. LM Forward
            lm_output = self.lm(
                inputs_embeds=lm_input, 
                attention_mask=att_mask,
                past_key_values=past_key_values, 
                use_cache=True, 
                encoder_hidden_states=None, encoder_attention_mask=None
            )
            
            h = lm_output.last_hidden_state
            new_past_key_values = lm_output.past_key_values

            # 3. Heads & Sampling
            h = self.lm_head(h)
            h_mu, h_log_sigma = torch.chunk(h, 2, dim=-1)
            h_sigma = torch.exp(h_log_sigma.clamp(-20, 20))
            
            if temperature > 1e-5:
                eps = torch.randn_like(h_mu)
                h = h_mu + h_sigma * eps * temperature
            else:
                h = h_mu
                
            # 4. Project to DiT Dimension
            h_cond = self.lm_postnet(h)
            
            return h_cond, new_past_key_values

        # --- 函数 B: DiT Derivative (ODE Solver 需要) ---
        def forward_dit_step(x, cond, t, seq_cfg_w=(1.4, 3), timestep_annealing_w=(1.0, 0.0, 1.0)):
            z_ctx_inner = cond['z_ctx']
            h_inner = cond['h']

            # 拼接 Context
            if z_ctx_inner is not None:
                # 扩充为 3 倍 Batch 以匹配输入 x (CFG)
                z_ctx_expanded = torch.cat([z_ctx_inner] * 3, dim=0)
                x = torch.cat([z_ctx_expanded, x], dim=1)
            
            x = self.decoder_prenet(x)
            
            # 拼接 Condition
            h_expanded = torch.cat([h_inner] * 3, dim=0)
            x = torch.cat([h_expanded, x], dim=1)
            
            # Masking
            x_mask = torch.ones_like(x[..., 0]).long()
            ctx_mask = torch.zeros_like(x_mask)
            if z_ctx_inner is not None:
                ctx_mask[:, :1 + z_ctx_inner.shape[1]] = 1
            else:
                ctx_mask[:, :1] = 1 
            x = x + self.ctx_mask_embed(ctx_mask.long()).to(x.dtype)
            
            with torch.amp.autocast('cuda', dtype=torch.float32):
                t_embed = self.f5_time_embed(t).expand(x.shape[0], -1)
            
            # DiT Forward
            pred = self.decoder(
                inputs_embeds=x, time_step=t_embed.to(x.dtype), attention_mask=x_mask
            ).last_hidden_state
            
            # Slice
            if z_ctx_inner is not None:
                pred = pred[:, 1 + z_ctx_inner.shape[1]:]
            else:
                pred = pred[:, 1:]
            
            pred = self.decoder_postnet(pred)
            
            # CFG
            cond_all, cond_txt, uncond = pred.chunk(3)
            a, b, p = timestep_annealing_w
            gamma_t = a + b * torch.pow(1 - t, p)
            seq_cfg_w_t = [gamma_t * w for w in seq_cfg_w]
            
            grad = uncond + seq_cfg_w_t[0] * (cond_txt - uncond) + seq_cfg_w_t[1] * (cond_all - cond_txt)
            return grad

        # ===========================
        # 4. 流式生成
        # ===========================
        iterator = range(T_aud)
        
        for step in iterator:
            if step == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t0 = time.time()

            h, past_key_values = forward_step(lm_input, past_key_values)

            if step == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t1 = time.time()

            cond = {'z_ctx': z_ctx, 'h': h}
            t_schedule = torch.linspace(0, 1, timesteps + 1, device=device)
            t_schedule = 0.5 * (1 - torch.cos(torch.pi * t_schedule))

            z_init = torch.randn([bsz, patch_size, self.config.out_channels], device=device)

            traj = torchdiffeq.odeint(
                lambda t, x: forward_dit_step(
                    torch.cat([x] * 3), cond, t=t.unsqueeze(0),
                    seq_cfg_w=seq_cfg_w, timestep_annealing_w=timestep_annealing_w
                ),
                z_init,
                t_schedule,
                atol=1e-4, rtol=1e-4, method="euler",
            )
            x_patch = traj[-1]  # 拿到当前步生成的音频 Patch

            if step == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t2 = time.time()
                self._stream_first_patch_timing = {
                    'lm_and_before': _t1 - _t0,
                    'dit_after_lm': _t2 - _t1,
                }

            # 立即 Yield 当前生成的 patch
            yield x_patch
            if step < T_aud - 1:
                # 1. Encode Audio History
                h_patch, _ = self.forward_encoder(x_patch, torch.ones_like(x_patch[..., 0]).long())
                
                # 2. Get Next Video Feature
                next_vid_feat = x_vid_aligned[:, step+1 : step+2, :]
                
                # 3. Prepare Next Input
                lm_input = h_patch + next_vid_feat
                
                # 4. Update Context
                if ctx_n_patches > 0:
                    z_ctx = torch.cat([z_ctx, x_patch], dim=1)
                    z_ctx = z_ctx[:, -ctx_n_patches * patch_size:]
        
if __name__ == '__main__':
    
    import time 
    
    config = ModelArgs()
    config.text_vocab_size = 65536
    config.use_caption_encoder = True

    device = torch.device('cuda:0')

    model = DiTARModel(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"{params / 1024 /1024} MB")

    inputs = {
        'lat': torch.randn(2, 216, 128, device=device),
        'lat_lens': torch.tensor([215, 215], device=device),
        'video_feats': torch.randn(2, 40, 512, device=device),
        'v_mask': torch.ones(2, 40, device=device),
        
        'global_clip_embedding': torch.randn(2, 40, 512, device=device),
    }

    with torch.autocast(device_type='cuda'):
        ret = model(inputs)
    
    inputs = {
        'video_feats': torch.randn(1, 40, 512, device=device, dtype=torch.float32),
        'v_mask': torch.ones(1, 40, device=device, dtype=torch.long),
        
        'global_clip_embedding': torch.randn(1, 40, 512, device=device),

    }
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            
            # === 【计时开始】 ===
            if torch.cuda.is_available():
                torch.cuda.synchronize() # 等待之前的数据传输等操作完成
            start_time = time.time()
            
            # V2A 逻辑: 自动根据视频长度推导音频长度，无需 ref_lat
            # 这一步会跑完整个音频生成过程才返回
            generated_lat, _ = model.inference(inputs, use_tqdm=True)
            
            # === 【计时结束】 ===
            if torch.cuda.is_available():
                torch.cuda.synchronize() # 确保模型计算彻底完成
            end_time = time.time()
            
            latency = end_time - start_time
            print(f"\n⏱️ 非流式总推理时间 (Total Latency): {latency:.4f} 秒\n")

    ### streaming
    print("开始流式生成...")
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            
            # 1. 创建生成器
            generator = model.inference_stream(inputs, timesteps=12)
            all_audio_patches = []
            
            # === 【计时开始】 ===
            # 先同步 CUDA，确保之前的 GPU 任务（如数据搬运）都已完成，计时才准确
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            # 2. 开始迭代
            for i, x_patch in enumerate(generator):
                
                # === 【计时结束】仅在收到第 0 帧时触发 ===
                if i == 0:
                    # 再次同步，确保第 0 帧的计算真正完成
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()
                    ttft = end_time - start_time
                    print(f"\n⚡️ 首帧返回时间 (TTFT): {ttft:.4f} 秒")
                    timing = model._stream_first_patch_timing
                    print(f"   ├─ LM及之前: {timing['lm_and_before']:.4f} 秒")
                    print(f"   └─ LM之后(DiT): {timing['dit_after_lm']:.4f} 秒\n")

                print(f"收到第 {i} 个 Patch，形状: {x_patch.shape}")
                
                # 实时处理逻辑...
                all_audio_patches.append(x_patch)

    # 循环结束后再合并
    full_audio = torch.cat(all_audio_patches, dim=1)
    print("生成结束。")