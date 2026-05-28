import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torchdiffeq

from modules.tts.llama_dit.llama_prompt import LLaMa
from utils.nn.seq_utils import sequence_mask, remove_prefix, remove_suffix, add_prefix_nd

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

    # trainging
    do_checkpoint: bool = False
    use_qk_norm: bool = False

    cfg_mask_text_token: int = None
    text_fill_token: int = None
    use_caption_pool_in_adaln: bool = False
    use_caption_text_mark: bool = False
    use_spk_mask: bool = False 

    use_gated_attention: bool = False
    use_dynamic_cross_gate: bool = False

    use_moe_ffn: bool = False
    moe_p: float = 0.7                # Top-P routing threshold
    moe_num_routed: int = 8           # routed experts count
    moe_num_shared: int = 2           # shared experts count (always-on)
    moe_num_null: int = 4             # null experts count (no compute)
    moe_aux_loss_weight: float = 0.01 # training-time weight (you can anneal in loop)
    moe_use_gumbel: bool = False
    moe_gumbel_tau_start: float = 1.0       # 初始温度（训练早期）
    moe_gumbel_tau_end: float = 0.3         # 最终温度（训练后期）
    moe_gumbel_tau_anneal_steps: int = 200_000  # 退火步数
    moe_expert_dropout: float = 0.0   # 每个 step 随机屏蔽部分专家（0~1）

class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp

        self.encoder = LLaMa(hp)
        self.prenet = nn.Linear(self.hp.encoder_dim * 2 , self.hp.encoder_dim)

        self.lat_proj = nn.Linear(self.hp.in_channels, self.hp.encoder_dim)
        self.ctx_proj = nn.Linear(self.hp.in_channels, self.hp.encoder_dim)
        self.ctx_mask_proj = nn.Linear(1, self.hp.encoder_dim)
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels)
        self.caption_proj = nn.Linear(self.hp.caption_dim, self.hp.encoder_dim)
        if hp.use_caption_text_mark:
            self.caption_text_mark_embed = nn.Embedding(3, self.hp.encoder_dim)

        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)

        from modules.asr.llama.llama import LLaMa as LLaMaSmall, ModelArgs as ModelArgsSmall

        self.text_embedder = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        self.text_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=8, n_heads=16,
            use_causal_attn=False, 
        ))

        if hp.use_spk_mask:
            self.spk_mask_embedder = nn.Embedding(5, hp.encoder_dim)  # 0=none, 1..4=S1..S4

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

    def forward_text_encoder(self, inputs, x_mask):
        tgt_len = x_mask.shape[1]
        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        bsz = txt_tokens.shape[0]

        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        # ===== 1) build filled text ids: [B, tgt_len] =====
        x_txt_ids = torch.full((bsz, tgt_len), self.hp.text_fill_token, device=txt_tokens.device, dtype=txt_tokens.dtype)
        fill_pos = sequence_mask(txt_mask.long().sum(1), tgt_len)  # [B, tgt_len]
        x_txt_ids[fill_pos] = txt_tokens[txt_mask]

        token_emb = self.text_embedder(x_txt_ids.long())  # [B, tgt_len, C]

        # ===== 2) add spk_mask embedding if enabled =====
        if self.hp.use_spk_mask and ('spk_mask' in inputs) and (inputs['spk_mask'] is not None):
            spk_mask = inputs['spk_mask']  # expected shape [B, T_txt] with same txt_mask
            # 对齐到和 x_txt_ids 一样的“前缀填充”布局
            spk_ids = torch.zeros((bsz, tgt_len), device=txt_tokens.device, dtype=torch.long)
            spk_ids[fill_pos] = spk_mask[txt_mask].long().clamp_(0, 4)
            token_emb = token_emb + self.spk_mask_embedder(spk_ids)  # [B, tgt_len, C]

        # ===== 3) run text encoder =====
        x_txt = self.text_encoder(token_emb, x_mask)  # [B, tgt_len, C]
        return x_txt

    def forward(self, inputs, sigmas=None, x_noisy=None):
        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask
        x = inputs['lat']
        x_mask = sequence_mask(inputs['lat_lens'], maxlen=x.shape[1])

        # CFM: x is x1
        x0 = torch.randn_like(x)
        t = self.flow_matcher.time_sampler.sample([x0.shape[0]], x0.device).type_as(x0)
        xt = t[:, None, None] * x + (1 - t[:, None, None]) * x0
        ut = x - x0

        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_emb = self.f5_time_embed(t)
        x_noisy = (xt * (1 - ctx_mask)).bfloat16()
        target = ut

        x_txt = self.forward_text_encoder(inputs, x_mask)

        if 'caption_emb' in inputs and inputs['caption_emb'] is not None:
            caption_embs = self.caption_proj(inputs['caption_emb'])
            if self.hp.use_caption_text_mark:
                caption_text_mark_embed = self.caption_text_mark_embed(inputs['caption_text_mark'].long())
                caption_embs = caption_embs + caption_text_mark_embed
        else:
            caption_embs = None

        x_noisy = self.lat_proj(x_noisy) + self.ctx_proj(ctx_feature) + self.ctx_mask_proj(ctx_mask)
        x_noisy = self.prenet(torch.cat([x_noisy, x_txt], dim=-1))

        use_moe = bool(getattr(self.hp, "use_moe_ffn", False))

        if use_moe:
            encoder_out, moe_aux = self.encoder(
                x_noisy, t_emb, attn_mask=x_mask,
                do_checkpoint=self.hp.do_checkpoint,
                context=caption_embs,
                context_lens=inputs['caption_lens'],
            )
        else:
            encoder_out = self.encoder(
                x_noisy, t_emb, attn_mask=x_mask,
                do_checkpoint=self.hp.do_checkpoint,
                context=caption_embs,
                context_lens=inputs['caption_lens'],
            )
            moe_aux = None

        pred = self.postnet(encoder_out)

        if use_moe:
            return pred, target, moe_aux
        return pred, target

    def _forward(self, x, cond, timesteps, seq_cfg_w=[1.5, 3.0], timestep_annealing_w=(0.6, 0.6, 1.0)):
        """When we use torchdiffeq, we need to include the CFG process inside _forward()."""
        ctx = cond['ctx']
        ctx_mask = cond['ctx_mask']
        attn_mask = cond['attn_mask']
        x_txt = cond['x_txt']

        if 'caption_emb' in cond and cond['caption_emb'] is not None:
            caption_embs = self.caption_proj(cond['caption_emb'])
            if self.hp.use_caption_text_mark:
                caption_text_mark_embed = self.caption_text_mark_embed(cond['caption_text_mark'].long())
                caption_embs = caption_embs + caption_text_mark_embed
        else:
            caption_embs = None

        x = x * (1 - ctx_mask)
        x = self.lat_proj(x) + self.ctx_proj(ctx) + self.ctx_mask_proj(ctx_mask)
        x = self.prenet(torch.cat([x, x_txt], dim=-1))

        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_emb = self.f5_time_embed(timesteps)

        use_moe = bool(getattr(self.hp, "use_moe_ffn", False))
        pred_v = self.encoder(
            x, t_emb, attn_mask=attn_mask,
            context=caption_embs,
            context_lens=cond['caption_lens'],
        )
        if use_moe:
            pred_v = pred_v[0]

        pred = self.postnet(pred_v)

        if isinstance(timesteps, torch.Tensor) and timesteps.ndim > 0 and timesteps.shape[0] > pred.shape[0] // 3:
            timesteps, _, _ = timesteps.chunk(3)
            if timesteps.ndim == 1:
                timesteps = timesteps[:, None, None]
        a, b, p = timestep_annealing_w
        gamma_t = a + b * torch.pow(1 - timesteps, p)
        seq_cfg_w = [gamma_t * w for w in seq_cfg_w]

        cond_all, cond_txt, uncond = pred.chunk(3)

        pred = (
            uncond + 
            seq_cfg_w[0] * (cond_txt - uncond) + 
            seq_cfg_w[1] * (cond_all - cond_txt)
        )

        return pred

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, seq_cfg_w=[1.5, 3.0], timestep_annealing_w=(0.6, 0.6, 1.0), return_timesteps=False, **kwargs):
        tgt_len = inputs['tgt_len']     # reference + target
        x_mask = sequence_mask(tgt_len)
        
        x_txt = self.forward_text_encoder(inputs, x_mask)

        (bsz, tgt_len, _), device = x_txt.shape, x_txt.device
        bsz = bsz // 3

        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        cond = {
            'ctx': ctx_feature,
            'ctx_mask': ctx_mask,
            'attn_mask': x_mask,
            'x_txt': x_txt,
            'txt_lens': inputs['txt_lens'],
            'caption_emb':inputs['caption_emb'] if 'caption_emb' in inputs else None,
            'caption_lens': inputs['caption_lens'] if 'caption_lens' in inputs else None,
            'caption_text_mark': inputs.get('caption_text_mark', None),
        }

        ''' Euler ODE solver '''
        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1).to(device)
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

        traj = torchdiffeq.odeint(
            lambda t, x: self._forward(
                torch.cat([x] * 3), cond, timesteps=t.unsqueeze(0), seq_cfg_w=seq_cfg_w),
            torch.randn([bsz, tgt_len, self.hp.out_channels], device=device),
            t_schedule,
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        x = traj[-1]

        if return_timesteps:
            return x, t_schedule
        
        return x
