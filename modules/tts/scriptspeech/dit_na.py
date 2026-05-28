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
        if not hparams.get('drop_st', False):
            self.audio_token_proj = nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=3, padding='same')
            if hp.audio_tokenizer == 'glm4v':
                self.audio_token_embed = nn.Embedding(hp.audio_vocab_size, hp.encoder_dim)
                self.audio_token_upsampler = nn.Upsample(scale_factor=2, mode='nearest')

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

    @torch.no_grad()
    def text_encode(
        self, texts: List[str], special_tokens = None
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]:
        # Text encoder forward.
        text_outputs = self.text_encoder(texts, special_tokens)
        # Convert to nadit input format.
        if isinstance(text_outputs.embeddings, list):
            text_embeds = [e[m] for e, m in zip(text_outputs.embeddings, text_outputs.masks)]
            text_shapes = [m.sum(-1).unsqueeze(-1) for m in text_outputs.masks]
        else:
            text_embeds = text_outputs.embeddings[text_outputs.masks]
            text_shapes = text_outputs.masks.sum(-1).unsqueeze(-1)
        # Return flattened embeddings and shapes.
        return text_embeds, text_shapes, text_outputs.input_token_ids


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
        t = t.bfloat16()
        x_noisy = (xt * (1 - ctx_mask)).bfloat16() + ctx_feature # prefix ref wav
        target = ut

        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        x_txt = torch.full((bsz, x.shape[1]), self.hp.cfg_mask_text_token).to(txt_tokens)
        x_txt[sequence_mask(txt_mask.sum(1), x.shape[1])] = txt_tokens[txt_mask]
        x_txt = self.text_encoder(x=self.text_embedder(x_txt.long()), attn_mask=x_mask)   # [B, T, C]

        if not hparams.get('drop_st', False):
            audio_tokens = self.audio_token_embed(inputs['semantic_tokens'])  # 9, 540
            audio_tokens = self.audio_token_upsampler(audio_tokens.transpose(1, 2)).transpose(1, 2) # [B, T, C]
            audio_tokens = self.audio_token_proj(audio_tokens.transpose(1, 2)).transpose(1, 2) # 9, 1080, 1536
        else:
            audio_tokens = torch.zeros_like(x_txt)
        if 'caption_emb' in inputs and inputs['caption_emb'] is not None:
            caption_embs = self.caption_proj(inputs['caption_emb'])
        else:
            caption_embs = None
        if self.add_vad_mask:
            x_noisy = self.lat_proj(
                torch.cat([self.prenet(torch.cat([x_noisy, ctx_mask, inputs['vad_mask']], -1)), x_txt], -1)) + audio_tokens
        else:
            x_noisy = self.lat_proj(torch.cat([self.prenet(torch.cat([x_noisy, ctx_mask], -1)), x_txt], -1)) + audio_tokens
        encoder_out = self.encoder(x_noisy, self.f5_time_embed(t), attn_mask=x_mask,
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
        audio_tokens = cond['audio_tokens']
        x = x * (1 - ctx_mask) + ctx
        if self.add_vad_mask:
            x = self.lat_proj(
                torch.cat([self.prenet(torch.cat([x, ctx_mask, cond['vad_mask']], -1)), x_txt],
                          -1)) + audio_tokens
        else:
            x = self.lat_proj(torch.cat([self.prenet(torch.cat([x, ctx_mask], -1)), x_txt], -1)) + audio_tokens
        if 'caption_emb' in cond and cond['caption_emb'] is not None:
            caption_embs = self.caption_proj(cond['caption_emb'])
        else:
            caption_embs = None
        pred_v = self.encoder(x, self.f5_time_embed(timesteps), attn_mask=attn_mask, context=caption_embs, context_lens=cond['caption_lens'])
        pred = self.postnet(pred_v)

        """ Perform CFG """
        ## old MegaTTS3 CFG:
        # uncond + w0 * cond_txt - w0 * uncond + w1 * cond_spk_txt - w1 * cond_txt
        # (1 - w0) * uncond + (w0 - w1) * cond_txt + w1 * cond_spk_txt
        
        ## new CFG:
        # uncond + s1 * (cond_txt - uncond) + s2 * (cond_spk - uncond) + s3 * (cond_spk_txt - uncond)
        # (1 - s1 - s2 - s3) * uncond + s1 * cond_txt + s2 * cond_spk + s3 * cond_spk_txt

        ## equation:
        # 1 - w0 = 1 - s1 - s2 - s3
        # w0 - w1 = s1
        # s2 = 0
        # w1 = s3
        ## answer:
        # w1 = s3
        # w0 = s1 + s3

        # cond_all, cond_cap, cond_ref, uncond = pred.chunk(4)
        #
        # pred = (
        #     uncond +
        #     seq_cfg_w[0] * (cond_all - cond_cap) + # txt
        #     seq_cfg_w[1] * (cond_cap - cond_ref) + # cap
        #     seq_cfg_w[2] * (cond_ref - uncond) # ref wav
        # )

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
        if not hparams.get('drop_st', False):
            audio_tokens = self.audio_token_embed(inputs['semantic_tokens'])
            audio_tokens = self.audio_token_upsampler(audio_tokens.transpose(1, 2)).transpose(1, 2) # [B, T, C]
            audio_tokens = self.audio_token_proj(audio_tokens.transpose(1, 2)).transpose(1, 2)  # [B, T, C]
        else:
            audio_tokens = torch.zeros(inputs['lat_ctx'].shape[0],
                                       inputs['lat_ctx'].shape[1],
                                       self.hp.encoder_dim).to(inputs['lat_ctx'])
        x_mask = torch.ones_like(audio_tokens[:, :, 0])

        (bsz, tgt_len, _), device = audio_tokens.shape, audio_tokens.device

        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        x_txt = torch.full((bsz, tgt_len), self.hp.cfg_mask_text_token).to(txt_tokens)
        x_txt[sequence_mask(txt_mask.sum(1), tgt_len)] = txt_tokens[txt_mask]
        x_txt = self.text_encoder(x=self.text_embedder(x_txt.long()), attn_mask=x_mask)   # [B, T, C]

        cond = {
            'ctx': ctx_feature,
            'ctx_mask': ctx_mask,
            'attn_mask': x_mask,
            'x_txt': x_txt,
            'audio_tokens': audio_tokens,
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
