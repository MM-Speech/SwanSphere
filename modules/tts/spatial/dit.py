import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# import attrdict
import torchdiffeq

from utils.commons.hparams import hparams
from modules.tts.ar_dur.commons.align_ops import expand_states
from modules.tts.llama_dit.llama_avgen import LLaMa
from utils.nn.seq_utils import sequence_mask

logger = logging.getLogger(__name__)


class AmbisonicsEncode(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.direction_embedding = nn.Sequential(
            nn.Linear(3, hparams.d_model),
            nn.GELU(),
            nn.Linear(hparams.d_model, hparams.d_model),
        )
        self.energy_map_projection = nn.Sequential(
            nn.Linear(49, hparams.d_model),
            nn.GELU(),
            nn.Linear(hparams.d_model, hparams.d_model),
        )
        
        embed_dim = hparams.d_model
        
        self.clip_projection = nn.Linear(hparams.d_clip, embed_dim)

        self.embed_positions = nn.Embedding(
            hparams.max_source_length,
            embed_dim,
        )
        # self.embed_spatial = nn.Embedding(4, embed_dim)
        
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.gradient_checkpointing = False
    
    def forward(
        self,
        direction: Optional[torch.FloatTensor] = None,
        energy_map: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        # retrieve inputs_embeds
        if inputs_embeds is None:
            raise ValueError("You have to specify inputs_embeds")
        # print(f'{inputs_embeds.shape =}')[17, 20, 512]
        bsz, seq, dim = inputs_embeds.shape
        
        # Prepare position ids
        position_ids = torch.arange(seq, device=inputs_embeds.device)
        # Prepare input embeddings
        inputs_embeds = self.clip_projection(inputs_embeds)
        if energy_map is not None:
            inputs_embeds = inputs_embeds + self.energy_map_projection(energy_map)
        
        embed_pos = self.embed_positions(position_ids)
        inputs_embeds = inputs_embeds + embed_pos
        
        if direction is not None:
            inputs_embeds = inputs_embeds + self.direction_embedding(
                direction
            ).unsqueeze(1)
        return inputs_embeds

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

    d_model: int = 1024
    d_clip: int = 512
    max_source_length: int = 20
    dac_frame_rate: int = 86

    # diffusion
    cfg_mask_text_token: int = None

    # trainging
    do_checkpoint: bool = False
    use_qk_norm: bool = True
    

class Diffusion(nn.Module):
    '''
    最基本的dit based video to spatial audio的模型
    在inference时，inputs包括inputs_embeds（也就是clip emb），direction，energy_map；
    输出是dac 的latents
    
    '''
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp
        
        ### for spatial ###
        self.ambisonics_encoder = AmbisonicsEncode(hp)
        self.lat_proj = nn.Linear(self.hp.d_model*4 + self.hp.d_model, self.hp.d_model)
        self.postnet = nn.Linear(self.hp.d_model, self.hp.d_model*4)
        self.global_cond = nn.Linear(self.hp.d_clip, self.hp.d_model)
        ###################

        self.encoder = LLaMa(hp)
        # self.add_vad_mask = hparams.get('add_vad_mask', False)
        # if self.add_vad_mask:
        #     print('| use vad mask!')
        #     self.prenet = nn.Linear(self.hp.in_channels + 2, self.hp.encoder_dim)
        # else:
        #     self.prenet = nn.Linear(self.hp.in_channels + 1, self.hp.encoder_dim)

        # self.lat_proj = nn.Linear(self.hp.encoder_dim * 2, self.hp.encoder_dim)
        # self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels)
        # self.caption_proj = nn.Linear(self.hp.caption_dim, self.hp.encoder_dim)
        # if not hparams.get('drop_st', False):
        #     self.audio_token_proj = nn.Conv1d(hp.encoder_dim, hp.encoder_dim, kernel_size=3, padding='same')
        #     if hp.audio_tokenizer == 'glm4v':
        #         self.audio_token_embed = nn.Embedding(hp.audio_vocab_size, hp.encoder_dim)
        #         self.audio_token_upsampler = nn.Upsample(scale_factor=2, mode='nearest')

        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.d_model)

        # from modules.flow_matching.llama import LLaMa as LLaMaSmall,  ModelArgs as ModelArgsSmall
        # self.text_embedder = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        # self.text_encoder = LLaMaSmall(ModelArgsSmall(
        #     dim=hp.encoder_dim,
        #     n_layers=6, n_heads=8
        # ))

        # # init all weights
        self._init_weights()
        
        print("---args check---")
        print(f"{self.hp.use_qk_norm =}")
        print("---args check---")

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
        ### for spatial
        inputs_embeds = inputs['inputs_embeds']
        global_clip_embedding = inputs['global_clip_embedding']
        direction = inputs['direction']
        energy_map = inputs['energy_map']
        
        x = inputs['lat']
        
        cond_emb = self.ambisonics_encoder(
            direction=direction,
            energy_map=energy_map,
            inputs_embeds=inputs_embeds
        ) # [b, 20, 1024]
        
        B, T, _ = x.shape
        T1, T2 = cond_emb.shape[1], x.shape[1]
        device = x.device
        repeat_factor = torch.arange(T2).float()
        idx = (repeat_factor * T1 / T2).floor().long().clamp(max=T1 - 1) 
        cond_emb = cond_emb[:, idx, :]
        
        # perform CFM
        # x is x1 in CFM
        x0 = torch.randn_like(x)
        t = self.flow_matcher.time_sampler.sample([B], x0.device).type_as(x0)
        xt = t[:, None, None] * x + (1 - t[:, None, None]) * x0
        ut = x - x0

        t = t.bfloat16()
        target = ut

        x_in = self.lat_proj(torch.cat(
            [xt, cond_emb], -1
        ))
        attn_mask = torch.ones_like(x[:, :, 0]).to(device)
        
        ### for global_clip_embedding
        global_embed = self.global_cond(global_clip_embedding)
        if len(global_clip_embedding.shape) == 3:
            global_embed = torch.mean(global_clip_embedding, dim=1).values
        if global_embed is None:
            global_cond = self.f5_time_embed(t)
        else:
            global_cond = self.f5_time_embed(t) + global_embed 
        
        
        encoder_out = self.encoder(x_in, global_cond, attn_mask=attn_mask)
        pred = self.postnet(encoder_out)

        return pred, target
        
        
        

    def _forward(self, x, cond, timesteps, seq_cfg_w=[1.0, 1.5, 1.5, 1.5]):
        ### for spatial
        inputs_embeds = cond['inputs_embeds']
        direction = cond['direction']
        energy_map = cond['energy_map']
        
        cond_emb = self.ambisonics_encoder(
            direction=direction,
            energy_map=energy_map,
            inputs_embeds=inputs_embeds
        ) # [b, 20, 1024]
        
        T1, T2 = cond_emb.shape[1], x.shape[1]
        B, T, _ = x.shape
        device = x.device
        repeat_factor = torch.arange(T2).float()
        idx = (repeat_factor * T1 / T2).floor().long().clamp(max=T1 - 1) 
        cond_emb = cond_emb[:, idx, :]
        
        x = self.lat_proj(torch.cat(
            [x, cond_emb], -1
        ))
        
        attn_mask = torch.ones_like(x[:, :, 0]).to(device)
        
        pred_v = self.encoder(x, self.f5_time_embed(timesteps), attn_mask=attn_mask)
        pred = self.postnet(pred_v)
        
        ### CFG
        # To be implemented
        ###
        
        return pred
        

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, seq_cfg_w=[1.0, 1.5, 1.5], **kwargs):
        
        ### for spatial ###
        inputs_embeds = inputs['inputs_embeds']
        direction = inputs['direction']
        energy_map = inputs['energy_map']

        device = inputs_embeds.device
        bsz, seq, dim = inputs_embeds.shape
        tgt_len = int(seq / 4 * self.hp.dac_frame_rate)
        
        cond = {
            'inputs_embeds': inputs_embeds,
            'direction': direction,
            'energy_map': energy_map,
        }
        ''' Euler ODE solver '''
        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1).to(device)
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)


        traj = torchdiffeq.odeint(
            lambda t, x: self._forward(
                torch.cat([x] * bsz), cond, timesteps=t.unsqueeze(0), seq_cfg_w=seq_cfg_w),
            torch.randn([1, tgt_len, self.hp.d_model], device=device),
            t_schedule,
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        x = traj[-1]
        return x

        ###################
        
        
        # if not hparams.get('drop_st', False):
        #     audio_tokens = self.audio_token_embed(inputs['semantic_tokens'])
        #     audio_tokens = self.audio_token_upsampler(audio_tokens.transpose(1, 2)).transpose(1, 2) # [B, T, C]
        #     audio_tokens = self.audio_token_proj(audio_tokens.transpose(1, 2)).transpose(1, 2)  # [B, T, C]
        # else:
        #     audio_tokens = torch.zeros(inputs['lat_ctx'].shape[0],
        #                                inputs['lat_ctx'].shape[1],
        #                                self.hp.encoder_dim).to(inputs['lat_ctx'])
        # x_mask = torch.ones_like(audio_tokens[:, :, 0])

        # (bsz, tgt_len, _), device = audio_tokens.shape, audio_tokens.device

        # ctx_mask = inputs['ctx_mask']
        # ctx_feature = inputs['lat_ctx'] * ctx_mask

        # txt_tokens = inputs["txt_tokens"]
        # txt_mask = inputs["txt_mask"]
        # x_txt = torch.full((bsz, tgt_len), self.hp.cfg_mask_text_token).to(txt_tokens)
        # x_txt[sequence_mask(txt_mask.sum(1), tgt_len)] = txt_tokens[txt_mask]
        # x_txt = self.text_encoder(x=self.text_embedder(x_txt.long()), attn_mask=x_mask)   # [B, T, C]

        # cond = {
        #     'ctx': ctx_feature,
        #     'ctx_mask': ctx_mask,
        #     'attn_mask': x_mask,
        #     'x_txt': x_txt,
        #     'audio_tokens': audio_tokens,
        #     'caption_emb':inputs['caption_emb'] if 'caption_emb' in inputs else None,
        #     'caption_lens': inputs['caption_lens'] if 'caption_lens' in inputs else None,
        #     'vad_mask': inputs['vad_mask']
        # }

        # ''' Euler ODE solver '''
        # sway_sampling_coef = -1.0
        # t_schedule = torch.linspace(0, 1, timesteps + 1).to(device)
        # if sway_sampling_coef is not None:
        #     t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

        # traj = torchdiffeq.odeint(
        #     lambda t, x: self._forward(
        #         torch.cat([x] * bsz), cond, timesteps=t.unsqueeze(0), seq_cfg_w=seq_cfg_w),
        #     torch.randn([1, tgt_len, self.hp.out_channels], device=device),
        #     t_schedule,
        #     atol=1e-4,
        #     rtol=1e-4,
        #     method="euler",
        # )
        # x = traj[-1]
        # return x

def print_next_level_params(model: nn.Module):
    for name, submodule in model.named_children():  # 只遍历下一层子模块
        num_params = sum(p.numel() for p in submodule.parameters()
                         if p.requires_grad)       # 只算可训练参数
        num_params /= 1024 * 1024
        print(f"{name}: {num_params:,} MB")
        
if __name__ == '__main__':
    
    # from utils.commons.hparams import hparams, set_hparams
    # set_hparams()
    
    # args = ModelArgs(
    #     d_model = hparams['d_model'],
    #     dac_frame_rate = hparams['dac_frame_rate']
    # )
    
    # device = torch.device('cuda')
    # model = Diffusion(args).to(device)
    
    # params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"{params / 1024 /1024} MB")
    # print_next_level_params(model)
    
    # inputs = {}
    # inputs['inputs_embeds'] = torch.randn([1, 20, 512], device=device)
    # inputs['direction'] = torch.randn([1, 3], device=device)
    # inputs['lat'] = torch.randn([1, 431, 4096], device=device)
    # inputs['energy_map'] = torch.randn([1, 20, 49], device=device)
    
    # # print(cond_emb.shape) # [b, 20, 1024]
    
    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #     ret = model.inference(inputs)
    # print(f'{ret.shape =}')
    
    from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
    flow_matcher = ConditionalFlowMatcher(sigma=0.0)
    from modules.tts.f5_dit.f5_modules import TimestepEmbedding
    f5_time_embed = TimestepEmbedding(1024)
    
    B = 2
    x = torch.randn([2, 20, 1024])
    x0 = torch.randn_like(x)
    t = flow_matcher.time_sampler.sample([B], x0.device).type_as(x0)
    
    print(t, t.shape) # [2]
    print(f"{f5_time_embed(t).shape =}") # [2, 1024]