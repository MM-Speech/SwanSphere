import logging
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torchdiffeq
from torch import Tensor, nn
from modules.flow_matching.dit1d import TimeEmbedding, UniformDistribution
from modules.flow_matching.logitnormal import LogitNormalTrainingTimesteps
from modules.flow_matching.vp_cfm import ConditionalFlowMatcher
from modules.flow_matching.llama import LLaMa
from utils.commons.hparams import hparams
import numpy as np

logger = logging.getLogger(__name__)


class Diffusion(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.min_t = 0
        self.max_t = 1
        self.target_type = hp.target_type
        self.act_fn = nn.GELU()
        self.bias = False
        # time-embedding
        self.time_embedding = TimeEmbedding(hp.time_embed_dim, bias=self.bias)
        if hparams.get("ori_ref", False):
            patch_size = hparams['patch_size']
            img_proj_modules = []
            img_proj_modules.append(nn.Conv2d(
                3, hp.patch_conv_channels, kernel_size=2, stride=2, bias=False
            ))
            for i in range(int(np.log2(patch_size)) - 2):
                img_proj_modules.append(nn.Conv2d(
                    hp.patch_conv_channels, hp.patch_conv_channels, kernel_size=2,
                    stride=2, bias=False
                ))
                img_proj_modules.append(nn.ReLU())
            img_proj_modules.append(nn.Conv2d(
                hp.patch_conv_channels, hp.encoder_dim, kernel_size=2, stride=2, bias=False
            ))
            self.img_proj = nn.Sequential(*img_proj_modules)
        if not hasattr(hp, "window_size"):
            hp.window_size = [-1, -1]

        # backbone
        @dataclass
        class LLaMaArgs:
            dim: int = 1024
            n_layers: int = 24
            n_heads: int = 16
            n_kv_heads: Optional[int] = None
            multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
            norm_eps: float = 1e-5
            max_seq_len: int = 16384
            dropout: float = 0.0
            ffn_dim_multiplier: Optional[float] = None
            use_qk_norm = False

        llama_args = LLaMaArgs()
        llama_args.dim = hp.encoder_dim
        llama_args.n_layers = hp.encoder_n_layers
        llama_args.n_heads = hp.encoder_n_heads
        llama_args.use_qk_norm = hp.use_qk_norm
        self.encoder = LLaMa(llama_args)
        if hparams.get('use_audio_cond', False):
            self.prenet = nn.Linear(
                hp.time_embed_dim + hp.local_cond_dim + hp.in_channels + hp.audio_cond_dim,
                hp.encoder_dim, bias=self.bias
            )
        else:
            self.prenet = nn.Linear(
                hp.time_embed_dim + hp.local_cond_dim + hp.in_channels,
                hp.encoder_dim, bias=self.bias
            )
        if hparams.get('use_text_cond', False):
            self.text_prenet = nn.Linear(hp.text_dim, hp.encoder_dim, bias=self.bias)
        if hparams.get('use_camera_cond', False):
            self.camera_emb = nn.Embedding(hp.camera_num, embedding_dim=hp.encoder_dim)
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels, bias=False)
        if hparams.get("use_motion_region", False):
            self.motion_region_prenet = nn.Linear(4, hp.encoder_dim, bias=self.bias)
        self.sigma_distribution = UniformDistribution(vmin=self.min_t, vmax=self.max_t)
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        self.timestep_sampler = LogitNormalTrainingTimesteps()

    def get_alpha_beta(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        angle = t * math.pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def get_t_from_sigma(self, sigma):
        return torch.arctan(sigma) / math.pi * 2

    def forward(self, inputs, sigmas=None, x_noisy=None, debug=False):
        cond = {"lat_lens": inputs['lat_lens'], 'text_embed': inputs.get('text_embed'), 'text_mask': inputs.get('text_mask')}
        if hparams.get("use_audio_cond", False) and "audio_fea" in inputs.keys():
            cond['audio_fea'] = inputs['audio_fea']
        # ref_kp_frames = inputs['ref_kp_frames']
        # ref_kp_encoders = self.kp_encoder(ref_kp_frames)
        self.get_ref_embeds(inputs, cond) # ['lat_lens', 'text_embed', 'audio_fea']
        lat_lens = inputs["lat_lens"]
        local_cond = inputs['lat_ctx']
        if inputs.get("body_kp_ctx") is not None:
            local_cond = torch.cat([local_cond, inputs.get("body_kp_ctx")], -1)
        if inputs.get("bg_kp_ctx") is not None:
            local_cond = torch.cat([local_cond, inputs.get("bg_kp_ctx")], -1)
        if inputs.get("domain_tag") is not None:
            local_cond = torch.cat([local_cond, inputs.get("domain_tag")], -1)
        if hparams.get("use_res_embed", False):
            local_cond = torch.cat([local_cond, inputs['res_region'][:, None].repeat(1, local_cond.shape[1], 1)], -1)
        ctx_mask = inputs['ctx_mask']
        x = inputs['lat']
        if inputs.get('camera_ids') is not None:
            cond['camera_embedding'] = self.camera_emb(inputs['camera_ids'])
            cond['camera_mask'] = inputs.get('camera_mask')
        x0 = torch.randn_like(x)
        if hparams.get('use_logitnormal_time'):
            t = self.timestep_sampler.sample([x0.shape[0]], x0.device)
            t_float, x_noisy, target = self.flow_matcher.sample_location_and_conditional_flow(x0, x, t=t)
        else:
            t_float, x_noisy, target = self.flow_matcher.sample_location_and_conditional_flow(x0, x)
        pred = self._forward(
            x_noisy, local_cond, cond, ctx_mask, t_float,
            lat_lens=lat_lens)

        ret_dict = {
            "pred_v": pred,
            "target_v": target,
        }
        return ret_dict

    def sequence_mask(self, seq_lens, max_len=None, device='cpu'):
        if max_len is None:
            max_len = seq_lens.max()
        mask = torch.arange(max_len).unsqueeze(0).to(device)  # [1, t]
        mask = mask < (seq_lens.unsqueeze(1))  # [1, t] + [b, 1] = [b, t]
        mask = mask.float()
        return mask

    def get_ref_embeds(self, inputs, cond):
        if hparams.get("ori_ref", False):
            image_embs = self.img_proj(
                torch.stack(inputs['images'], 0).float() / 127.5 - 1
            ).flatten(2).transpose(1, 2)
            cond['image_embs'] = image_embs
        if hparams.get("use_motion_region", False):
            cond['motion_region_embs'] = self.motion_region_prenet(inputs['kp_region'])

    def _forward(self, x, local_cond, cond, ctx_mask, timesteps, lat_lens=None, use_torchdiffeq=False,
                 torchdiffeq_cfg=1.0):
        x = x * (1 - ctx_mask)
        time_emb = self.time_embedding(timesteps)
        time_emb = time_emb.unsqueeze(1).expand(local_cond.shape[0], local_cond.shape[1], -1)
        if "audio_fea" in cond.keys() and hparams.get("use_audio_cond", False):
            x = self.prenet(torch.cat([x, time_emb, local_cond, cond['audio_fea']], dim=-1))
        else:
            x = self.prenet(torch.cat([x, time_emb, local_cond], dim=-1))
        encoder_inp = x
        seq_len = lat_lens if lat_lens is not None else torch.LongTensor([x.shape[1]] * x.shape[0]).to(x.device)
        T_tgt = x.shape[1]
        prefix = []
        if "image_embs" in cond.keys():
            prefix = [cond['image_embs']] + prefix
            seq_len = seq_len + cond['image_embs'].shape[1]
        if cond['text_embed'] is not None:
            text_embed = self.text_prenet(cond['text_embed'])
            prefix = [text_embed] + prefix
            # seq_len = seq_len + text_embed.shape[1]
            assert cond['text_mask'] is not None
        if 'motion_region_embs' in cond.keys():
            prefix = [cond['motion_region_embs'][:, None]] + prefix
            seq_len = seq_len + 1
        if 'camera_embedding' in cond.keys():
            if cond['camera_embedding'].ndim == 2:
                prefix = [cond['camera_embedding'][:, None]] + prefix
            elif cond['camera_embedding'].ndim == 3:
                prefix = [cond['camera_embedding']] + prefix
            else:
                raise ValueError
        
        encoder_inp = torch.cat(prefix + [encoder_inp], 1)
        attn_mask = self.sequence_mask(seq_len, device=x.device) > 0
        if cond['text_mask'] is not None:
            attn_mask = torch.cat([cond['text_mask'], attn_mask], dim=1)
        if "camera_mask" in cond.keys():
            attn_mask = torch.cat([cond['camera_mask'], attn_mask], dim=1)
        pred_v = self.encoder(encoder_inp, attn_mask=attn_mask)
        pred_v = pred_v[:, -T_tgt:]
        pred = self.postnet(pred_v)
        if use_torchdiffeq:
            pred = torchdiffeq_cfg * pred[0:1] + (1 - torchdiffeq_cfg) * pred[1:2]
        return pred

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, sampler="ddim", cfg_w=1.0, **kwargs):
        cond = {"lat_lens": inputs['lat_lens'], 'text_embed': inputs.get('text_embed'), 'text_mask': inputs.get('text_mask')}
        if hparams.get("use_audio_cond", False) and "audio_fea" in inputs.keys():
            cond['audio_fea'] = inputs['audio_fea']
        ctx_feature = "lat_ctx"
        local_cond = inputs[ctx_feature]
        if inputs.get("body_kp_ctx") is not None:
            local_cond = torch.cat([local_cond, inputs.get("body_kp_ctx")], -1)
        if inputs.get("bg_kp_ctx") is not None:
            local_cond = torch.cat([local_cond, inputs.get("bg_kp_ctx")], -1)
        if inputs.get("domain_tag") is not None:
            local_cond = torch.cat([local_cond, inputs.get("domain_tag")], -1)
        if hparams.get("use_res_embed", False):
            local_cond = torch.cat([local_cond, inputs['res_region'][:, None].repeat(1, local_cond.shape[1], 1)], -1)
        ctx_mask = inputs['ctx_mask']
        _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
        if cfg_w != 1:
            traj = torchdiffeq.odeint(
                lambda t, x: self._forward(
                    torch.cat([x] * 2), local_cond, cond, ctx_mask,
                    timesteps=t.unsqueeze(0).repeat(2),
                    use_torchdiffeq=True, torchdiffeq_cfg=cfg_w),
                torch.randn([1, frm_len, self.hp.out_channels], device=device,dtype=local_cond.dtype),
                torch.linspace(0, 1, timesteps + 1, device=device,dtype=local_cond.dtype),
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
        else:
            for k in cond.keys():
                if cond[k] is not None:
                    cond[k] = cond[k][:1]
            traj = torchdiffeq.odeint(
                lambda t, x: self._forward(x, local_cond[:1], cond, ctx_mask[:1],
                                           timesteps=t.unsqueeze(0)),
                torch.randn([1, frm_len, self.hp.out_channels], device=device,dtype=local_cond.dtype),
                torch.linspace(0, 1, timesteps + 1, device=device,dtype=local_cond.dtype),
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
        x = traj[-1]

        return x

class ConditionEmbedder(nn.Module):
    def __init__(self, hidden_size, context_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_size, bias=True),
            # nn.GELU(approximate='tanh'),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    def forward(self,x):
        return self.mlp(x)

class Diffusion_mlp(Diffusion):
    def __init__(self, hp):
        super(Diffusion, self).__init__()
        self.hp = hp

        self.min_t = 0
        self.max_t = 1
        self.target_type = hp.target_type
        self.act_fn = nn.GELU()
        self.bias = False
        # time-embedding
        self.time_embedding = TimeEmbedding(hp.time_embed_dim, bias=self.bias)
        # backbone
        @dataclass
        class LLaMaArgs:
            dim: int = 1024
            n_layers: int = 24
            n_heads: int = 16
            n_kv_heads: Optional[int] = None
            multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
            norm_eps: float = 1e-5
            max_seq_len: int = 16384
            dropout: float = 0.0
            ffn_dim_multiplier: Optional[float] = None
            use_qk_norm = False

        llama_args = LLaMaArgs()
        llama_args.dim = hp.encoder_dim
        llama_args.n_layers = hp.encoder_n_layers
        llama_args.n_heads = hp.encoder_n_heads
        llama_args.use_qk_norm = hp.use_qk_norm
        self.encoder = LLaMa(llama_args)
        self.audio_proj = ConditionEmbedder(hidden_size=hp.audio_hidden_dim,context_dim=hp.audio_cond_dim)
        self.body_proj = ConditionEmbedder(hidden_size=hp.body_kp_dim,context_dim=hp.body_kp_dim)
        self.prenet = nn.Linear(
            hp.time_embed_dim + hp.local_cond_dim + hp.in_channels + hp.audio_hidden_dim + hp.body_kp_dim,
            hp.encoder_dim, bias=self.bias
        )
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels, bias=False)
        self.sigma_distribution = UniformDistribution(vmin=self.min_t, vmax=self.max_t)
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        self.timestep_sampler = LogitNormalTrainingTimesteps()

    def forward(self, inputs, sigmas=None, x_noisy=None, debug=False):
        cond = {"lat_lens": inputs['lat_lens'], 'text_embed': inputs.get('text_embed'), 'text_mask': inputs.get('text_mask')}
        cond['audio_fea'] = inputs['audio_fea']
        cond['body_kp_ctx'] = inputs["body_kp_ctx"]
        lat_lens = inputs["lat_lens"]
        local_cond = inputs['lat_ctx']

        ctx_mask = inputs['ctx_mask']
        x = inputs['lat']
        x0 = torch.randn_like(x)
        if hparams.get('use_logitnormal_time'):
            t = self.timestep_sampler.sample([x0.shape[0]], x0.device)
            t_float, x_noisy, target = self.flow_matcher.sample_location_and_conditional_flow(x0, x, t=t)
        else:
            t_float, x_noisy, target = self.flow_matcher.sample_location_and_conditional_flow(x0, x)
        pred = self._forward(
            x_noisy, local_cond, cond, ctx_mask, t_float,
            lat_lens=lat_lens)

        ret_dict = {
            "pred_v": pred,
            "target_v": target,
        }
        return ret_dict

    def _forward(self, x, local_cond, cond, ctx_mask, timesteps, lat_lens=None, use_torchdiffeq=False,
                 torchdiffeq_cfg=1.0):
        x = x * (1 - ctx_mask)
        time_emb = self.time_embedding(timesteps)
        time_emb = time_emb.unsqueeze(1).expand(local_cond.shape[0], local_cond.shape[1], -1)
        audio_fea = self.audio_proj(cond['audio_fea'])
        body_fea = self.body_proj(cond["body_kp_ctx"])
        x = self.prenet(torch.cat([x, time_emb, local_cond,body_fea,audio_fea], dim=-1))
        encoder_inp = x
        seq_len = lat_lens if lat_lens is not None else torch.LongTensor([x.shape[1]] * x.shape[0]).to(x.device)
        T_tgt = x.shape[1]
        prefix = []
        
        encoder_inp = torch.cat(prefix + [encoder_inp], 1)
        attn_mask = self.sequence_mask(seq_len, device=x.device) > 0
        pred_v = self.encoder(encoder_inp, attn_mask=attn_mask)
        pred_v = pred_v[:, -T_tgt:]
        pred = self.postnet(pred_v)
        if use_torchdiffeq:
            pred = torchdiffeq_cfg * pred[:pred.shape[0]//2] + (1 - torchdiffeq_cfg) * pred[pred.shape[0]//2:]
        return pred

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, sampler="ddim", cfg_w=1.0, **kwargs):
        cond = {"lat_lens": inputs['lat_lens'], 'text_embed': inputs.get('text_embed'), 'text_mask': inputs.get('text_mask')}
        cond['audio_fea'] = inputs['audio_fea']
        cond['body_kp_ctx'] = inputs["body_kp_ctx"]
        ctx_feature = "lat_ctx"
        local_cond = inputs[ctx_feature]
        ctx_mask = inputs['ctx_mask']
        _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
        if cfg_w != 1:
            traj = torchdiffeq.odeint(
                lambda t, x: self._forward(
                    torch.cat([x] * 2), local_cond, cond, ctx_mask,
                    timesteps=t.unsqueeze(0).repeat(2),
                    use_torchdiffeq=True, torchdiffeq_cfg=cfg_w),
                torch.randn([1, frm_len, self.hp.out_channels], device=device,dtype=local_cond.dtype),
                torch.linspace(0, 1, timesteps + 1, device=device,dtype=local_cond.dtype),
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
        else:
            for k in cond.keys():
                if cond[k] is not None:
                    cond[k] = cond[k][:1]
            traj = torchdiffeq.odeint(
                lambda t, x: self._forward(x, local_cond, cond, ctx_mask,
                                           timesteps=t.unsqueeze(0)),
                torch.randn([1, frm_len, self.hp.out_channels], device=device,dtype=local_cond.dtype),
                torch.linspace(0, 1, timesteps + 1, device=device,dtype=local_cond.dtype),
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
        x = traj[-1]

        return x