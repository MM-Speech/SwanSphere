import logging
import math
from dataclasses import dataclass, field
from math import pi
from typing import Sequence, Tuple, Union, Optional

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np

# from samantha.criterion.masked_loss import sequence_mask
from modules.flow_matching.llama import LLaMa
# from modules.text2motion.voice_conversion.vc_modules import ConvGlobalStacks
from modules.commons.conv_stacks import ConvGlobalStacks
import torchdiffeq
from utils.commons.hparams import hparams
from diffusers import DDPMScheduler, DDIMScheduler

logger = logging.getLogger(__name__)


class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256):
        super().__init__()
        assert dim % 2 == 0, f"dim must be divisible by 2, found {dim}"
        self.features = features
        self.weights = nn.Parameter(torch.randn(dim // 2))
        self.to_out = nn.Linear(in_features=dim + 1, out_features=features)

    def to_embedding(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return self.to_out(fouriered)

    def forward(self, x: Union[Sequence[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.weights.device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        return self.to_embedding(x).view(*shape, self.features)  # type: ignore


class RMSNorm(nn.Module):
    def __init__(self, dim, feat_dim=-1, eps=1e-5):
        super().__init__()
        self.rms = dim ** -0.5
        self.feat_dim = feat_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x, unscaled=False):
        norm = torch.norm(x, dim=self.feat_dim, keepdim=True) * self.rms
        if unscaled:
            return x / norm.clamp(min=self.eps)
        g = self.scale
        if self.feat_dim != -1:
            while g.ndim <= self.feat_dim:
                g = g[None]
            while g.ndim < x.ndim:
                g = g.unsqueeze(-1)
        return x / norm.clamp(min=self.eps) * g


"""For VDiffusion"""


class Distribution:
    """Interface used by different distributions"""

    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class UniformDistribution(Distribution):
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin


class LogitNormalDistribution(Distribution):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.mean, self.std = mean, std

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        x = torch.from_numpy(np.random.lognormal(self.mean, self.std, num_samples)).to(device)
        return x / (1 + x)


class BernoulliDistribution(Distribution):
    def __init__(self, v1, v2):
        super().__init__()
        self.map = torch.tensor([v1, v2]).unsqueeze(0)

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        index = (torch.rand(num_samples, device=device) > 0.5).long()
        return self.map.repeat(num_samples, 1).to(device)[
            torch.arange(num_samples), index
        ]


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))


def Ts(t):
    """Builds a type template for a given type that accepts a list of instances"""
    return lambda *types: lambda: t(*[tp() for tp in types])


class Sequential(nn.Module):
    """Custom Sequential that includes all args"""

    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, *args) -> Tensor:
        for block in self.blocks:
            x = block(x, *args)
        return x


def Repeat(m, times: int):
    ms = (m,) * times
    return Sequential(*ms) if isinstance(m, nn.Module) else Ts(Sequential)(*ms)


class TimeEmbedding(nn.Module):
    def __init__(self, modulation_features, num_layers: int = 2, bias=True):
        super().__init__()
        self.embedding = NumberEmbedder(features=modulation_features)
        self.mlp = Repeat(
            nn.Sequential(
                nn.Linear(modulation_features, modulation_features, bias=bias),
                nn.GELU(),
            ),
            times=num_layers,
        )

    def forward(self, time):
        # Process time to time_features
        time_features = F.gelu(self.embedding(time))
        time_features = self.mlp(time_features)
        # Overlap features if more than one per batch
        if time_features.ndim == 3:
            time_features = reduce(time_features, "b n d -> b d", "sum")

        return time_features


def get_sigma(t: Tensor):
    angle = t * math.pi / 2
    sigma = torch.tan(angle)
    return sigma


@dataclass
class ModelArgs:
    # frontend
    phone_embed_dim: int = 512
    tone_embed_dim: int = 128
    n_phone: int = 300 + 2
    n_tone: int = 30 + 2

    local_cond_dim: int = 512
    time_embed_dim: int = 256

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
    ctx_feature: str = "bn"
    in_channels: int = 16
    out_channels: int = 16
    use_textprefix: bool = True

    bias: bool = False
    target_type: str = "velocity"
    use_prompt: bool = True
    use_unet_style_skip_connect: bool = True

    # for uniform t
    min_t: float = 0.0
    max_t: float = 1.0
    lognormal_mean: float = 0.0
    lognormal_std: float = 1.0

    p_max_t: float = 0.0

    use_seg_embed: bool = True
    use_bn_eos_bos: bool = True

    max_phone_len: int = 2000
    max_bn_len: int = 4000

    flashattn_version: str = "2.3"
    use_qk_norm = False


class Diffusion(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.min_t = hp.min_t if hasattr(hp, "min_t") else 0
        self.max_t = hp.max_t if hasattr(hp, "max_t") else 1

        self.target_type = hp.target_type if hasattr(hp, "target_type") else "velocity"
        self.use_prompt = hp.use_prompt if hasattr(hp, "use_prompt") else True

        self.act_fn = nn.GELU()
        self.bias = hp.bias

        # # text.
        # self.ph_proj = nn.Sequential(
        #     nn.Embedding(hp.n_phone, hp.phone_embed_dim, padding_idx=0),
        #     nn.Linear(hp.phone_embed_dim, hp.encoder_dim)
        # )
        # self.tone_proj = nn.Sequential(
        #     nn.Embedding(hp.n_tone, hp.tone_embed_dim, padding_idx=0),
        #     nn.Linear(hp.tone_embed_dim, hp.encoder_dim)
        # )

        # audio
        self.audio_proj = nn.Linear(hparams.get("hubert_dim", 1024), hp.encoder_dim)

        self.middle_proj = nn.Linear(hp.encoder_dim * 2, hp.encoder_dim)

        # time-embedding
        self.time_embedding = TimeEmbedding(hp.time_embed_dim, bias=self.bias)

        # global speaker-embedding
        self.prompt_encoder = ConvGlobalStacks(
            idim=hp.in_channels, n_chans=hp.spk_e_dim, odim=hp.spk_embed_dim)

        local_cond_in_channels = hp.out_channels + hp.spk_embed_dim

        self.local_cond_project = nn.Linear(
            local_cond_in_channels, hp.local_cond_dim, bias=self.bias)

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
        llama_args.use_qk_norm = hp.use_qk_norm
        llama_args.dim = hp.encoder_dim
        llama_args.n_layers = hp.encoder_n_layers
        llama_args.n_heads = hp.encoder_n_heads
        llama_args.max_seq_len = hp.max_seq_len
        self.encoder = LLaMa(llama_args)

        self.x_prenet = nn.Linear(hp.in_channels, hp.encoder_dim, bias=self.bias)
        self.prenet = nn.Linear(
            hp.time_embed_dim + hp.local_cond_dim, hp.encoder_dim, bias=self.bias
        )

        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels, bias=False)

        self.sigma_distribution = UniformDistribution(vmin=self.min_t, vmax=self.max_t)

        self.use_seg_embed = hp.use_seg_embed
        if hp.use_seg_embed:
            self.seg_embed = nn.Embedding(3, hp.encoder_dim, padding_idx=0)
            nn.init.trunc_normal_(self.seg_embed.weight, std=0.02, a=-0.04, b=0.04)

        if hp.use_bn_eos_bos:
            self.bn_eos_bos = nn.Parameter(torch.randn(2, hp.encoder_dim))

        if self.target_type == 'epsilon':
            self.noise_scheduler = DDPMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                num_train_timesteps=1000, rescale_betas_zero_snr=False, clip_sample=False)
        if self.target_type == 'vector_field':
            ''' sigma==0 means ODE'''
            from modules.flow_matching.vp_cfm import VariancePreservingConditionalFlowMatcher
            self.flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.0)

    def get_alpha_beta(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        angle = t * math.pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def get_t_from_sigma(self, sigma):
        return torch.arctan(sigma) / math.pi * 2

    def forward(self, inputs, sigmas=None, x_noisy=None, debug=False):
        audio_fea = self.audio_proj(inputs["audio_fea"])
        feat_lens = inputs["lat_lens"]
        if debug:
            breakpoint()

        ctx_feature = "lat_ctx"
        B, device = inputs[ctx_feature].size(0), inputs[ctx_feature].device

        local_cond = inputs[ctx_feature]

        # diffusion target
        x = inputs['lat']

        if self.target_type == "epsilon":
            noise = torch.randn_like(x)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (B,), device=x.device,
                dtype=torch.int64
            )
            x_noisy = self.noise_scheduler.add_noise(x, noise, timesteps)
            # time embedding
            t = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
            time_emb = self.time_embedding(t)
            time_emb = time_emb.unsqueeze(1).expand(-1, local_cond.shape[1], -1)
        elif self.target_type == "vector_field":
            """ Here, x is x1 in CFM """
            x0 = torch.randn_like(x)
            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x)
            time_emb = self.time_embedding(t)
            time_emb = time_emb.unsqueeze(1).expand(-1, local_cond.shape[1], -1)
            x_noisy = xt
        else:
            if sigmas is None:
                sigmas = self.sigma_distribution(num_samples=B, device=device).float()
            sigmas = torch.where(torch.rand_like(sigmas) < self.hp.p_max_t, torch.full_like(sigmas, self.hp.max_t),
                                 sigmas)
            sigmas_batch = extend_dim(sigmas, dim=x.ndim)
            alphas, betas = self.get_alpha_beta(sigmas_batch)

            # time embedding
            time_emb = self.time_embedding(sigmas)
            time_emb = time_emb.unsqueeze(1).expand(-1, local_cond.shape[1], -1)

            noise = torch.randn_like(x)

            if x_noisy is None:
                x_noisy = alphas * x + betas * noise

        residual = x_noisy
        if self.target_type == "velocity":
            target = alphas * noise - betas * x
        elif self.target_type == "x0":
            target = x
        elif self.target_type == "recflow":
            target = x - noise
        elif self.target_type == "epsilon":
            target = noise
        elif self.target_type == "vector_field":
            target = ut

        if debug:
            breakpoint()

        # concat condition.
        x_noisy = self.x_prenet(x_noisy) + self.prenet(
            torch.cat([time_emb, local_cond], dim=-1)
        )

        audio_fea = F.avg_pool1d(audio_fea.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)  # [B, T, H]
        assert audio_fea.shape[1] == x_noisy.shape[1]
        x = torch.cat((x_noisy, audio_fea), dim=2)
        encoder_input = self.middle_proj(x)
        attn_mask = self.sequence_mask(feat_lens, device=feat_lens.device) > 0

        if debug:
            breakpoint()

        encoder_out = self.encoder(encoder_input, attn_mask)

        pred_v = self.postnet(encoder_out)

        if self.target_type == "velocity":
            if self.hp.use_unet_style_skip_connect:
                pred = pred_v + residual
        else:
            pred = pred_v

        ret_dict = {
            "pred_v": pred,
            "target_v": target,
        }
        return ret_dict

    def sequence_mask(self, seq_lens, max_len=None, device='cpu'):
        b = seq_lens.shape[0]
        if max_len is None:
            max_len = seq_lens.max()
        mask = torch.arange(max_len).unsqueeze(0).to(device)  # [1, t]
        mask = mask < (seq_lens.unsqueeze(1))  # [1, t] + [b, 1] = [b, t]
        mask = mask.float()
        return mask

    def _forward(self, x, local_cond, audio_fea, timesteps, ctx_mask, use_torchdiffeq=False, torchdiffeq_cfg=3.5):
        """ When we use torchdiffeq, we need to include the CFG process inside _forward() """
        # print("Call _forward at", timesteps)
        x = x * (1 - ctx_mask)
        residual = x
        time_emb = self.time_embedding(timesteps)
        time_emb = time_emb.unsqueeze(1).expand(local_cond.shape[0], local_cond.shape[1], -1)
        x = self.x_prenet(x) + self.prenet(torch.cat([time_emb, local_cond], dim=-1))

        audio_fea = F.avg_pool1d(audio_fea.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)  # [B, T, H]
        assert audio_fea.shape[1] == x.shape[1]
        x = torch.cat((x, audio_fea), dim=2)
        x = self.middle_proj(x)

        pred_v = self.encoder(x, attn_mask=torch.ones((x.size(0), x.size(1)), device=x.device))

        pred_v = self.postnet(pred_v)

        if self.target_type == "velocity":
            if self.hp.use_unet_style_skip_connect:
                pred = pred_v + residual
        else:
            pred = pred_v

        if use_torchdiffeq:
            pred = torchdiffeq_cfg * pred[0:1] + (1 - torchdiffeq_cfg) * pred[1:2]
        return pred

    def ddim_sample(
            self,
            timesteps,
            local_cond,
            text_embed,
            cfg_w=1.0,
            inpaint_x=None,
            eta=0.0,
    ):
        t = timesteps
        _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
        x = torch.randn([1, frm_len, self.hp.out_channels], device=device)

        sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        for i in range(t):
            if self.target_type == "velocity":
                if cfg_w != 1:
                    v_pred, v_pred_uncond = self._forward(
                        x, local_cond, text_embed, timesteps=sigmas[i]
                    ).chunk(2)
                    v_pred = cfg_w * v_pred + (1 - cfg_w) * v_pred_uncond
                else:
                    v_pred = self._forward(
                        x, local_cond, text_embed, timesteps=sigmas[i]
                    )

                x_pred = alphas[i] * x - betas[i] * v_pred
                noise_pred = betas[i] * x + alphas[i] * v_pred

                # disable
                if inpaint_x is not None:
                    x_pred[:, : inpaint_x.shape[1], :] = inpaint_x
                    noise_pred[:, : inpaint_x.shape[1], :] = (
                                                                     x[:, : inpaint_x.shape[1], :] - alphas[
                                                                 i] * inpaint_x
                                                             ) / betas[i]

            if eta > 0:
                sigma = (
                        eta
                        * betas[i + 1]
                        / betas[i]
                        * torch.sqrt(1 - (alphas[i] / alphas[i + 1]) ** 2)
                )
                noise = torch.randn_like(noise_pred)
                x = (
                        alphas[i + 1] * x_pred
                        + torch.sqrt(betas[i + 1] ** 2 - sigma ** 2) * noise_pred
                        + sigma * noise
                )
            else:
                x = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

        return x

    def plms_sample(self, timesteps, local_cond, text_embed, text_cfg_w=1.0):
        t = timesteps
        batch_size, device, frm_len = (
            local_cond.size(0),
            local_cond.device,
            local_cond.size(1),
        )
        x = torch.randn([1, frm_len, self.hp.out_channels], device=device)
        if t > 20:
            sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
        else:
            sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device) ** 2
            # sigmas = torch.linspace(self.max_t, self.min_t, t+1, device=device)
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        pred_list = []
        for i in tqdm(range(t)):
            if text_cfg_w > 1:
                pred, pred_uncond = self._forward(
                    x,
                    local_cond,
                    text_embed,
                    timesteps=sigmas[i].expand(batch_size, -1),
                ).chunk(2)
                pred = text_cfg_w * pred + (1 - text_cfg_w) * pred_uncond
            else:
                pred = self._forward(x, local_cond, text_embed, timesteps=sigmas[i])

            if self.target_type == "velocity":
                x_pred = alphas[i] * x - betas[i] * pred
                noise_pred = betas[i] * x + alphas[i] * pred
            elif self.target_type == "noise":
                x_pred = (x - betas[i] * pred) / alphas[i]
                noise_pred = pred
            else:
                raise NotImplementedError

            if len(pred_list) == 0:
                x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
                if text_cfg_w > 1:
                    pred_prev, pred_prev_uncond = self._forward(
                        x_noisy,
                        local_cond,
                        text_embed,
                        timesteps=sigmas[i + 1].expand(batch_size, -1),
                    ).chunk(2)
                    pred_prev = (
                            text_cfg_w * pred_prev + (1 - text_cfg_w) * pred_prev_uncond
                    )
                else:
                    pred_prev = self._forward(
                        x_noisy, local_cond, text_embed, timesteps=sigmas[i + 1]
                    )
                pred_prime = (pred + pred_prev) / 2
            elif len(pred_list) == 1:
                pred_prime = (3 * pred - pred_list[-1]) / 2
            elif len(pred_list) == 2:
                pred_prime = (23 * pred - 16 * pred_list[-1] + 5 * pred_list[-2]) / 12
            elif len(pred_list) >= 3:
                pred_prime = (
                                     55 * pred
                                     - 59 * pred_list[-1]
                                     + 37 * pred_list[-2]
                                     - 9 * pred_list[-3]
                             ) / 24

            if self.target_type == "velocity":
                x_pred_prime = alphas[i] * x - betas[i] * pred_prime
                noise_pred_prime = betas[i] * x + alphas[i] * pred_prime
            elif self.target_type == "noise":
                x_pred_prime = (x - betas[i] * pred_prime) / alphas[i]
                noise_pred_prime = pred
            else:
                raise NotImplementedError

            x = alphas[i + 1] * x_pred_prime + betas[i + 1] * noise_pred_prime
            pred_list.append(pred)
        return x

    def consistency_sample(
            self,
            timesteps,
            local_cond,
            text_embed,
    ):
        t = timesteps
        _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))

        x = torch.randn([1, frm_len, self.hp.out_channels], device=device)

        sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
        sigmas = repeat(sigmas, "i -> i b", b=1)
        sigmas_batch = extend_dim(sigmas, dim=3)
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        for i in range(t):
            v_pred = self._forward(x, local_cond, text_embed, timesteps=sigmas[i])
            x_pred = alphas[i] * x - betas[i] * v_pred
            noise = torch.randn_like(v_pred)
            x = alphas[i + 1] * x_pred + betas[i + 1] * noise
        return x

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, sampler="ddim", cfg_w=1.0, **kwargs):
        # text_embed = self.ph_proj(inputs["phone"]) + self.tone_proj(inputs["tone"])  # [B, T, 1024]
        audio_fea = self.audio_proj(inputs["audio_fea"])

        ctx_feature = "lat_ctx"
        local_cond = inputs[ctx_feature]

        if self.target_type == 'epsilon':
            # Build scheduler
            scheduler = DDIMScheduler.from_config(
                self.noise_scheduler.config,
                rescale_betas_zero_snr=False, clip_sample=False, set_alpha_to_one=False, thresholding=False)
            scheduler.set_timesteps(timesteps)

            t = timesteps
            _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
            x = torch.randn([1, frm_len, self.hp.out_channels], device=device)
            x = x * scheduler.init_noise_sigma
            # sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
            # sigmas = repeat(sigmas, "i -> i b", b=1)

            for t in tqdm(scheduler.timesteps):
                x = scheduler.scale_model_input(x, timestep=t)
                t_ = torch.LongTensor([t]).to(local_cond.device)
                t_ = t_ / self.noise_scheduler.config.num_train_timesteps
                if cfg_w != 1:
                    v_pred, v_pred_uncond = self._forward(
                        torch.cat([x] * 2), local_cond, audio_fea, timesteps=t_, ctx_mask=inputs['ctx_mask']
                    ).chunk(2)
                    v_pred = cfg_w * v_pred + (1 - cfg_w) * v_pred_uncond
                else:
                    v_pred = self._forward(
                        x, local_cond, audio_fea, timesteps=t_
                    )
                x = scheduler.step(v_pred, t, x).prev_sample

        elif self.target_type == 'vector_field':
            t = timesteps
            _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
            sigmas = torch.linspace(self.max_t, self.min_t, t + 1, device=device)
            sigmas = repeat(sigmas, "i -> i b", b=1)
            if cfg_w != 1:
                traj = torchdiffeq.odeint(
                    lambda t, x: self._forward(torch.cat([x] * 2), local_cond, audio_fea, timesteps=t.unsqueeze(0),
                                               ctx_mask=inputs['ctx_mask'], use_torchdiffeq=True,
                                               torchdiffeq_cfg=cfg_w),
                    torch.randn([1, frm_len, self.hp.out_channels], device=device),
                    torch.linspace(0, 1, timesteps, device=device),
                    atol=1e-4,
                    rtol=1e-4,
                    # method="dopri5",
                    # method="rk4",
                    method="euler",
                )
            else:
                traj = torchdiffeq.odeint(
                    lambda t, x: self._forward(x, local_cond, audio_fea, timesteps=t.unsqueeze(0)),
                    torch.randn([1, frm_len, self.hp.out_channels], device=device),
                    torch.linspace(0, 1, timesteps, device=device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="dopri5",
                )
                print(traj.shape)
            x = traj[-1]

        else:
            if sampler == "ddim":
                x = self.ddim_sample(
                    timesteps, local_cond, audio_fea, cfg_w=cfg_w, **kwargs
                )
            elif sampler == "dpmsolver":
                x = self.dpmsolver_sample(
                    timesteps, local_cond, audio_fea, cfg_w=cfg_w, **kwargs
                )
            elif sampler == "plms":
                x = self.plms_sample(
                    timesteps, local_cond, audio_fea, cfg_w=cfg_w, **kwargs
                )
            elif sampler == "consistency":
                x = self.consistency_sample(
                    timesteps, local_cond, audio_fea
                )
            else:
                raise NotImplementedError

        return x


if __name__ == "__main__":
    config = ModelArgs()
    inputs = {}

    frontend = {
        "phone": torch.randint(0, 1000, (32, 20)).to("cuda:0"),
        "tone": torch.randint(0, 30, (32, 20)).to("cuda:0"),
        "word_seg": torch.randint(0, 8, (32, 20)).to("cuda:0"),
    }
    text_lens = torch.randint(0, 20, [32]).to("cuda:0")
    text_lens[0] = 20

    mel_ctx = torch.randn((32, 100, 80), dtype=torch.bfloat16).to("cuda:0")
    mel = torch.randn((32, 100, 80), dtype=torch.bfloat16).to("cuda:0")
    mel_len = torch.randint(0, 100, [32]).to("cuda:0")
    mel_len[0] = 100
    token = torch.randint(0, 1000, (32, 100)).to("cuda:0")

    prompt_mel = torch.randn((32, 80, 50), dtype=torch.bfloat16).to("cuda:0")

    text_mel_len = text_lens + mel_len

    mel_mask = sequence_mask(mel_len, max_len=100, device="cuda:0")
    text_mel_mask = sequence_mask(text_mel_len, max_len=120, device="cuda:0")

    inputs = {}
    inputs["token"] = token
    inputs["prompt_mel"] = prompt_mel
    inputs["mel"] = mel
    inputs["mel_mask"] = mel_mask
    inputs["mel_ctx"] = mel_ctx
    inputs["frontend"] = frontend
    inputs["text_mel_mask"] = text_mel_mask
    inputs["text_lens"] = text_lens
    inputs["mel_lens"] = mel_len
    from samantha.criterion.masked_loss import MaskedMAELoss

    loss_funcs = MaskedMAELoss()
    with torch.autocast(device_type="cuda", enabled=True):
        model = LlamaDiffusion(config).to("cuda:0")
        pred_v, v_target = model(inputs)
        mask = torch.ones(32, 100).to("cuda:0")
        loss = loss_funcs(pred_v, v_target, mask)
