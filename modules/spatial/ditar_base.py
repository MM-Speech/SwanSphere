import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
# import attrdict
from einops import rearrange, repeat
import torchdiffeq

from modules.spatial.align_ops import expand_states
from modules.commons.hf.transformer import TransformerEncoderModel, TransformerDecoderModel
from modules.commons.hf.transformer_config import TransformerConfig
from modules.commons.hf.transformer_dit import TransformerDiTModel
from modules.commons.hf.transformer_dit_config import TransformerDiTConfig

from utils.nn.seq_utils import sequence_mask, add_prefix, add_prefix_nd, remove_prefix, last_token_mask
from utils.nn.generation_utils import amo_sampling
from utils.commons.io import print_once
from utils.nn.seq_utils import get_incremental_state, set_incremental_state, softmax, make_positions


"""Shared spatial DiT-AR architecture with bottlenecked history conditioning.

The model omits stop prediction and injects the LM hidden condition through every DiT AdaLN block.
Instead of exposing the previous clean latent patch to DiT as four raw tokens,
it compresses that patch to a 64-D embedding and injects it through a separate
zero-initialized AdaLN adapter.  The bottleneck history can be continuously
corrupted and is replaced by a learned null embedding with a controlled
probability during training.  The exact h consumed by DiT can additionally be
aligned to frozen sequence-level audio-semantic embeddings.

The frozen audio teacher is intentionally owned by the training task rather
than by :class:`DiTARModel`.  Consequently teacher weights are neither
optimized nor copied into every DiTAR checkpoint, and inference only keeps the
learned h-to-semantic projection head contained in this module.
"""

@dataclass
class ModelArgs:    
    # text -> video token
    text_vocab_size: int = None
    text_dim: int = 1024
    video_input_dim: int = 512*2

    # audio
    patch_size: int = 4
    # This version accepts either no DiT latent-context history (0) or exactly
    # the immediately preceding latent patch (1).  Raw history tokens are never
    # concatenated to the current noisy patch.
    ctx_n_patches: int = 1
    # Width of the information bottleneck extracted from the previous DiT
    # latent patch before it is injected into DiT through AdaLN.
    dit_history_embedding_dim: int = 64
    # Total probability that an available DiT latent-context history embedding
    # is replaced by the learned null embedding.  This includes the fully
    # unconditional CFG branch and is 0.3 by design for this experiment.
    dit_history_dropout_prob: float = 0.3
    add_vad_mask: bool = False

    # caption
    caption_dim: int = 1024
    use_caption_encoder: bool = True
    exist_caption: bool = False
    crossattn_n_layers: int = 24
    
    # encoder
    encoder_dim: int = 1024
    encoder_n_layers: int = 6
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = 8
    
    # lm
    # False removes audio-patch feedback from LM in both training/inference.
    # Keep BOS, causal video-token attention/KV cache, and DiT e_ctx unchanged.
    use_lm_audio_history: bool = True
    lm_dim: int = 1024
    lm_enc_n_layers: int = 2
    lm_dec_n_layers: int = 36
    lm_n_heads: int = 16
    lm_n_kv_heads: int = 8
    
    # stop clf
    focal_loss_alpha: float = 0.99
    focal_loss_gamma: float = 1.5
    
    # decoder
    decoder_dim: int = 1024
    decoder_n_layers: int = 6
    decoder_n_heads: int = 16
    decoder_n_kv_heads: int = 8
    training_patch_keep_ratio: float = -1.0

    # Probability of the fully unconditional CFG branch.  It must not exceed
    # dit_history_dropout_prob so unconditional samples are a subset of the
    # samples whose DiT latent-history embedding is null.
    cfg_unconditional_dropout: float = 0.15

    # Continuously perturb the clean previous DiT latent patch on a
    # subset of full-CFG patches; the other full-CFG patches stay exactly clean.
    dit_ctx_corrupt_prob: float = 0.5
    # Noise RMS is scaled by the per-channel std of real z_ctx and by a ratio
    # sampled independently for every patch from [min, max].
    dit_ctx_noise_ratio_min: float = 0.0
    dit_ctx_noise_ratio_max: float = 0.25
    # Linearly ramp both ratio bounds from zero over this many training forwards.
    dit_ctx_noise_warmup_steps: int = 5000

    # Directly supervise the same post-lm_postnet h that conditions DiT.
    # ``audio_semantic_encoder`` selects the frozen sequence teacher owned by
    # the task: official M2D-CLAP frame features or ATST-Frame features.
    use_audio_semantic_supervision: bool = False
    audio_semantic_encoder: str = "m2d_clap"
    # M2D-CLAP p16x16 uses the official 5 * 768 concatenation by default.
    audio_semantic_embedding_dim: int = 3840
    
    # latent
    in_channels: int = 64
    out_channels: int = 64

    # trainging
    do_checkpoint: bool = False
    warm_up_lm: bool = False
    attn_implementation: str = 'flash_attention_2'
    instruction_finetuning: bool = False


class DiTARModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = self.hp = config

        if config.ctx_n_patches not in {0, 1}:
            raise ValueError(
                "Bottlenecked DiT history supports ctx_n_patches in {0, 1}, "
                f"got {config.ctx_n_patches}"
            )
        if config.dit_history_embedding_dim <= 0:
            raise ValueError("dit_history_embedding_dim must be positive")
        if not 0 <= config.dit_history_dropout_prob <= 1:
            raise ValueError("dit_history_dropout_prob must be in [0, 1]")
        if not 0 <= config.cfg_unconditional_dropout <= 1:
            raise ValueError("cfg_unconditional_dropout must be in [0, 1]")
        if config.cfg_unconditional_dropout > config.dit_history_dropout_prob:
            raise ValueError(
                "cfg_unconditional_dropout must not exceed "
                "dit_history_dropout_prob because unconditional samples must "
                "also use null DiT history"
            )
        if not 0 <= config.dit_ctx_corrupt_prob <= 1:
            raise ValueError("dit_ctx_corrupt_prob must be in [0, 1]")
        if config.dit_ctx_noise_ratio_min < 0:
            raise ValueError("dit_ctx_noise_ratio_min must be non-negative")
        if config.dit_ctx_noise_ratio_max < config.dit_ctx_noise_ratio_min:
            raise ValueError(
                "dit_ctx_noise_ratio_max must be no smaller than "
                "dit_ctx_noise_ratio_min"
            )
        if config.dit_ctx_noise_warmup_steps < 0:
            raise ValueError("dit_ctx_noise_warmup_steps must be non-negative")
        if config.audio_semantic_encoder not in {"m2d_clap", "atst_frame"}:
            raise ValueError(
                "audio_semantic_encoder must be 'm2d_clap' or 'atst_frame', "
                f"got {config.audio_semantic_encoder!r}"
            )
        if (
            config.use_audio_semantic_supervision
            and config.audio_semantic_embedding_dim <= 0
        ):
            raise ValueError(
                "audio_semantic_embedding_dim must be positive when audio "
                "semantic supervision is enabled, got "
                f"{config.audio_semantic_embedding_dim}"
            )
        
        # linguistic
        # self.text_embedder = nn.Embedding(config.text_vocab_size, config.text_dim)
        self.video_proj = nn.Linear(config.video_input_dim, config.lm_dim)
        self.video_encoder = TransformerEncoderModel(TransformerConfig(
            vocab_size=0, hidden_size=config.lm_dim, 
            intermediate_size=config.lm_dim * 4, num_hidden_layers=4, 
            num_attention_heads=16, num_key_value_heads=8, head_dim=config.lm_dim // 16,
            attn_implementation=config.attn_implementation, pad_token_id=0,
            rope_scaling={"rope_type": "linear", "factor": 1.0}
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
                attn_implementation=config.attn_implementation, pad_token_id=0,
                rope_scaling={"rope_type": "linear", "factor": 1.0}
            ))
            # raise NotImplementedError
        
        # encoder 编码latent，得到初步的h
        self.encoder_prenet = nn.Linear(config.in_channels, config.encoder_dim)
        self.encoder = TransformerEncoderModel(TransformerConfig(
            vocab_size=0, hidden_size=config.encoder_dim, 
            intermediate_size=config.encoder_dim * 4, num_hidden_layers=config.encoder_n_layers, 
            num_attention_heads=config.encoder_n_heads, num_key_value_heads=config.encoder_n_kv_heads, 
            head_dim=config.encoder_dim // config.encoder_n_heads,
            attn_implementation=config.attn_implementation, pad_token_id=0,
            rope_scaling={"rope_type": "linear", "factor": 1.0}
        ))
        self.encoder_cls_token = nn.Parameter(torch.randn((1, 1, config.encoder_dim)))
        self.encoder_postnet = nn.Linear(config.encoder_dim, config.lm_dim, bias=False)
        if not config.use_lm_audio_history:
            # Keep weights/keys for checkpoint compatibility, but do not train
            # or execute the now-unused LM audio-patch encoder.
            for module in (self.encoder_prenet, self.encoder, self.encoder_postnet):
                module.requires_grad_(False)
            self.encoder_cls_token.requires_grad_(False)
        
        # lm 进一步处理 h
        self.lm = TransformerDecoderModel(TransformerConfig(
            vocab_size=0, hidden_size=config.lm_dim, 
            intermediate_size=config.lm_dim * 4, num_hidden_layers=config.lm_dec_n_layers, 
            num_attention_heads=config.lm_n_heads, num_key_value_heads=config.lm_n_kv_heads, 
            head_dim=config.lm_dim // config.lm_n_heads, 
            use_dynamic_cross_gate=True, use_gated_attention=True,
            num_cross_attention_layers=config.lm_dec_n_layers if config.use_caption_encoder else 0,
            attn_implementation=config.attn_implementation, pad_token_id=0,
            rope_scaling={"rope_type": "linear", "factor": 1.0}
        ))
        self.lm.checkpoint_activations = config.do_checkpoint
        self.speech_start_token = nn.Parameter(torch.randn((1, 1, config.lm_dim)))
        self.lm_head = nn.Linear(config.lm_dim, config.lm_dim)
        self.lm_postnet = nn.Linear(config.lm_dim, config.decoder_dim, bias=False)
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
            attn_implementation=config.attn_implementation, is_decoder=True,
            pad_token_id=0,
            rope_scaling={"rope_type": "linear", "factor": 1.0}
        ))
        self.decoder.gradient_checkpointing = config.do_checkpoint
        self.decoder_prenet = nn.Linear(config.in_channels, config.decoder_dim)
        self.ctx_mask_embed = nn.Embedding(2, config.decoder_dim)
        self.decoder_postnet = nn.Linear(config.decoder_dim, config.out_channels)

        # Compress the immediately preceding latent patch [P, C_lat] into one
        # low-bandwidth embedding.  DiT never receives the raw history tokens.
        # Keeping the hidden width modest prevents the MLP from bypassing the
        # requested 64-D information bottleneck internally.
        history_input_dim = config.patch_size * config.in_channels
        history_hidden_dim = max(2 * config.dit_history_embedding_dim, 64)
        self.dit_history_encoder = nn.Sequential(
            nn.LayerNorm(history_input_dim),
            nn.Linear(history_input_dim, history_hidden_dim),
            nn.SiLU(),
            nn.Linear(history_hidden_dim, config.dit_history_embedding_dim),
            nn.LayerNorm(config.dit_history_embedding_dim),
        )
        self.dit_history_null = nn.Parameter(
            torch.randn(1, config.dit_history_embedding_dim) * 0.02
        )
        self.dit_history_adaln_norm = nn.LayerNorm(
            config.dit_history_embedding_dim,
            elementwise_affine=False,
        )
        self.dit_history_adaln_proj = nn.Linear(
            config.dit_history_embedding_dim,
            config.decoder_dim,
            bias=False,
        )
        # The new path is neutral for an old/random DiT at initialization.  Its
        # adapter still receives diffusion gradients on the first update.
        nn.init.zeros_(self.dit_history_adaln_proj.weight)

        # Map the LM condition into the shared AdaLN conditioning stream used
        # by every DiT block and the final AdaLN.  Normalizing first prevents
        # the rapidly growing LM-hidden norm from dominating the timestep
        # embedding.  A zero-initialized, bias-free connector makes this path
        # exactly neutral at initialization while still allowing its weights
        # to receive gradients from the diffusion loss on the first update.
        self.h_adaln_norm = nn.LayerNorm(
            config.decoder_dim, elementwise_affine=False
        )
        self.h_adaln_proj = nn.Linear(
            config.decoder_dim, config.decoder_dim, bias=False
        )
        nn.init.zeros_(self.h_adaln_proj.weight)

        # This head reads the exact 1024-D h used by DiT.  The frozen teacher
        # target is computed by the task and passed in as
        # ``audio_semantic_embeddings``.  M2D-CLAP therefore implements the
        # requested 1024 -> 3840 projection without changing h itself.
        if config.use_audio_semantic_supervision:
            self.h_audio_semantic_norm = nn.LayerNorm(
                config.decoder_dim, elementwise_affine=False
            )
            self.h_audio_semantic_proj = nn.Linear(
                config.decoder_dim,
                config.audio_semantic_embedding_dim,
                bias=False,
            )
        
        from modules.spatial.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.spatial.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(config.decoder_dim)

        # Keep the corruption warmup resumable without introducing a trainable
        # parameter.  One increment corresponds to one training forward on
        # every DDP rank; evaluation/inference does not advance this counter.
        self.register_buffer(
            "_dit_ctx_noise_train_step",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )

    def _h_to_adaln_condition(self, h: torch.Tensor) -> torch.Tensor:
        """Project one LM hidden condition per patch into DiT AdaLN space."""
        if h.ndim == 3:
            if h.shape[1] != 1:
                raise ValueError(
                    f"Expected a single LM condition token, got {tuple(h.shape)}"
                )
            h = h[:, 0]
        elif h.ndim != 2:
            raise ValueError(
                f"Expected LM condition [B, C] or [B, 1, C], got {tuple(h.shape)}"
            )

        h = self.h_adaln_norm(h.float()).to(self.h_adaln_proj.weight.dtype)
        return self.h_adaln_proj(F.silu(h))

    def _dit_history_to_adaln_condition(
        self,
        e_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Project one bottlenecked DiT-history embedding into AdaLN space."""
        if e_ctx.ndim == 3:
            if e_ctx.shape[1] != 1:
                raise ValueError(
                    "Expected one DiT-history embedding, got "
                    f"{tuple(e_ctx.shape)}"
                )
            e_ctx = e_ctx[:, 0]
        elif e_ctx.ndim != 2:
            raise ValueError(
                "Expected DiT-history embedding [B, D] or [B, 1, D], got "
                f"{tuple(e_ctx.shape)}"
            )

        e_ctx = self.dit_history_adaln_norm(e_ctx.float()).to(
            self.dit_history_adaln_proj.weight.dtype
        )
        return self.dit_history_adaln_proj(F.silu(e_ctx))

    def _null_dit_history_embedding(
        self,
        batch_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return self.dit_history_null.to(device=device, dtype=dtype).expand(
            batch_size, -1
        )

    def _encode_dit_history_patch(
        self,
        history_patch: torch.Tensor,
        history_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a masked previous latent patch into the 64-D bottleneck."""
        if history_patch.ndim != 3:
            raise ValueError(
                "Expected history_patch [B, P, C], got "
                f"{tuple(history_patch.shape)}"
            )
        expected_tail = (self.config.patch_size, self.config.in_channels)
        if tuple(history_patch.shape[1:]) != expected_tail:
            raise ValueError(
                f"Expected history_patch tail {expected_tail}, got "
                f"{tuple(history_patch.shape[1:])}"
            )
        if tuple(history_token_mask.shape) != tuple(history_patch.shape[:2]):
            raise ValueError(
                "history_token_mask must match history_patch[:2], got "
                f"{tuple(history_token_mask.shape)} versus "
                f"{tuple(history_patch.shape[:2])}"
            )

        masked_patch = history_patch * history_token_mask[..., None].to(
            history_patch.dtype
        )
        flattened = masked_patch.reshape(masked_patch.shape[0], -1)
        return self.dit_history_encoder(flattened)

    def _compute_audio_semantic_supervision(
        self,
        h: torch.Tensor,
        audio_semantic_embeddings: Optional[torch.Tensor],
        latent_lens: torch.Tensor,
    ):
        """Align current-patch h to a detached W-channel semantic target."""
        zero = h.new_zeros((), dtype=torch.float32)
        if not self.config.use_audio_semantic_supervision:
            return zero, zero, zero
        if audio_semantic_embeddings is None:
            raise KeyError(
                "Audio semantic supervision is enabled but "
                "inputs['audio_semantic_embeddings'] was not provided by "
                "the training task"
            )
        if audio_semantic_embeddings.ndim != 3:
            raise ValueError(
                "Expected audio_semantic_embeddings [B, T_patch, C], got "
                f"{tuple(audio_semantic_embeddings.shape)}"
            )
        expected_shape = (
            h.shape[0],
            h.shape[1],
            self.config.audio_semantic_embedding_dim,
        )
        if tuple(audio_semantic_embeddings.shape) != expected_shape:
            raise ValueError(
                f"Expected audio_semantic_embeddings {expected_shape}, got "
                f"{tuple(audio_semantic_embeddings.shape)}"
            )

        student = self.h_audio_semantic_norm(h.float())
        student = self.h_audio_semantic_proj(
            student.to(self.h_audio_semantic_proj.weight.dtype)
        )
        student = F.normalize(student.float(), dim=-1, eps=1e-6)
        teacher = F.normalize(
            audio_semantic_embeddings.detach().to(
                device=h.device,
                dtype=torch.float32,
            ),
            dim=-1,
            eps=1e-6,
        )

        cosine = (student * teacher).sum(dim=-1)
        patch_lens = torch.div(
            latent_lens.to(device=h.device, dtype=torch.long)
            + self.config.patch_size
            - 1,
            self.config.patch_size,
            rounding_mode="floor",
        )
        valid_mask = sequence_mask(patch_lens, maxlen=h.shape[1]).float()
        valid_count = valid_mask.sum().clamp_min(1.0)
        semantic_loss = ((1.0 - cosine) * valid_mask).sum() / valid_count
        mean_cosine = (cosine * valid_mask).sum() / valid_count
        return semantic_loss, mean_cosine.detach(), valid_count.detach()

    def _prepare_dit_history_embedding(
        self,
        history_patch: torch.Tensor,
        history_token_mask: torch.Tensor,
        history_drop_mask: torch.Tensor,
    ):
        """Corrupt, compress, and optionally null the previous latent patch.

        ``history_patch`` contains only the immediately preceding DiT latent
        patch.  Continuous noise is applied before the bottleneck encoder and
        only to real, retained history tokens.  A dropped or unavailable
        history is represented by the same learned null embedding used by
        inference CFG.
        """
        config = self.config
        dtype = history_patch.dtype
        token_mask_3d = history_token_mask[..., None].to(dtype)
        history_available_mask = history_token_mask.any(
            dim=1, keepdim=True
        ).to(dtype)
        history_keep_mask = (
            (1 - history_drop_mask.to(dtype)) * history_available_mask
        )

        corruption_stats = {
            "dit_history_available_fraction": history_available_mask.mean(),
            "dit_history_drop_fraction": history_patch.new_zeros(()),
            "dit_ctx_corrupt_fraction": history_patch.new_zeros(()),
            "dit_ctx_noise_ratio": history_patch.new_zeros(()),
            "dit_ctx_noise_ratio_max": history_patch.new_zeros(()),
            "dit_ctx_noise_delta_rms": history_patch.new_zeros(()),
        }
        with torch.no_grad():
            available_count = history_available_mask.sum().clamp_min(1.0)
            corruption_stats["dit_history_drop_fraction"] = (
                (history_available_mask * (1 - history_keep_mask)).sum()
                / available_count
            )

        history_clean = history_patch * token_mask_3d
        delta = torch.zeros_like(history_clean)
        corruption_enabled = (
            self.training
            and config.ctx_n_patches > 0
            and config.dit_ctx_corrupt_prob > 0
            and config.dit_ctx_noise_ratio_max > 0
        )
        if corruption_enabled:
            if config.dit_ctx_noise_warmup_steps > 0:
                warmup_scale = (
                    (self._dit_ctx_noise_train_step.to(torch.float32) + 1)
                    / config.dit_ctx_noise_warmup_steps
                ).clamp(max=1.0).to(dtype)
            else:
                warmup_scale = history_patch.new_ones(())

            ratio_min = config.dit_ctx_noise_ratio_min * warmup_scale
            ratio_max = config.dit_ctx_noise_ratio_max * warmup_scale
            ratio = torch.rand(
                (history_patch.shape[0], 1, 1),
                device=history_patch.device,
                dtype=dtype,
            )
            ratio = ratio_min + ratio * (ratio_max - ratio_min)

            corrupt_patch_mask = (
                torch.rand(
                    (history_patch.shape[0], 1, 1),
                    device=history_patch.device,
                ) < config.dit_ctx_corrupt_prob
            ).to(dtype)
            eligible_patch_mask = history_keep_mask[..., None]
            corrupt_patch_mask = corrupt_patch_mask * eligible_patch_mask

            # Estimate the channel scale from real previous-history tokens;
            # synthetic first-patch history and padding never affect it.
            history_stats = history_clean.detach().float()
            valid_stats_mask = token_mask_3d.float()
            valid_count = valid_stats_mask.sum(
                dim=(0, 1), keepdim=True
            ).clamp_min(1.0)
            channel_mean = (
                (history_stats * valid_stats_mask).sum(
                    dim=(0, 1), keepdim=True
                ) / valid_count
            )
            channel_var = (
                ((history_stats - channel_mean).square() * valid_stats_mask).sum(
                    dim=(0, 1), keepdim=True
                ) / valid_count
            )
            channel_scale = channel_var.sqrt().clamp_min(1e-4).to(dtype)
            active_element_mask = (
                corrupt_patch_mask * token_mask_3d
            )
            delta = (
                torch.randn_like(history_clean)
                * channel_scale
                * ratio
                * active_element_mask
            )

            with torch.no_grad():
                active_patches = corrupt_patch_mask.sum()
                eligible_patches = eligible_patch_mask.sum().clamp_min(1.0)
                active_elements = active_element_mask.sum().clamp_min(1.0)
                corruption_stats["dit_ctx_corrupt_fraction"] = (
                    active_patches / eligible_patches
                )
                corruption_stats["dit_ctx_noise_ratio"] = (
                    (ratio * corrupt_patch_mask).sum()
                    / active_patches.clamp_min(1.0)
                )
                corruption_stats["dit_ctx_noise_ratio_max"] = ratio_max.detach()
                corruption_stats["dit_ctx_noise_delta_rms"] = (
                    delta.float().square().sum()
                    / (active_elements * history_patch.shape[-1])
                ).sqrt().to(dtype)

        if self.training:
            with torch.no_grad():
                self._dit_ctx_noise_train_step.add_(1)

        encoded_history = self._encode_dit_history_patch(
            history_clean + delta,
            history_token_mask,
        )
        null_history = self._null_dit_history_embedding(
            history_patch.shape[0],
            dtype=encoded_history.dtype,
            device=encoded_history.device,
        )
        e_ctx = torch.where(
            history_keep_mask.bool(),
            encoded_history,
            null_history,
        )
        return e_ctx, history_keep_mask, corruption_stats
        
    def _training_history_latents(self, inputs, x, x_mask):
        """Input-side audio history, independent of the clean flow target."""
        return x, x_mask

    def _training_target_indices(self, inputs, batch_size, num_patches, device):
        """Optional flattened target selection before the expensive DiT path."""
        return None

    def _training_dit_conditions(self, inputs, h, history_patch, history_token_mask):
        # One categorical draw retains the original three CFG branches.
        cfg_rand = torch.rand((history_patch.shape[0], 1), device=history_patch.device)
        uncond_mask = (cfg_rand < self.config.cfg_unconditional_dropout).to(h.dtype)
        history_drop_mask = (
            cfg_rand < self.config.dit_history_dropout_prob
        ).to(history_patch.dtype)
        e_ctx, history_keep_mask, stats = self._prepare_dit_history_embedding(
            history_patch=history_patch,
            history_token_mask=history_token_mask,
            history_drop_mask=history_drop_mask,
        )
        return h * (1 - uncond_mask), e_ctx, history_keep_mask, stats

    def forward(self, inputs):
        if self.config.warm_up_lm:
            raise NotImplementedError
        
        x: torch.Tensor = inputs['lat']
        bsz, device = x.size(0), x.device
        x_lens = inputs['lat_lens']
        x_mask = sequence_mask(inputs['lat_lens'], maxlen=x.shape[1]).long() # 这个是attn mask
        history_x, history_x_mask = self._training_history_latents(inputs, x, x_mask)
        
        x_vid, x_vid_mask = self.forward_ling_encoder(inputs)
        if self.config.use_lm_audio_history:
            h, h_mask = self._encode_lm_history(history_x, history_x_mask)
        else:
            if x.shape[1] % self.config.patch_size:
                raise ValueError("latent length must be divisible by patch_size")
            num_patches = x.shape[1] // self.config.patch_size
            h = x_vid.new_zeros(bsz, num_patches, self.config.lm_dim)
            # Match forward_encoder's always-valid CLS mask without reading
            # audio values. Target padding is still handled by the loss mask.
            h_mask = x_mask.new_ones(bsz, num_patches)
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
        
        h = self.lm_head(h)  # [B, T, 2 * C] 修改为 [B, T, C]
        # h_mu, h_log_sigma = torch.chunk(h, 2, dim=-1)
        # h_sigma = torch.exp(h_log_sigma.clamp(-20, 20))     # [B, T, C]
        # eps = torch.randn_like(h_mu)
        # h = h_mu + h_sigma * eps
        # log_prob = -0.5 * (((h - h_mu) / h_sigma)**2 + 2*h_log_sigma + math.log(2*math.pi)).sum(-1)

        # Fixed-length spatial generation does not require stop prediction.
        # Keep zero-valued compatibility outputs so the existing task can use
        # this model without adding a stop-classifier parameter or computation.
        stop_loss = h.new_zeros(())
        stop_labels = h.new_zeros(h_mask.shape)

        h = self.lm_postnet(h)  # [B, T, C]

        # Supervise h before CFG condition dropout and diffusion-patch
        # subsampling.  The target is the current patch's pooled W-channel
        # teacher embedding, not an embedding of LM audio-history.
        semantic_loss, semantic_cosine, semantic_valid_patches = (
            self._compute_audio_semantic_supervision(
                h=h,
                audio_semantic_embeddings=inputs.get(
                    "audio_semantic_embeddings"
                ),
                latent_lens=x_lens,
            )
        )
        
        #######
        # dit #
        #######
        h = rearrange(h, 'b t c -> (b t) c')  # [BxT, C]
        if self.config.instruction_finetuning:
            prefix_mask = rearrange(prefix_mask, 'b t -> (b t) 1')  # [BxT, 1]
        ctx_n_patches, patch_size = self.config.ctx_n_patches, self.config.patch_size
        # The flow-matching sequence contains only the current target patch.
        # The previous DiT latent patch is extracted separately and compressed
        # to one e_ctx embedding, so raw history can no longer be copied through
        # self-attention.
        z = x.unfold(
            dimension=1,
            size=patch_size,
            step=patch_size,
        )
        z_mask = x_mask.unfold(
            dimension=1,
            size=patch_size,
            step=patch_size,
        )
        if ctx_n_patches > 0:
            history_sequence = torch.cat(
                [history_x.new_zeros((x.shape[0], patch_size, x.shape[2])),
                 history_x[:, :-patch_size]],
                dim=1,
            )
            history_mask_sequence = torch.cat(
                [history_x_mask.new_zeros((x_mask.shape[0], patch_size)),
                 history_x_mask[:, :-patch_size]],
                dim=1,
            )
            history_patch = history_sequence.unfold(
                dimension=1,
                size=patch_size,
                step=patch_size,
            )
            history_token_mask = history_mask_sequence.unfold(
                dimension=1,
                size=patch_size,
                step=patch_size,
            )
        else:
            history_patch = torch.zeros_like(z)
            history_token_mask = torch.zeros_like(z_mask)

        z = rearrange(z, 'b t c p -> (b t) p c')    # [BxT, P, C]
        z_mask = rearrange(z_mask, 'b t p -> (b t) p')  # [BxT, P]
        history_patch = rearrange(
            history_patch,
            'b t c p -> (b t) p c',
        )
        history_token_mask = rearrange(
            history_token_mask,
            'b t p -> (b t) p',
        )
        target_indices = self._training_target_indices(
            inputs, bsz, x.shape[1] // patch_size, device,
        )
        if target_indices is not None:
            h = h[target_indices]
            z = z[target_indices]
            z_mask = z_mask[target_indices]
            history_patch = history_patch[target_indices]
            history_token_mask = history_token_mask[target_indices]
        
        # CFG uses the same three branches as inference:
        #   1) full: h + bottlenecked DiT latent-history embedding;
        #   2) semantic-only: h + learned null DiT-history embedding;
        #   3) unconditional: zero h + learned null DiT-history embedding.
        # A single categorical draw avoids training a fourth, context-only
        # branch.  Hence unconditional is a subset of the exact total 0.3
        # DiT-history dropout probability.
        # Keep a clean endpoint for the current patch's flow-matching target.
        # Only the separate, input-side history patch may be corrupted.
        z_clean = z
        h, e_ctx, history_keep_mask, dit_ctx_corruption_stats = (
            self._training_dit_conditions(
            inputs=inputs,
            h=h,
            history_patch=history_patch,
            history_token_mask=history_token_mask,
            )
        )
        
        z0 = torch.randn_like(z_clean)
        t = self.flow_matcher.time_sampler.sample([z0.shape[0]], z0.device).type_as(z0)
        zt = t[:, None, None] * z_clean + (1 - t[:, None, None]) * z0
        ut = z_clean - z0
        
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t = self.f5_time_embed(t)
        # Both h and the low-bandwidth e_ctx have independent zero-initialized
        # adapters into the AdaLN stream consumed by every DiT layer and the
        # final AdaLN.  They remain separately controllable during inference.
        t = (
            t
            + self._h_to_adaln_condition(h).to(t)
            + self._dit_history_to_adaln_condition(e_ctx).to(t)
        )
        z_noisy = zt
        target = ut
        
        z_noisy = self.decoder_prenet(z_noisy)
        # All decoder tokens are current-patch tokens; history is AdaLN-only.
        target_token_mask = torch.zeros_like(z_mask)
        z_noisy = z_noisy + self.ctx_mask_embed(
            target_token_mask
        ).to(z_noisy.dtype)
        
        if target_indices is None and 0 < self.config.training_patch_keep_ratio < 1:
            n_patches_keep = max(1, int(z_clean.shape[0] * self.config.training_patch_keep_ratio))
            indices = torch.randperm(z_clean.shape[0], device=z_clean.device)[:n_patches_keep]
            z_noisy = z_noisy[indices]
            t = t[indices]
            z_mask = z_mask[indices]
            history_keep_mask = history_keep_mask[indices]
            target = target[indices]
        
        pred = self.decoder(
            inputs_embeds=z_noisy, 
            time_step=t.to(z_noisy.dtype), 
            attention_mask=z_mask
        ).last_hidden_state
        
        pred = self.decoder_postnet(pred)
        
        loss_mask = z_mask[..., None]
        if self.config.instruction_finetuning:
            loss_mask = loss_mask * (1 - prefix_mask[..., None])   # [BxT, P, 1] * [BxT, 1, 1]
        diff_loss = F.mse_loss(pred.float(), target.float(), reduction='none')
        diff_loss = (diff_loss * loss_mask).sum() / loss_mask.sum() / target.shape[-1]
        
        ret = {
            'stop_loss': stop_loss,
            'diff_loss': diff_loss,
            'audio_semantic_loss': semantic_loss,
            'audio_semantic_cosine': semantic_cosine,
            'audio_semantic_valid_patches': semantic_valid_patches,
            'pred': pred,
            'target': target,
            'loss_mask': loss_mask,
            # Compatibility key used only as a scalar monitor by the task.  It
            # now counts retained history embeddings rather than raw ctx tokens.
            'ctx_mask': history_keep_mask,
            'dit_history_keep_mask': history_keep_mask,
            'stop_labels': stop_labels,
            **dit_ctx_corruption_stats,
            # 'h_sigma': h_sigma.mean().detach(),
        }
        
        return ret
        
    def forward_ling_encoder(self, inputs):
        video_feats = inputs["video_feats"]
        global_video_feats = inputs["global_clip_embedding"]
        video_feats = torch.cat([video_feats, global_video_feats], dim=-1)
        v_embed = self.video_proj(video_feats)
        v_mask = inputs["v_mask"]
        x_vid = self.video_encoder(inputs_embeds=v_embed, attention_mask=v_mask).last_hidden_state
        x_vid = self.video_postnet(x_vid)       
        return x_vid, v_mask
    
    def _encode_lm_history(self, x, x_mask, *, inference=False):
        # Extension point for LM-only history processing. The shared base keeps its
        # original LocEnc behavior; DiT history and flow targets stay separate.
        return self.forward_encoder(x, x_mask)

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
                  past_key_values=None, use_tqdm=False, max_new_patches=None, temperature=0.0,
                  teacher_latent=None, lm_history_source=None, dit_history_source=None,
                  initial_noise=None, video_alignment_patches=None):
        video_feats = inputs['video_feats']
        # 确保 mask 类型为 long
        v_mask = inputs.get('v_mask', torch.ones(video_feats.shape[:2], device=video_feats.device).long())
        bsz, device = video_feats.size(0), video_feats.device
        ctx_n_patches, patch_size = self.config.ctx_n_patches, self.config.patch_size

        # Keep the original teacher_latent-only call backward compatible: when
        # neither source is specified, teacher_latent means teacher forcing for
        # both feedback paths; without it both paths are free-running.
        if lm_history_source is None:
            lm_history_source = "teacher" if teacher_latent is not None else "generated"
        if dit_history_source is None:
            dit_history_source = "teacher" if teacher_latent is not None else "generated"
        lm_history_source = str(lm_history_source).lower()
        dit_history_source = str(dit_history_source).lower()
        valid_history_sources = {"teacher", "generated"}
        if lm_history_source not in valid_history_sources:
            raise ValueError(
                f"lm_history_source must be one of {sorted(valid_history_sources)}, "
                f"got {lm_history_source!r}"
            )
        if dit_history_source not in valid_history_sources:
            raise ValueError(
                f"dit_history_source must be one of {sorted(valid_history_sources)}, "
                f"got {dit_history_source!r}"
            )
        if (
            (self.config.use_lm_audio_history and lm_history_source == "teacher")
            or dit_history_source == "teacher"
        ) and teacher_latent is None:
            raise ValueError(
                "teacher_latent is required when an enabled history source is 'teacher'"
            )

        # Predictions are always recorded in the returned sequence. Teacher
        # patches, when requested, are used only as the feedback history for a
        # later step and never replace the current prediction.
        teacher_patches = None
        if teacher_latent is not None:
            if teacher_latent.ndim != 3:
                raise ValueError(
                    f"teacher_latent must be [B, T, C], got {tuple(teacher_latent.shape)}"
                )
            if teacher_latent.shape[0] != bsz:
                raise ValueError(
                    f"teacher_latent batch {teacher_latent.shape[0]} != video batch {bsz}"
                )
            if teacher_latent.shape[-1] != self.config.out_channels:
                raise ValueError(
                    f"teacher_latent channels {teacher_latent.shape[-1]} != "
                    f"model channels {self.config.out_channels}"
                )
            if teacher_latent.shape[1] % patch_size != 0:
                raise ValueError(
                    f"teacher_latent length {teacher_latent.shape[1]} must be "
                    f"divisible by patch_size={patch_size}"
                )
            teacher_latent = teacher_latent.to(device=device)
            teacher_patches = rearrange(
                teacher_latent,
                "b (t p) c -> b t p c",
                p=patch_size,
            )
        
        global_clip_embedding = inputs['global_clip_embedding']
        
        # 编码视频
        x_vid, _ = self.forward_ling_encoder({"video_feats": video_feats, "v_mask": v_mask, "global_clip_embedding": global_clip_embedding})
        T_vid_raw = x_vid.shape[1]

        # 计算目标音频长度
        if max_new_patches is not None:
            T_aud = max_new_patches
        elif teacher_patches is not None:
            T_aud = teacher_patches.shape[1]
        else:
            scale_factor = getattr(self.config, 'patches_per_frame', 21.53 / (4.0 * self.config.patch_size))
            T_aud = max(1, math.ceil(T_vid_raw * scale_factor))
        if teacher_patches is not None and teacher_patches.shape[1] < T_aud:
            raise ValueError(
                f"teacher_latent contains {teacher_patches.shape[1]} patches, "
                f"but inference requested {T_aud}"
            )
        if initial_noise is not None:
            expected_noise_shape = (
                T_aud,
                bsz,
                patch_size,
                self.config.out_channels,
            )
            if tuple(initial_noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"initial_noise must be [T_aud, B, P, C] with shape "
                    f"{expected_noise_shape}, got {tuple(initial_noise.shape)}"
                )
            if not torch.is_floating_point(initial_noise):
                raise TypeError(
                    f"initial_noise must be floating point, got {initial_noise.dtype}"
                )
            if not torch.isfinite(initial_noise).all():
                raise ValueError("initial_noise contains NaN or Inf")
            initial_noise = initial_noise.to(device=device)

        # A sampled prefix must keep the full clip's video timeline instead of
        # stretching the complete video over only max_new_patches audio tokens.
        alignment_patches = T_aud if video_alignment_patches is None else video_alignment_patches
        if (type(alignment_patches) is not int or alignment_patches < T_aud
                or alignment_patches <= 0):
            raise ValueError("video_alignment_patches must be an integer >= generated patch count")
        # 显式对齐
        step_indices = torch.arange(T_aud, device=device).float()
        vid_indices = (step_indices * T_vid_raw / alignment_patches).floor().long().clamp(max=T_vid_raw - 1)
        x_vid_aligned = x_vid[:, vid_indices, :] # [B, T_aud, C]

        # ===========================
        # 2. 初始化状态
        # ===========================
        # A. LM 初始输入: Start + Video[0]
        current_token = self.speech_start_token.expand(bsz, 1, -1)
        lm_input = current_token + x_vid_aligned[:, 0:1, :]
        
        # B. DiT latent-context history.  Keep only the previous raw patch as
        # state; it is compressed once to e_ctx before each ODE solve and is
        # never concatenated to the noisy current-patch tokens.
        dit_history_patch = torch.zeros(
            bsz,
            patch_size,
            self.config.out_channels,
            device=device,
        )
        dit_history_available = torch.zeros(
            bsz,
            1,
            device=device,
            dtype=torch.bool,
        )
        
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
            h = self.lm_head(h) # 抛弃了采样
            # h_mu, h_log_sigma = torch.chunk(h, 2, dim=-1)
            # h_sigma = torch.exp(h_log_sigma.clamp(-20, 20))
            
            # if temperature > 1e-5:
            #     eps = torch.randn_like(h_mu)
            #     h = h_mu + h_sigma * eps * temperature
            # else:
            #     h = h_mu
                
            # 4. Project to DiT Dimension
            h_cond = self.lm_postnet(h)
            
            return h_cond, new_past_key_values

        # --- 函数 B: DiT Derivative (ODE Solver 需要) ---
        def forward_dit_step(x, cond, t, seq_cfg_w=(1.4, 3), timestep_annealing_w=(1.0, 0.0, 1.0)):
            e_ctx_inner = cond['e_ctx']
            h_inner = cond['h']

            if x.shape[0] != 3 * bsz:
                raise ValueError(
                    f"CFG expects a 3x batch, got {x.shape[0]} for batch {bsz}"
                )

            # Build genuinely different CFG branches without exposing raw DiT
            # latent-context tokens: full=(h,e_ctx), semantic-only=(h,null),
            # unconditional=(0,null).
            x = self.decoder_prenet(x)
            
            h_expanded = torch.cat(
                [h_inner, h_inner, torch.zeros_like(h_inner)], dim=0
            )
            h_adaln = self._h_to_adaln_condition(h_expanded)

            null_e_ctx = self._null_dit_history_embedding(
                bsz,
                dtype=e_ctx_inner.dtype,
                device=e_ctx_inner.device,
            )
            e_ctx_expanded = torch.cat(
                [e_ctx_inner, null_e_ctx, null_e_ctx],
                dim=0,
            )
            e_ctx_adaln = self._dit_history_to_adaln_condition(
                e_ctx_expanded
            )
            
            # Every decoder token belongs to the current patch.  The old token
            # type embedding is retained for checkpoint compatibility, but its
            # context class is no longer used.
            x_mask = torch.ones_like(x[..., 0]).long()
            ctx_mask = torch.zeros_like(x_mask)
            x = x + self.ctx_mask_embed(ctx_mask.long()).to(x.dtype)
            
            with torch.amp.autocast('cuda', dtype=torch.float32):
                t_embed = self.f5_time_embed(t).expand(x.shape[0], -1)
            adaln_condition = (
                t_embed
                + h_adaln.to(t_embed)
                + e_ctx_adaln.to(t_embed)
            )
            
            # DiT Forward
            pred = self.decoder(
                inputs_embeds=x,
                time_step=adaln_condition.to(x.dtype),
                attention_mask=x_mask,
            ).last_hidden_state
            
            pred = self.decoder_postnet(pred)
            
            # CFG
            cond_all, cond_semantic, uncond = pred.chunk(3)
            a, b, p = timestep_annealing_w
            gamma_t = a + b * torch.pow(1 - t, p)
            seq_cfg_w_t = [gamma_t * w for w in seq_cfg_w]
            
            grad = (
                uncond
                + seq_cfg_w_t[0] * (cond_semantic - uncond)
                + seq_cfg_w_t[1] * (cond_all - cond_semantic)
            )
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
            
            # === Step B: compress DiT latent-context history and generate ===
            history_token_mask = dit_history_available.expand(
                -1, patch_size
            )
            encoded_e_ctx = self._encode_dit_history_patch(
                dit_history_patch,
                history_token_mask,
            )
            null_e_ctx = self._null_dit_history_embedding(
                bsz,
                dtype=encoded_e_ctx.dtype,
                device=encoded_e_ctx.device,
            )
            e_ctx = torch.where(
                dit_history_available,
                encoded_e_ctx,
                null_e_ctx,
            )
            cond = {'e_ctx': e_ctx, 'h': h}
            
            t_schedule = torch.linspace(0, 1, timesteps + 1, device=device)
            t_schedule = 0.5 * (1 - torch.cos(torch.pi * t_schedule))
            
            if initial_noise is None:
                z_init = torch.randn(
                    [bsz, patch_size, self.config.out_channels], device=device
                )
            else:
                # odeint currently does not mutate its initial state, but clone
                # here so every ablation mode is isolated from implementation
                # changes in the solver.
                z_init = initial_noise[step].clone()
            
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
                next_dit_history_patch = (
                    teacher_patches[:, step]
                    if dit_history_source == "teacher"
                    else x_patch
                )

                # Keep causal video context in the LM KV cache in both modes.
                next_vid_feat = x_vid_aligned[:, step+1 : step+2, :]
                if self.config.use_lm_audio_history:
                    lm_history_patch = (
                        teacher_patches[:, step]
                        if lm_history_source == "teacher"
                        else x_patch
                    )
                    h_patch, _ = self._encode_lm_history(
                        lm_history_patch,
                        torch.ones_like(lm_history_patch[..., 0]).long(),
                        inference=True,
                    )
                    lm_input = h_patch + next_vid_feat
                else:
                    # No teacher/generated audio reaches LM; DiT still has
                    # its independent bottlenecked latent-context feedback.
                    lm_input = next_vid_feat
                
                # 4. Update DiT latent-context history.  The source can still
                # be teacher-forced for diagnosis, but only its bottleneck
                # embedding is visible to the next DiT patch.
                if ctx_n_patches > 0:
                    dit_history_patch = next_dit_history_patch
                    dit_history_available = torch.ones_like(
                        dit_history_available
                    )
        
        res_lat = torch.cat(res_patches, dim=1)
        return res_lat, []
    
if __name__ == '__main__':
    config = ModelArgs()
    config.text_vocab_size = 65536
    config.use_caption_encoder = True

    device = torch.device('cuda:0')

    model = DiTARModel(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"{params / 1024 /1024} MB")

    inputs = {
        'lat': torch.randn(2, 216, 64, device=device),
        'lat_lens': torch.tensor([215, 215], device=device),
        'video_feats': torch.randn(2, 40, 512, device=device),
        'v_mask': torch.ones(2, 40, device=device),
        'global_clip_embedding': torch.randn(2, 40, 512, device=device),
        
        # 'caption_emb': torch.randn(2, 13, 1024, device=device),
        # 'caption_lens': torch.tensor([13, 13], device=device),
        
    }

    with torch.autocast(device_type='cuda'):
        ret = model(inputs)
    inputs = {
        'video_feats': torch.randn(1, 20, 512, device=device, dtype=torch.float32),
        'v_mask': torch.ones(1, 20, device=device, dtype=torch.long),
        
        'global_clip_embedding': torch.randn(1, 20, 512, device=device),

    }
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # V2A 逻辑: 自动根据视频长度推导音频长度，无需 ref_lat
            generated_lat, _ = model.inference(inputs, use_tqdm=True)
    
    print(generated_lat.shape)
