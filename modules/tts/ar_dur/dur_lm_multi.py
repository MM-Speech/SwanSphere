from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.asr.llama.llama import LLaMa, ModelArgs as LLaMaModelArgs
from modules.commons.pos_encoding import SinusoidalPositionalEmbedding
from utils.nn.seq_utils import add_prefix_nd, sequence_mask, remove_prefix, remove_suffix
from utils.nn.generation_utils import sample_topk, sample
from utils.commons.dataset_utils import pad_or_cut_xd
from utils.text.text_encoder import TokenTextEncoder
from utils.text import PHONE_VOCAB, TONE_VOCAB
from utils.nn.tokenizers import QuantTokenizer
from utils.commons.io import print_once
from modules.tts.ar_dur.dur_lm import sample_dur,ContinuousTimeEncoderV2,ContinuousTimeEncoder,ModelArgs

class MultiConditionedSeq2SeqDurationLM(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.txt_embed = nn.Embedding(config.vocab_size, config.lm_config.dim, config.padding_idx)

        # --- NEW: speaker embedding（与 ph 对齐，逐 token 相加）---
        self.spk_embed = nn.Embedding(
            getattr(config, 'spk_vocab_size', 3),
            config.lm_config.dim,
            getattr(config, 'spk_padding_idx', 0)
        )

        if config.loss_type in ['nll', 'mse']:
            if config.dur_embed_v2:
                self.dur_embed = ContinuousTimeEncoderV2(
                    config.lm_config.dim,
                    config.dur_embed_fourier_K,
                    config.dur_embed_fourier_scale,
                    min_value=0,
                    max_value=config.dur_max_value + 10,
                    use_norm=True
                )
            else:
                self.dur_embed = ContinuousTimeEncoder(
                    config.lm_config.dim,
                    config.dur_embed_fourier_K,
                    config.dur_embed_fourier_scale
                )
            if config.loss_type == 'nll':
                self.lm_head = nn.Linear(config.lm_config.dim, 2, bias=config.use_out_bias)
            elif config.loss_type == 'mse':
                self.lm_head = nn.Linear(config.lm_config.dim, 1, bias=config.use_out_bias)

        elif config.loss_type == 'ce':
            self.dur_embed = nn.Embedding(config.dur_max_value + 5, config.lm_config.dim)
            self.lm_head = nn.Linear(config.lm_config.dim, config.dur_max_value + 5, bias=False)

        if config.is_seq2seq:
            self.cond_proj = nn.Linear(config.cond_dim, config.lm_config.dim, bias=False)

        if config.use_sep:
            self.sep_token = nn.Parameter(torch.randn((1, 1, config.lm_config.dim)))
        if config.use_hard_pos_encoding:
            self.hard_pos_embed = SinusoidalPositionalEmbedding(config.lm_config.dim)

        if config.is_seq2seq:
            from modules.asr.llama.llama_seq2seq import Seq2SeqLLaMA
            self.lm = Seq2SeqLLaMA(config.lm_config)
        else:
            self.lm = LLaMa(config.lm_config)

        self.config = config
        
    def forward_dur_embed(self, dur_tokens):
        if self.config.loss_type in ['nll', 'mse']:
            dur_tokens = torch.log1p(dur_tokens.float())
        elif self.config.loss_type == 'ce':
            dur_tokens = dur_tokens.long()
        dur_embeds = self.dur_embed(dur_tokens)
        return dur_embeds
        
    # ===== modified =====
    def forward(self, inputs, instruction_finetuning=False):
        config = self.config

        # text / phone tokens
        txt_tokens = inputs['merged_ph_tokens']      # [B, T]
        txt_lens   = inputs['merged_ph_tokens_len']
        txt_embeds = self.txt_embed(txt_tokens)      # [B, T, C]

        # --- NEW: speaker ids，与 ph 对齐；未提供则回退为全 0 ---
        spk_ids = inputs.get('spk_ids', None)
        if spk_ids is None:
            spk_ids = torch.zeros_like(txt_tokens)   # 0 为 padding/无说话人
        spk_embeds = self.spk_embed(spk_ids)         # [B, T, C]

        # 与 ph 路径一致：逐位相加后再加位置编码（避免 pos 叠加两次）
        txt_embeds = txt_embeds + spk_embeds

        # duration tokens / embeds
        dur_tokens = inputs['dur_tokens']            # [B, T]
        dur_embeds = self.forward_dur_embed(dur_tokens)

        if self.config.use_hard_pos_encoding:
            txt_embeds = txt_embeds + self.hard_pos_embed(
                torch.arange(txt_embeds.shape[1], device=txt_embeds.device),
                out_dtype=txt_embeds.dtype, device=txt_embeds.device
            )[None, ...]
            dur_embeds = dur_embeds + self.hard_pos_embed(
                torch.arange(dur_embeds.shape[1], device=dur_embeds.device),
                out_dtype=dur_embeds.dtype, device=dur_embeds.device
            )[None, ...]

        # 组装解码序列
        if self.config.modeling_type == 'ar_interleaved':
            x = torch.zeros_like(txt_embeds).repeat(1, 2, 1)
            x[:, 0::2, :] = txt_embeds
            x[:, 1::2, :] = dur_embeds
            x_mask = sequence_mask(2 * txt_lens)

        elif self.config.modeling_type == 'ar':
            if self.config.use_sep:
                x = add_prefix_nd(
                    txt_embeds, txt_lens, self.sep_token.to(txt_embeds).repeat(txt_embeds.shape[0], 1, 1),
                    torch.ones(txt_embeds.shape[0], dtype=torch.long, device=txt_embeds.device)
                )
                x = add_prefix_nd(x, txt_lens + 1, dur_embeds, txt_lens)
                x_mask = sequence_mask(2 * txt_lens + 1)
            else:
                x = add_prefix_nd(txt_embeds, txt_lens, dur_embeds, txt_lens)
                x_mask = sequence_mask(2 * txt_lens)

        # 主干 LM
        if self.config.is_seq2seq:
            cond = inputs['condition']
            cond_lens = inputs['condition_lens']
            cond = self.cond_proj(cond)

            x = self.lm(
                encoder_x=cond,
                encoder_padding_mask=sequence_mask(cond_lens),
                decoder_x=x,
                decoder_padding_mask=x_mask,
                do_checkpoint=self.config.do_checkpoint
            )
        else:
            x = self.lm(x, attn_mask=x_mask, do_checkpoint=self.config.do_checkpoint)

        # 取出与 ph 对齐的时刻
        if self.config.modeling_type == 'ar_interleaved':
            x = x[:, 0::2, :]
        elif self.config.modeling_type == 'ar':
            if self.config.use_sep:
                x = remove_prefix(x, txt_lens, txt_lens)
            else:
                x = remove_prefix(x, txt_lens - 1, txt_lens)

        x = self.lm_head(x)

        # === loss 计算保持不变 ===
        loss_mask = sequence_mask(txt_lens)[..., None]
        if instruction_finetuning:
            dur_ctx_len_min = torch.clamp_min(txt_lens * 0.01, 1)
            dur_ctx_len_max = torch.clamp_min(txt_lens - dur_ctx_len_min, 1)
            dur_ctx_len = torch.rand(txt_lens.shape[0]).to(txt_lens.device) * (dur_ctx_len_max - dur_ctx_len_min) + dur_ctx_len_min
            dur_ctx_len = dur_ctx_len.long()
            ctx_mask = sequence_mask(dur_ctx_len, loss_mask.shape[1])[..., None]
            loss_mask[ctx_mask > 0] = 0
            if loss_mask.sum() == 0:
                loss_mask = sequence_mask(txt_lens)[..., None]

        loss_w_mask = sequence_mask(txt_lens)[..., None]
        if 'loss_w_mask' in inputs:
            loss_w_mask = inputs['loss_w_mask']

        if config.loss_type == 'nll':
            mu_raw, s_raw = x.chunk(2, dim=-1)
            sigma = F.softplus(s_raw) + 1e-5
            z = dur_tokens[..., None].log1p()
            nll = 0.5 * ((z - mu_raw)**2 / (sigma**2) + 2.0 * torch.log(sigma))
            loss = (nll * loss_mask * loss_w_mask).sum() / loss_mask.sum()
            with torch.no_grad():
                r2 = (((z - mu_raw) / sigma)**2)
                r2 = (r2 * loss_mask).sum() / loss_mask.sum()
                m_log_sigma = torch.log(sigma)
                m_log_sigma = (m_log_sigma * loss_mask).sum() / loss_mask.sum()
            outputs = {
                'loss': loss,
                'gt_dur': (dur_tokens[..., None].log1p() * loss_mask).sum() / loss_mask.sum(),
                'mu': (mu_raw * loss_mask).sum() / loss_mask.sum(),
                'sigma': (sigma * loss_mask).sum() / loss_mask.sum(),
                'r2': r2,
                'm_log_sigma': m_log_sigma,
                'ntokens': (txt_lens * 2).sum()
            }

        elif config.loss_type == 'mse':
            z = dur_tokens[..., None].log1p()
            mse = F.mse_loss(x, z, reduction='none')
            loss = (mse * loss_mask * loss_w_mask).sum() / loss_mask.sum()
            outputs = {'loss': loss, 'ntokens': (txt_lens * 2).sum()}

        elif config.loss_type == 'ce':
            loss = F.cross_entropy(x.transpose(1, 2), dur_tokens.long(), reduction='none')
            loss_mask_ = loss_mask[..., 0]
            loss = (loss * loss_mask_ * loss_w_mask[..., 0]).sum() / loss_mask_.sum()
            outputs = {'loss': loss, 'ntokens': (txt_lens * 2).sum()}

        return outputs

    
    @torch.no_grad()
    def prefill(self, txt_tokens, dur_tokens, enc_out=None, spk_ids=None):
        """
        预填缓存：支持可选 spk_ids（与 txt_tokens 对齐）；不传则默认全 0。
        """
        self.reset_kv_cache()

        if spk_ids is None:
            spk_ids = torch.zeros_like(txt_tokens)
        spk_embeds = self.spk_embed(spk_ids)

        if self.config.modeling_type == 'ar_interleaved':
            txt_embeds = self.txt_embed(txt_tokens) + spk_embeds
            dur_embeds = self.forward_dur_embed(dur_tokens)

            if self.config.use_hard_pos_encoding:
                txt_embeds = txt_embeds + self.hard_pos_embed(
                    torch.arange(txt_embeds.shape[1], device=txt_embeds.device),
                    out_dtype=txt_embeds.dtype, device=txt_embeds.device
                )[None, ...]
                dur_embeds = dur_embeds + self.hard_pos_embed(
                    torch.arange(dur_embeds.shape[1], device=dur_embeds.device),
                    out_dtype=dur_embeds.dtype, device=dur_embeds.device
                )[None, ...]

            prefix = torch.zeros_like(txt_embeds).repeat(1, 2, 1)
            prefix[:, 0::2, :] = txt_embeds
            prefix[:, 1::2, :] = dur_embeds
            prefix_mask = torch.ones_like(prefix)[..., 0]

        elif self.config.modeling_type == 'ar':
            prefix = self.txt_embed(txt_tokens) + spk_embeds
            if self.config.use_hard_pos_encoding:
                prefix = prefix + self.hard_pos_embed(
                    torch.arange(prefix.shape[1], device=prefix.device),
                    out_dtype=prefix.dtype, device=prefix.device
                )[None, ...]
            prefix_mask = torch.ones_like(prefix)[..., 0]

        if self.config.is_seq2seq:
            if enc_out is None:
                enc_out = torch.zeros((1, 1, self.config.cond_dim)).to(prefix)
            _ = self.lm.decode(
                decoder_x=prefix,
                decoder_padding_mask=None,
                enc_out=enc_out,
                encoder_padding_mask=None,
                start_pos=0,
                use_cache=True
            )
            start_pos = prefix.shape[1]
        else:
            _ = self.lm(prefix, attn_mask=prefix_mask, start_pos=0, use_cache=True)
            start_pos = prefix.shape[1]

        return start_pos

    
    def reset_kv_cache(self):
        if self.config.is_seq2seq:
            self.lm.reset_kv_cache()
    
    @torch.no_grad()
    def inference(self, txt_tokens, condition=None, dur_tokens=None, spk_ids=None, start_pos=0, timestep=0,
                topk=1, temperature=0.1, stochastic_round=True,
                use_tqdm=True, print_candidates=False):
        """
        推理：新增可选 spk_ids（与 txt_tokens 对齐）；不传则默认全 0。
        """
        if self.config.is_seq2seq:
            if condition is not None:
                enc_out = self.lm.encode(self.cond_proj(condition), encoder_padding_mask=torch.ones_like(condition)[..., 0])
            else:
                enc_out = torch.zeros((1, 1, self.config.cond_dim)).to(dtype=self.precision, device=txt_tokens.device)

        if spk_ids is None:
            spk_ids = torch.zeros_like(txt_tokens)

        tgt_len = txt_tokens.shape[1]
        if dur_tokens is not None:
            if self.config.modeling_type == 'ar_interleaved':
                assert dur_tokens.shape[1] <= txt_tokens.shape[1]
                prefix_tokens = txt_tokens[:, :dur_tokens.shape[1]]
                prefix_spk_ids = spk_ids[:, :dur_tokens.shape[1]]
                txt_tokens = txt_tokens[:, dur_tokens.shape[1]:]
                spk_ids   = spk_ids[:, dur_tokens.shape[1]:]
                start_pos = self.prefill(prefix_tokens, dur_tokens, enc_out, spk_ids=prefix_spk_ids)
                if self.config.use_hard_pos_encoding:
                    timestep = prefix_tokens.shape[1]

            elif self.config.modeling_type == 'ar':
                if start_pos == 0:
                    assert dur_tokens.shape[1] <= txt_tokens.shape[1]
                    tgt_len = txt_tokens.shape[1] - dur_tokens.shape[1]
                tgt_len = tgt_len - 1
                if self.config.use_hard_pos_encoding:
                    timestep = dur_tokens.shape[1]

        it = range(tgt_len)
        if use_tqdm:
            from tqdm import tqdm
            it = tqdm(it, desc='| Generating Dur')

        if self.config.modeling_type == 'ar_interleaved':
            def forward_ar_interleaved(txt_token, spk_token, start_pos):
                txt_embeds = self.txt_embed(txt_token) + self.spk_embed(spk_token)
                if self.config.use_hard_pos_encoding:
                    pos_embed = self.hard_pos_embed(timestep=timestep, out_dtype=txt_embeds.dtype, device=txt_embeds.device)[None, None, :]
                    txt_embeds = txt_embeds + pos_embed
                if self.config.is_seq2seq:
                    x = self.lm.decode(
                        decoder_x=txt_embeds, decoder_padding_mask=None, enc_out=enc_out, encoder_padding_mask=None,
                        start_pos=start_pos, use_cache=True
                    )
                else:
                    x = self.lm(txt_embeds, attn_mask=None, start_pos=start_pos, use_cache=True)
                x = self.lm_head(x)

                dur_pred = sample_dur(x, topk, temperature, stochastic_round, self.config.loss_type)
                dur_embeds = self.forward_dur_embed(dur_pred)
                if self.config.use_hard_pos_encoding:
                    dur_embeds = dur_embeds + pos_embed

                if self.config.is_seq2seq:
                    _ = self.lm.decode(
                        decoder_x=dur_embeds, decoder_padding_mask=None, enc_out=enc_out, encoder_padding_mask=None,
                        start_pos=start_pos + 1, use_cache=True
                    )
                else:
                    _ = self.lm(dur_embeds, attn_mask=None, start_pos=start_pos + 1, use_cache=True)

                return dur_pred

            dur_preds = []
            for txt_i in it:
                dur_pred = forward_ar_interleaved(
                    txt_tokens[:, txt_i:txt_i+1],
                    spk_ids[:,  txt_i:txt_i+1],
                    start_pos
                )
                dur_preds.append(dur_pred)
                start_pos = start_pos + 2
                if self.config.use_hard_pos_encoding:
                    timestep = timestep + 1

            dur_pred = torch.cat(dur_preds, dim=1)
            return dur_pred

        elif self.config.modeling_type == 'ar':
            def forward_ar(dur_token, start_pos):
                dur_embeds = self.forward_dur_embed(dur_token)
                if self.config.use_hard_pos_encoding:
                    pos_embed = self.hard_pos_embed(timestep=timestep, out_dtype=dur_embeds.dtype, device=dur_embeds.device)[None, None, :]
                    dur_embeds = dur_embeds + pos_embed
                if self.config.is_seq2seq:
                    x = self.lm.decode(
                        decoder_x=dur_embeds, decoder_padding_mask=None, enc_out=enc_out, encoder_padding_mask=None,
                        start_pos=start_pos, use_cache=True
                    )
                else:
                    x = self.lm(dur_embeds, attn_mask=None, start_pos=start_pos, use_cache=True)
                x = self.lm_head(x)
                dur_pred = sample_dur(x, topk, temperature, stochastic_round, self.config.loss_type)
                return dur_pred

            if dur_tokens is not None:
                txt_embeds = self.txt_embed(txt_tokens) + self.spk_embed(spk_ids)
                dur_embeds = self.forward_dur_embed(dur_tokens)
                if self.config.use_hard_pos_encoding:
                    txt_embeds = txt_embeds + self.hard_pos_embed(
                        torch.arange(txt_embeds.shape[1], device=txt_embeds.device), offset=timestep,
                        out_dtype=txt_embeds.dtype, device=txt_embeds.device
                    )[None, ...]
                    dur_embeds = dur_embeds + self.hard_pos_embed(
                        torch.arange(dur_embeds.shape[1], device=dur_embeds.device),
                        out_dtype=dur_embeds.dtype, device=dur_embeds.device
                    )[None, ...]
                if self.config.use_sep:
                    prefix = torch.cat([txt_embeds, self.sep_token, dur_embeds], dim=1)
                else:
                    prefix = torch.cat([txt_embeds, dur_embeds], dim=1)
                timestep = dur_embeds.shape[1]
            else:
                txt_embeds = self.txt_embed(txt_tokens) + self.spk_embed(spk_ids)
                if self.config.use_hard_pos_encoding:
                    txt_embeds = txt_embeds + self.hard_pos_embed(
                        torch.arange(txt_embeds.shape[1], device=txt_embeds.device), offset=timestep,
                        out_dtype=txt_embeds.dtype, device=txt_embeds.device
                    )[None, ...]
                prefix = txt_embeds
                if self.config.use_sep:
                    prefix = torch.cat([prefix, self.sep_token], dim=1)
                timestep = 0

            if self.config.is_seq2seq:
                x = self.lm.decode(
                    decoder_x=prefix, decoder_padding_mask=None, enc_out=enc_out,
                    encoder_padding_mask=None, start_pos=start_pos, use_cache=True
                )
            else:
                x = self.lm(prefix, attn_mask=None, start_pos=start_pos, use_cache=True)

            start_pos = start_pos + prefix.shape[1]
            dur_pred = sample_dur(self.lm_head(x[:, -1:]), topk, temperature, stochastic_round, self.config.loss_type)

            dur_preds = [dur_pred]
            for txt_i in it:
                dur_pred = forward_ar(dur_pred, start_pos)
                dur_preds.append(dur_pred)
                start_pos = start_pos + 1
                if self.config.use_hard_pos_encoding:
                    timestep = timestep + 1

            dur_pred = torch.cat(dur_preds, dim=1)
            return dur_pred
