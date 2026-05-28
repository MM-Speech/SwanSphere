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


def build_dur_model(hparams, text_tokenizer=None, init_pretrained=False, vocab_size=None, padding_idx=None):
    if text_tokenizer is not None:
        vocab_size = len(text_tokenizer)
    model_config = ModelArgs(
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        init_pretrained=init_pretrained
    )
    
    if hparams.get('use_out_bias', False):
        model_config.use_out_bias = True
    model_config.modeling_type = hparams.get('dur_modeling_type', 'ar_interleaved')
    print_once(f"| use dur_modeling_type = {model_config.modeling_type}")
    
    model_config.loss_type = hparams.get('dur_loss_type', 'nll')
    print_once(f"| use dur_loss_type = {model_config.loss_type}")
    
    model_config.dur_max_value = hparams.get('dur_max_value', 128)
    
    model_config.dur_embed_v2 = hparams.get('dur_embed_v2', False)
    model_config.dur_embed_fourier_K = hparams.get('dur_embed_fourier_K', 64)
    if model_config.dur_embed_v2:
        model_config.dur_embed_fourier_scale = (1.0, 2.0, 4.0, 8.0)
    
    model_config.use_sep = hparams.get('dur_use_sep', False)
    model_config.use_hard_pos_encoding = hparams.get('dur_use_hard_pos_encoding', False)
    
    model_config.do_checkpoint = hparams.get('gradient_checkpointing', False)
    
    if hparams.get('backbone', 'llama') == 'llama':
        model_config.is_seq2seq = False
        if hparams.get('model_size', 'base') == 'small':
            model_config.lm_config.n_layers = 12
            model_config.lm_config.n_heads = 12
            model_config.lm_config.dim = 768 
        elif hparams.get('model_size', 'base') == '1b':
            model_config.lm_config.n_layers = 28
            model_config.lm_config.n_heads = 16
            model_config.lm_config.dim = 1536 
            
        model = ConditionedSeq2SeqDurationLM(model_config)
        
    elif hparams.get('backbone', 'llama') == 'llama_txtcond_seq2seq':
        model_config.is_seq2seq = True
        from modules.asr.llama.llama_seq2seq import ModelArgs as Seq2SeqLLaMaModelArgs
        model_config.backbone = 'llama_txtcond_seq2seq'
        model_config.lm_config = Seq2SeqLLaMaModelArgs()
        model_config.cond_dim = hparams.get('cond_dim', 1024)
        model_config.lm_config.enc_n_layers = 4
        model_config.lm_config.dec_n_layers = 20
        if hparams.get('model_size', 'base') == 'small':
            model_config.lm_config.enc_n_layers = 2
            model_config.lm_config.dec_n_layers = 10
            model_config.lm_config.n_heads = 12
            model_config.lm_config.dim = 768 
        elif hparams.get('model_size', 'base') == '1b':
            model_config.lm_config.enc_n_layers = 8
            model_config.lm_config.dec_n_layers = 24
            model_config.lm_config.n_heads = 16
            model_config.lm_config.dim = 1536 
    
        if hparams.get('multispk', False):
            from modules.tts.ar_dur.dur_lm_multi import MultiConditionedSeq2SeqDurationLM
            model = MultiConditionedSeq2SeqDurationLM(model_config)
        else:
            model = ConditionedSeq2SeqDurationLM(model_config)

    return model


@dataclass
class ModelArgs:
    backbone: str = 'llama'
    lm_config: LLaMaModelArgs = field(default_factory=LLaMaModelArgs)
    use_out_bias: bool = False

    vocab_size: int = None
    padding_idx: int = None
    cond_dim: int = None

    loss_type: str = 'nll'  # nll | ce
    use_dur_avg_match_loss: bool = False
    use_dur_var_match_loss: bool = False
    
    dur_embed_v2: bool = False
    dur_embed_fourier_K: int = 64
    dur_embed_fourier_scale: int = 4.0
    dur_max_value: int = 128
    use_sep: bool = False
    use_hard_pos_encoding: bool = False
    

    modeling_type: str = 'ar_interleaved'   # ar | ar_interleaved
    is_seq2seq: bool = True

    init_pretrained: bool = True
    
    do_checkpoint: bool = False


class ContinuousTimeEncoder(nn.Module):
    def __init__(self, dim, K=64, scale=4.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(K) * scale, requires_grad=False)
        self.fourier_proj = nn.Linear(2 * K, dim)
        self.linear_proj = nn.Linear(1, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        # x [B, T]
        z = x[..., None] * self.B   # [B, T, K]
        x1 = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        x1 = self.fourier_proj(x1)
        x = x1 + self.linear_proj(x[..., None])
        x = self.out_proj(F.silu(x))
        return x
    
class ContinuousTimeEncoderV2(nn.Module):
    def __init__(self, dim, K=64,
                 scales=(1.0, 2.0, 4.0, 8.0),  # y_norm 空间的尺度
                 min_value=0.0, max_value=128.0,
                 use_norm=True):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.use_norm = use_norm

        K_per = K // len(scales)
        assert K_per * len(scales) == K, f"{K} {scales}"
        
        Bs = []
        for s in scales:
            B = torch.randn(K_per) * s
            B = B.clamp(min=-3.0*s, max=3.0*s)
            Bs.append(B)
        B = torch.cat(Bs, dim=0)  # [K]
        self.register_buffer("B", B)
        
        self.fourier_proj = nn.Linear(2 * K, dim)
        self.linear_proj = nn.Linear(1, dim)
        self.discrete_proj = nn.Embedding(int(max_value) + 1, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.register_buffer("log1p_max", torch.tensor(math.log1p(max_value), dtype=torch.float32))
        
    def forward(self, x):
        x = torch.expm1(x)
        x = x.clamp(self.min_value, self.max_value).float()

        y = torch.log1p(x)  # [B, T]
        if self.use_norm:
            y = y / self.log1p_max
            
        # RFF
        z = y[..., None] * self.B  # [B, T, K]
        x1 = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        x1 = x1 / math.sqrt(self.B.shape[-1])  # 1/sqrt(K)
        x1 = self.fourier_proj(x1)
        
        x_lin = self.linear_proj(y[..., None])
        x_disc = self.discrete_proj(x.long())
        
        x = x1 + x_lin + x_disc
        x = self.out_proj(F.silu(x))
        
        return x
    
def sample_dur(x, topk=1, temperature=0.1, stochastic_round=True, loss_type='nll'):
    if loss_type == 'nll':
        mu, s_raw = x.chunk(2, dim=-1)
        sigma = F.softplus(s_raw) + 1e-5
        if temperature == 0.0:
            z = mu + 0.5 * (0.2 * sigma) ** 2
        else:
            sigma = sigma.clamp(0.05, 0.4)
            sigma = sigma * temperature
            eps = torch.randn_like(mu).clamp(-2.0, 2.0)
            z = mu + sigma * eps
        dur_pred = torch.expm1(z).clamp_min(0.0)[..., 0]
        if stochastic_round:
            d_floor = torch.floor(dur_pred)
            frac = (dur_pred - d_floor).clamp(0, 1)
            dur_pred = (d_floor + torch.bernoulli(frac)).to(torch.long)
        else:
            dur_pred = torch.round(dur_pred).to(torch.long)
            
    elif loss_type == 'mse':
        dur_pred = torch.expm1(x[..., 0]).clamp_min(0.0)
        if temperature > 0:
            dur_disturb_choice = (torch.rand_like(dur_pred.float()) > 0.5).float()
            dur_disturb_r = 1 + torch.rand_like(dur_pred.float()) * temperature
            dur_pred = dur_pred * dur_disturb_r * dur_disturb_choice + dur_pred / dur_disturb_r * (1 - dur_disturb_choice)
        if stochastic_round:
            d_floor = torch.floor(dur_pred)
            frac = (dur_pred - d_floor).clamp(0, 1)
            dur_pred = (d_floor + torch.bernoulli(frac)).to(torch.long)
        else:
            dur_pred = torch.round(dur_pred).to(torch.long)
            
    elif loss_type == 'ce':
        if topk == 1:
            dur_pred = torch.argmax(x[:, -1], dim=-1, keepdim=True)   # [1, 1]
        else:
            dur_pred = sample(x[:, -1:], topk, 1.0, temperature)
    return dur_pred

class ConditionedSeq2SeqDurationLM(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.txt_embed = nn.Embedding(config.vocab_size, config.lm_config.dim, config.padding_idx)
        
        if config.loss_type in ['nll', 'mse']:
            if config.dur_embed_v2:
                self.dur_embed = ContinuousTimeEncoderV2(config.lm_config.dim, config.dur_embed_fourier_K, config.dur_embed_fourier_scale,
                                                         min_value=0, max_value=config.dur_max_value + 10, use_norm=True)
            else:
                self.dur_embed = ContinuousTimeEncoder(config.lm_config.dim, config.dur_embed_fourier_K, config.dur_embed_fourier_scale)
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
        
    def forward(self, inputs, instruction_finetuning=False):
        config = self.config

        txt_tokens = inputs['merged_ph_tokens']  # [B, T]
        txt_lens = inputs['merged_ph_tokens_len']
        txt_embeds = self.txt_embed(txt_tokens) # [B, T, C]

        dur_tokens = inputs['dur_tokens']   # [B, T]
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
        
        if self.config.modeling_type == 'ar_interleaved':
            x = x[:, 0::2, :]
        elif self.config.modeling_type == 'ar':
            if self.config.use_sep:
                x = remove_prefix(x, txt_lens, txt_lens)
            else:
                x = remove_prefix(x, txt_lens - 1, txt_lens)
            
        x = self.lm_head(x)

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
            
            if self.config.use_dur_avg_match_loss:
                pass
            
        elif config.loss_type == 'mse':
            z = dur_tokens[..., None].log1p()
            mse = F.mse_loss(x, z, reduction='none')
            loss = (mse * loss_mask * loss_w_mask).sum() / loss_mask.sum()
            outputs = {
                'loss': loss,
                'ntokens': (txt_lens * 2).sum()
            }
            
        elif config.loss_type == 'ce':
            
            loss = F.cross_entropy(x.transpose(1, 2), dur_tokens.long(), reduction='none')
            loss_mask = loss_mask[..., 0]
            loss = (loss * loss_mask * loss_w_mask[..., 0]).sum() / loss_mask.sum()
            outputs = {
                'loss': loss,
                'ntokens': (txt_lens * 2).sum()
            }

        return outputs
    
    @torch.no_grad()
    def prefill(self, txt_tokens, dur_tokens, enc_out=None):
        self.reset_kv_cache()
        
        if self.config.modeling_type == 'ar_interleaved':
            txt_embeds = self.txt_embed(txt_tokens)
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
            prefix = self.txt_embed(txt_tokens)
            
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
    def inference(self, txt_tokens, condition=None, dur_tokens=None, start_pos=0, timestep=0,
                  topk=1, temperature=0.1, stochastic_round=True,
                  use_tqdm=True, print_candidates=False):        
        if self.config.is_seq2seq:
            if condition is not None:
                enc_out = self.lm.encode(self.cond_proj(condition), encoder_padding_mask=torch.ones_like(condition)[..., 0])
            else:
                enc_out = torch.zeros((1, 1, self.config.cond_dim)).to(dtype=self.precision, device=txt_tokens.device)
        
        tgt_len = txt_tokens.shape[1]
        if dur_tokens is not None:
            if self.config.modeling_type == 'ar_interleaved':
                assert dur_tokens.shape[1] <= txt_tokens.shape[1]
                prefix_tokens = txt_tokens[:, :dur_tokens.shape[1]]
                txt_tokens = txt_tokens[:, dur_tokens.shape[1]:]
                start_pos = self.prefill(prefix_tokens, dur_tokens, enc_out)
                if self.config.use_hard_pos_encoding:
                    timestep = prefix_tokens.shape[1]
                
            elif self.config.modeling_type == 'ar':
                if start_pos == 0:
                    assert dur_tokens.shape[1] <= txt_tokens.shape[1]
                    tgt_len = txt_tokens.shape[1] - dur_tokens.shape[1]
                tgt_len = tgt_len - 1
                if self.config.use_hard_pos_encoding:
                    timestep = dur_tokens.shape[1]
                
        if use_tqdm:
            from tqdm import tqdm
            it = tqdm(range(tgt_len), desc='| Generating Dur')
        else:
            it = range(tgt_len)
            
        if self.config.modeling_type == 'ar_interleaved':
            
            def forward_ar_interleaved(txt_token, start_pos):
                txt_embeds = self.txt_embed(txt_token)
                if self.config.use_hard_pos_encoding:
                    pos_embed = self.hard_pos_embed(timestep=timestep, out_dtype=txt_embeds.dtype, device=txt_embeds.device)[None, None, :]
                    txt_embeds = txt_embeds + pos_embed
                if self.config.is_seq2seq:
                    x = self.lm.decode(
                        decoder_x=txt_embeds, decoder_padding_mask=None, enc_out=enc_out, encoder_padding_mask=None, start_pos=start_pos, use_cache=True
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
                        decoder_x=dur_embeds, decoder_padding_mask=None, enc_out=enc_out, encoder_padding_mask=None, start_pos=start_pos+1, use_cache=True
                    )
                else:
                    _ = self.lm(dur_embeds, attn_mask=None, start_pos=start_pos+1, use_cache=True)

                return dur_pred
                    
            dur_preds = []
            for txt_i in it:
                dur_pred = forward_ar_interleaved(txt_tokens[:, txt_i:txt_i+1], start_pos)
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
                        decoder_x=dur_embeds, decoder_padding_mask=None, enc_out=enc_out, encoder_padding_mask=None, start_pos=start_pos, use_cache=True
                    )
                else:
                    x = self.lm(dur_embeds, attn_mask=None, start_pos=start_pos, use_cache=True)
                x = self.lm_head(x)
                dur_pred = sample_dur(x, topk, temperature, stochastic_round, self.config.loss_type)
                return dur_pred
            
            if dur_tokens is not None:
                txt_embeds = self.txt_embed(txt_tokens)
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
                txt_embeds = self.txt_embed(txt_tokens)
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