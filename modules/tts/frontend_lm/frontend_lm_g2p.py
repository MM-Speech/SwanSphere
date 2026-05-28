from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from modules.tts.frontend_lm.llama import LLaMa


@dataclass
class LLAMA_Config:
    encoder_dim: int = 1024
    encoder_n_layers: int = 8
    encoder_n_heads: int = 8
    encoder_n_kv_heads: Optional[int] = None
    mlp_extend: float = None
    out_vocab_size: int = 44810
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 16384
    dropout: float = 0.0
    ffn_dim_multiplier: Optional[float] = None
    use_causal_attn: bool = True

    ph_vocab_size: int = 0  # defined later by tokenizer
    acoustic_start: int = 0  # defined later by tokenizer
    phone_timestamp_start: int = 0  # defined later by tokenizer
    phone_timestamp_end: int = 0  # defined later by tokenizer
    bpe_start: int = 0  # defined later by tokenizer
    bpe_pad: int = 32005
    asr_eos: int = 0

    max_bn_len: int = 4000
    use_qk_norm: bool = False
    use_cache: bool = False
    max_cache_batch_size: int = 10
    n_mels: int = 80


def add_prefix(seq_1, seq_1_len, seq_2, seq_2_len, dim='BTC', value=0):
    if dim == 'BTC':
        B, T, C, device = seq_1.size(0), (seq_1_len+seq_2_len).max(), seq_1.shape[-1], seq_1.device
        seq_merged = torch.full([B, T, C], value, device=device, dtype=seq_2.dtype)
    elif dim == 'BT':
        B, T, device = seq_1.size(0), (seq_1_len+seq_2_len).max(), seq_1.device
        seq_merged = torch.full([B, T], value, device=device, dtype=seq_2.dtype)
    else:
        raise 'Error, not implemented'

    """ Assign seq 1"""
    T_text = seq_1.shape[1]
    T_feat = seq_2.shape[1]
    indics_x = torch.arange(T, device=device)[None, :]
    mask_x = (indics_x < seq_1_len[:, None]) & (indics_x < T_text)
    mask_text = (
        torch.arange(seq_1.shape[1], device=device)[None, :]
        < seq_1_len[:, None]
    )
    seq_merged[mask_x] = seq_1[mask_text].to(dtype=seq_merged.dtype)

    """ Assign seq 2"""
    mask_x = (
        (seq_1_len[:, None] <= indics_x)
        & (indics_x < (seq_1_len + seq_2_len)[:, None])
        & (indics_x - seq_1_len[:, None] < T_feat)
    )
    mask_noisy = (
        torch.arange(T_feat, device=device)[None, :] < seq_2_len[:, None]
    )
    seq_merged[mask_x] = seq_2[mask_noisy].to(dtype=seq_merged.dtype)
    return seq_merged

def remove_prefix(merged_out, prefix_lens, output_lens, dim='BTC'):
    B, device = merged_out.size(0), merged_out.device

    seq_output = torch.zeros(B, output_lens.max(), merged_out.size(-1), device=device)

    T0 = output_lens.max()
    T1 = merged_out.size(1)
    indics0 = torch.arange(T0, device=device)[None, :]
    indics1 = torch.arange(T1, device=device)[None, :]

    mask0 = (indics0 < output_lens[:, None]) & (
        (prefix_lens[:, None] + indics0) < T1
    )
    mask1 = (prefix_lens[:, None] <= indics1) & (
        indics1 < (prefix_lens[:, None] + output_lens[:, None])
    )

    seq_output[mask0] = merged_out[mask1].to(dtype=seq_output.dtype)
    return seq_output


class Frontend_LLAMA(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
                
        # Token projection
        # The length of the llama2 token_dict is 32005 (0~32004), 32005 is pad token, 32006 is bpe start token (mode 1), 32007 is bpe start token (mode 2)
        self.bpe_encoder = nn.Embedding(32200, params.encoder_dim, padding_idx=self.params.bpe_pad)
        self.ph_encoder = nn.Embedding(800, params.encoder_dim, padding_idx=797)

        # Decoder
        self.encoder = LLaMa(params)
        self.postnet = nn.Linear(params.encoder_dim, 800, bias=False)
        
    def forward(self, bpe_tokens, bpe_lengths, tokens, token_lengths):
        """ Stage 2 """
        """ Add prefix """
        bpe_tokens = bpe_tokens.clamp(min=0, max=32199)
        tokens = tokens.clamp(min=0, max=799)
        
        x = add_prefix(self.bpe_encoder(bpe_tokens), bpe_lengths, self.ph_encoder(tokens), token_lengths, dim='BTC', value=0).contiguous()
        attn_lengths =  bpe_lengths + token_lengths
        prefix_lens, output_lens = bpe_lengths, token_lengths

        """ Forward llama """
        attn_mask = self.sequence_mask(attn_lengths, device=x.device) > 0
        encoder_out = self.encoder(x, attn_mask)
        seq_output = remove_prefix(encoder_out, prefix_lens, output_lens, dim='BTC').contiguous()
        logits = self.postnet(seq_output).transpose(1, 2).contiguous()

        target = tokens
        ce_loss = F.cross_entropy(logits[:, :, :-1], target[:, 1:], ignore_index=797)
        return ce_loss

    def sequence_mask(self, seq_lens, max_len=None, device='cpu'):
        b = seq_lens.shape[0]
        if max_len is None:
            max_len = seq_lens.max()
        mask = torch.arange(max_len).unsqueeze(0).to(device)  # [1, t]
        mask = mask < (seq_lens.unsqueeze(1))  # [1, t] + [b, 1] = [b, t]
        mask = mask.float()
        return mask


class Frontend_G2P_Interface(Frontend_LLAMA):
    def g2p(self, bpe_tokens, bpe_lengths, ph_tokens, ph_lengths, max_decode_steps=1000):
        txt_embed = add_prefix(self.bpe_encoder(bpe_tokens), bpe_lengths, self.ph_encoder(ph_tokens), ph_lengths, dim='BTC').contiguous()
        txt_len = bpe_lengths + ph_lengths
        """ Forward llama """
        attn_mask = self.sequence_mask(txt_len, device=txt_embed.device) > 0
        _ = self.encoder(txt_embed, attn_mask, start_pos=0, use_cache=True)

        def forward_(token, start_pos):
            x = self.ph_encoder(token)
            encoder_out = self.encoder(x, attn_mask=None, start_pos=start_pos, use_cache=True)
            logit = self.postnet(encoder_out)
            return logit

        start_pos = (txt_len - 1).int()
        for step in range(max_decode_steps):
            logits = forward_(ph_tokens[:, -1:], start_pos)
            # Use given phone sequence
            token_pred = torch.argmax(F.softmax(logits[:, -1], dim=-1), 1)
          
            if token_pred == 799:
                print("Finished!")
                break

            token_pred = token_pred[None, :].repeat(bpe_tokens.size(0), 1)
            ph_tokens = torch.cat((ph_tokens, token_pred), dim=1)
            start_pos = start_pos + 1
        
        return ph_tokens[:, 1:]