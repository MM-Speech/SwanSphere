import torch
import torch.nn as nn
from transformers.models.qwen3 import Qwen3ForCausalLM
from einops import rearrange
import numpy as np

from utils.nn.seq_utils import add_prefix, sequence_mask, remove_prefix, add_prefix_2d, add_prefix_nd

class Qwen3TTSMimiModel(nn.Module):
    def __init__(
            self, 
            backbone: Qwen3ForCausalLM, 
            decoder: Qwen3ForCausalLM,
            text_acoustic_token: int = None,
            audio_codebook_size=2048,
            acoustic_n_quantizers=31,
            semantic_start_idx: int = None,
            eos_idx=-1,
            backbone_padding_idx=-100,
            decoder_padding_idx=-100,
            decoder_frame_ratio=1/16,
            # acoustic_start_idx: int = None
            decoder_proj_bias=True,
            acoustic_encoder_zero_init=True
        ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

        self.acoustic_n_quantizers = acoustic_n_quantizers
        self.text_acoustic_token = text_acoustic_token if text_acoustic_token is not None else audio_codebook_size * acoustic_n_quantizers
        self.acoustic_embedding = nn.Embedding(audio_codebook_size * acoustic_n_quantizers + 2, 512)
        self.acoustic_encoder = nn.Linear(acoustic_n_quantizers * 512, self.backbone.config.hidden_size)
        if acoustic_encoder_zero_init:
            torch.nn.init.zeros_(self.acoustic_encoder.weight)
            torch.nn.init.zeros_(self.acoustic_encoder.bias)

        self.decoder_proj = nn.Linear(self.backbone.config.hidden_size, 512, bias=decoder_proj_bias)

        self.semantic_start_idx = semantic_start_idx
        self.eos_idx = eos_idx
        self.backbone_padding_idx = backbone_padding_idx
        self.decoder_padding_idx = decoder_padding_idx
        self.decoder_frame_ratio = decoder_frame_ratio

    def forward(self, text_tokens, audio_tokens, text_mask, audio_mask):
        # text_tokens  [B, Tt]
        # audio_tokens [B, Ta, N]

        bsz, text_len = text_tokens.shape
        _, audio_len, nq = audio_tokens.shape
        device = text_tokens.device

        text_lens = text_mask.sum(1)
        audio_lens = audio_mask.sum(1)
        input_tokens = add_prefix(text_tokens, text_lens, audio_tokens[:, :, 0], audio_lens, padding_value=self.backbone_padding_idx)

        text_acoustic_tokens = torch.full((bsz, text_len, self.acoustic_n_quantizers), self.text_acoustic_token).to(text_tokens)   # [B, Tt, N-1]
        acoustic_tokens = audio_tokens[:, :, 1:]  # [B, Ta, N-1]
        acoustic_tokens = add_prefix_nd(text_acoustic_tokens, text_lens, acoustic_tokens, audio_lens, padding_value=self.decoder_padding_idx)   # [B, Tt+Ta, N-1]

        # Assign <EOS> token
        input_lens = text_lens + audio_lens
        input_tokens = nn.functional.pad(input_tokens, (0, 1), value=self.backbone_padding_idx)
        input_tokens[torch.arange(input_tokens.shape[0]).to(input_tokens.device), input_lens] = self.eos_idx
        input_lens += 1

        # compute embeds
        input_embeds = self.backbone.model.embed_tokens(input_tokens)     # [B, T, C]
        acoustic_embeds = self.acoustic_embedding(acoustic_tokens)        # [B, T, N-1, C]
        acoustic_embeds = self.acoustic_encoder(acoustic_embeds.flatten(start_dim=2))   # [B, T, (N-1)*C] -> [B, T, C]
        input_embeds[:, :-1] = input_embeds[:, :-1] + acoustic_embeds   # avoid eos
        attn_mask = sequence_mask(input_lens, dtype=input_embeds.dtype)

        backbone_outputs = self.backbone(
            input_ids=None,
            attention_mask=attn_mask,
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            use_cache=False
        )

        backbone_logits = backbone_outputs.logits[:, :-1]
        backbone_labels = input_tokens[:, 1:]

        # decoder
        last_hidden_states = backbone_outputs.hidden_states[-1]     # [B, T, C]
        last_hidden_states = remove_prefix(last_hidden_states, text_lens - 1, audio_lens + 2)[:, :-2]   # [B, Ta, C]
        last_hidden_states = last_hidden_states.flatten(0, 1).unsqueeze(1)   # [BxTa, 1, C]
        last_hidden_states = self.decoder_proj(last_hidden_states)          # [BxTa, 1, C]
        decoder_input_tokens = audio_tokens[:, :, 1:].flatten(0, 1)    # [BxTa, N-1]
        decoder_input_embeds = self.decoder.model.embed_tokens(decoder_input_tokens)    # [BxTa, N-1, C]
        decoder_input_embeds = torch.cat([last_hidden_states, decoder_input_embeds], dim=1)     # [BxTa, N, C]
        decoder_labels = decoder_input_tokens   # [BxTa, N-1]

        # remove padding
        audio_mask = audio_mask.flatten()     # [BxTa]
        decoder_input_embeds = decoder_input_embeds[audio_mask]     # [Td, N, C]
        decoder_labels = decoder_labels[audio_mask]                 # [Td, N-1]

        # computation amortize
        decoder_frame_mask = torch.rand_like(decoder_input_embeds[:, 0, 0])     # [Td]
        decoder_frame_mask = decoder_frame_mask < self.decoder_frame_ratio
        if not decoder_frame_mask.any():
            decoder_frame_mask[0] = True
        decoder_input_embeds = decoder_input_embeds[decoder_frame_mask]
        decoder_labels = decoder_labels[decoder_frame_mask]
        decoder_attn_mask = torch.ones_like(decoder_input_embeds[:, :, 0])

        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=decoder_attn_mask,
            inputs_embeds=decoder_input_embeds,
            use_cache=False
        )

        decoder_logits = decoder_outputs.logits[:, :-1]

        return {
            'backbone_logits': backbone_logits,
            'backbone_labels': backbone_labels,
            'decoder_logits': decoder_logits,
            'decoder_labels': decoder_labels,
        }

    def generate(self, text_tokens, audio_tokens=None, max_new_tokens=256, topk=1, temperature=1.0):
        bsz, text_len = text_tokens.shape
        device = text_tokens.device

        if audio_tokens is not None:
            input_tokens = torch.cat([text_tokens, audio_tokens[:, :, 0]], dim=1)
            text_acoustic_tokens = torch.full((bsz, text_len, self.acoustic_n_quantizers), self.text_acoustic_token).to(text_tokens)
            acoustic_tokens = audio_tokens[:, :, 1:]  # [B, Ta, N-1]
            acoustic_tokens = torch.cat([text_acoustic_tokens, acoustic_tokens], dim=1)
        else:
            input_tokens = text_tokens
            acoustic_tokens = torch.full((bsz, text_len, self.acoustic_n_quantizers), self.text_acoustic_token).to(text_tokens)   # [B, Tt, N-1]

        # compute embeds
        input_embeds = self.backbone.model.embed_tokens(input_tokens)     # [B, T, C]
        acoustic_embeds = self.acoustic_embedding(acoustic_tokens)        # [B, T, N-1, C]
        acoustic_embeds = self.acoustic_encoder(acoustic_embeds.flatten(start_dim=2))   # [B, T, (N-1)*C] -> [B, T, C]
        input_embeds = input_embeds + acoustic_embeds

        backbone_output_tokens = []
        decoder_output_tokens = []
        backbone_past_key_values = None
        from tqdm import tqdm
        for i in tqdm(range(max_new_tokens)):
            backbone_outputs = self.backbone.forward(
                input_ids=None, inputs_embeds=input_embeds, 
                output_hidden_states=True,
                past_key_values=backbone_past_key_values,
                use_cache=True,
                logits_to_keep=1
            )
            backbone_logtis = backbone_outputs.logits
            backbone_past_key_values = backbone_outputs.past_key_values

            backbone_next_token_logits = backbone_logtis[:, -1, :]      # [1, C]
            backbone_next_token_logits = backbone_next_token_logits[:, self.eos_idx:]
            if topk == 1:
                input_ids = torch.argmax(backbone_next_token_logits, dim=-1, keepdim=True)  # [1, 1]
            else:
                input_ids = sample_topk(backbone_next_token_logits, topk, temperature)   # [1, 1]
            input_ids = input_ids + self.eos_idx

            if input_ids[0].item() == self.eos_idx:
                break
            backbone_output_tokens.append(input_ids)    # [1, 1]

            last_hidden_states = backbone_outputs.hidden_states[-1]
            last_hidden_states = last_hidden_states[:, -1, :]   # [1, C]
            last_hidden_states = self.decoder_proj(last_hidden_states)
            decoder_input_embeds = last_hidden_states[:, None]  # [1, 1, C]
            decoder_attn_mask = torch.ones_like(decoder_input_embeds[:, :, 0])

            decoder_output_tokens_ = []
            decoder_past_key_values = None
            for j in range(self.acoustic_n_quantizers):
                decoder_outputs = self.decoder(
                    input_ids=None, inputs_embeds=decoder_input_embeds,
                    past_key_values=decoder_past_key_values,
                    use_cache=True,
                    logits_to_keep=1
                )
                decoder_logits = decoder_outputs.logits
                decoder_past_key_values = decoder_outputs.past_key_values

                decoder_next_token_logits = decoder_logits[:, -1, :]
                decoder_next_token_logits = decoder_next_token_logits[:, j * 2048: (j+1) * 2048]
                if topk == 1:
                    decoder_input_ids = torch.argmax(decoder_next_token_logits, dim=-1, keepdim=True)   # [1, 1]
                else:
                    decoder_input_ids = sample_topk(decoder_next_token_logits, topk, temperature)
                decoder_input_ids = decoder_input_ids + j * 2048
                decoder_output_tokens_.append(decoder_input_ids)

                decoder_input_embeds = self.decoder.model.embed_tokens(decoder_input_ids)   # [1, 1, C]

            decoder_output_tokens_ = torch.cat(decoder_output_tokens_, dim=-1)[:, None]    # [1, 1, 31]
            decoder_output_tokens.append(decoder_output_tokens_)

            input_embeds = self.backbone.model.embed_tokens(input_ids)  # [1, 1, C]
            acoustic_embeds = self.acoustic_embedding(decoder_output_tokens_)        # [1, 1, 31, C]
            acoustic_embeds = self.acoustic_encoder(acoustic_embeds.flatten(start_dim=2))   # [B, T, (N-1)*C] -> [B, T, C]
            input_embeds = input_embeds + acoustic_embeds   # avoid eos

        backbone_output_tokens = torch.cat(backbone_output_tokens, dim=-1)  # [1, Ta]
        decoder_output_tokens = torch.cat(decoder_output_tokens, dim=1)     # [1, Ta, 31]

        return backbone_output_tokens, decoder_output_tokens


def _multinomial_sample_one_no_sync(probs):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token
