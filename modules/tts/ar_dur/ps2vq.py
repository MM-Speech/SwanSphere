import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear

from modules.tts.ar_dur.commons.layers import LayerNorm, Embedding
from modules.tts.ar_dur.commons.nar_tts_modules import PosEmb
from modules.tts.ar_dur.commons.rel_transformer import RelTransformerEncoder
from modules.tts.ar_dur.commons.transformer import TransformerDecoderLayer, SinusoidalPositionalEmbedding
from modules.tts.ar_dur.commons.align_ops import expand_states
from modules.tts.ar_dur.commons.global_conv import ConvGlobalStacks
from modules.tts.ar_dur.commons.rel_transformer import RelTransformerEncoder


def fill_with_neg_inf2(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(-1e8).type_as(t)

FS_ENCODERS = {
    'rel_fft': lambda hp, dict_size: RelTransformerEncoder(
        dict_size, hp['hidden_size'], hp['hidden_size'],
        hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'],
        hp['enc_ffn_kernel_size'], hp['dropout'], prenet=hp['enc_prenet'], pre_ln=hp['enc_pre_ln']),
}


class PsdCodePredictor(nn.Module):
    def __init__(self, hparams, dict_size, psd_code_size):
        super().__init__()
        self.hparams = hparams
        self.hidden_size = hidden_size = hparams['hidden_size']
        if hparams['use_spk_id']:
            self.spk_id_proj = Embedding(hparams['num_spk'], self.hidden_size)
        if hparams['use_spk_embed']:
            self.spk_embed_proj = nn.Linear(256, self.hidden_size, bias=True)
        if hparams['use_spk_enc']:
            self.spk_enc = nn.Sequential(
                ConvGlobalStacks(
                    idim=hparams['audio_num_mel_bins'],
                    n_chans=self.hidden_size, odim=hparams['c_spk_enc']),
                nn.Linear(hparams['c_spk_enc'], self.hidden_size)
            )
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, dict_size)
        self.char_embed = FS_ENCODERS[hparams['encoder_type']](hparams, 4000)
        self.char_empty_embed = nn.Embedding(1, self.hidden_size)
        if hparams.get('use_bert_input'):
            self.bert_input_proj = nn.Linear(768, self.hidden_size)
            self.bert_encoder = RelTransformerEncoder(
                0, self.hidden_size, self.hidden_size,
                hparams['ffn_hidden_size'], hparams['num_heads'], hparams['enc_layers'],
                hparams['enc_ffn_kernel_size'], hparams['dropout'],
                prenet=hparams['enc_prenet'], pre_ln=hparams['enc_pre_ln'])
        self.ling_label_embed_layers = nn.ModuleDict()
        for k, s in zip(hparams['ling_labels'], hparams['ling_label_dict_size']):
            self.ling_label_embed_layers[k] = Embedding(s + 3, self.hidden_size, padding_idx=0)
        if hparams['use_ph_pos_embed']:
            self.ph_pos_embed = PosEmb(self.hidden_size)

        self.code_emb = Embedding(psd_code_size + 2, hidden_size, 0)
        self.embed_positions = SinusoidalPositionalEmbedding(hidden_size, 0, init_size=1024)
        dec_num_layers = 4
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(hidden_size, 0.0, kernel_size=3) for _ in
            range(dec_num_layers)
        ])
        self.layer_norm = LayerNorm(hidden_size)
        self.project_out_dim = Linear(hidden_size, psd_code_size + 1, bias=True)

    def forward(self, txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
                prev_psd_code, spk_id, mels_timbre, incremental_state=None, x_ling=None):
        if x_ling is None:
            x_ling = self.forward_ling_encoder(
                txt_tokens, ling_feas, char_tokens, ph2char, bert_embed, spk_id, None, mels_timbre)
        x = self.code_emb(prev_psd_code)

        # run decoder
        if incremental_state is not None:
            positions = self.embed_positions(
                prev_psd_code,
                incremental_state=incremental_state
            )
            x_ling = x_ling[:, x.shape[1] - 1:x.shape[1]]
            x = x[:, -1:]
            positions = positions[:, -1:]
            self_attn_padding_mask = None
        else:
            positions = self.embed_positions(
                prev_psd_code,
                incremental_state=incremental_state
            )
            self_attn_padding_mask = txt_tokens.eq(0).data

        x += positions
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        word_embed = x_ling.transpose(0, 1)
        x = x + word_embed

        for layer in self.layers:
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, attn_logits = layer(
                x,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )

        x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = self.project_out_dim(x)
        return x

    def forward_ling_encoder(
            self, txt_tokens, ling_feas, char_tokens, ph2char, bert_embed, spk_id, spk_embed, mels_timbre):
        ph_tokens = txt_tokens
        hparams = self.hparams
        ph_nonpadding = (ph_tokens > 0).float()[:, :, None]  # [B, T_phone, 1]
        x_spk = self.forward_style_embed(spk_embed, spk_id, mels_timbre)

        # enc_ph
        ph_enc_oembed = sum(
            [self.ling_label_embed_layers[k](ling_feas[k]) for k in hparams['ling_labels']]) \
            if len(hparams['ling_labels']) > 0 else 0
        ph_enc_oembed = ph_enc_oembed + self.ph_pos_embed(
            torch.arange(0, ph_tokens.shape[1])[None,].to(ph_tokens.device))
        ph_enc_oembed = ph_enc_oembed + x_spk
        ph_enc_oembed = ph_enc_oembed * ph_nonpadding
        x_ph = self.encoder(ph_tokens, other_embeds=ph_enc_oembed)

        # enc_char
        if hparams['use_char']:
            char_nonpadding = (char_tokens > 0).float()[:, :, None]
            x_char = self.char_embed(char_tokens)
            if hparams['use_bert_input']:
                x_char = self.bert_encoder(self.bert_input_proj(bert_embed) + x_char)
            empty_char = (ph2char > 100000).long()
            ph2char = ph2char * (1 - empty_char)
            x_char_phlevel = \
                expand_states(x_char * char_nonpadding, ph2char) \
                * (1 - empty_char)[..., None] + \
                self.char_empty_embed(torch.zeros_like(ph_tokens)) * empty_char[..., None]
        else:
            x_char_phlevel = 0
        # x_ling
        x_ling = x_ph + x_char_phlevel
        return x_ling

    def forward_style_embed(self, spk_embed=None, spk_id=None, mel_ref=None):
        # add spk embed
        style_embed = 0
        if self.hparams['use_spk_embed']:
            style_embed = style_embed + self.spk_embed_proj(spk_embed)[:, None, :]
        if self.hparams['use_spk_id']:
            style_embed = style_embed + self.spk_id_proj(spk_id)[:, None, :]
        if self.hparams['use_spk_enc']:
            style_embed = style_embed + self.spk_enc(mel_ref)[:, None, :]
        return style_embed

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(fill_with_neg_inf2(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def infer(self, txt_tokens, ling_feas, char_tokens, ph2char, bert_embed, spk_id, spk_embed, mels_timbre,
              incremental_state=None, prev_vqcodes=None):
        hparams = self.hparams
        if incremental_state is None:
            incremental_state = {}
        x_ling = self.forward_ling_encoder(txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
                                           spk_id, spk_embed, mels_timbre)
        vq_decoded = torch.zeros_like(txt_tokens)
        vq_decoded = F.pad(vq_decoded, [1, 0], value=hparams['nVQ'] + 1)
        decoder_output_hiddens = []
        for step in range(vq_decoded.shape[1] - 1):
            vq_pred = self(txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
                           vq_decoded[:, :step + 1], spk_id, spk_embed, mels_timbre,
                           incremental_state=incremental_state, x_ling=x_ling)
            decoder_output_hiddens.append(vq_pred[:, -1])
            if hparams.get('infer_argmax', True):
                vq_pred = torch.argmax(F.softmax(vq_pred[:, -1], dim=-1), 1)
            else:
                vq_pred = torch.multinomial(F.softmax(vq_pred[:, -1], dim=-1), 1).squeeze(1)
            vq_decoded[:, step + 1] = vq_pred \
                if prev_vqcodes is None or step >= prev_vqcodes.shape[1] else prev_vqcodes[:, step]
        return vq_decoded[:, 1:]
