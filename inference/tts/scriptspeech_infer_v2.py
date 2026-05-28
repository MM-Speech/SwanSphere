import os
import random
import re
import tempfile
from datetime import datetime
import collections
import collections.abc

import numpy as np

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from attrdict import AttrDict

import yaml
from argparse import ArgumentParser

import torch
import soundfile as sf
import librosa

from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams

from modules.asr.sensevoice.sensevoice_api import build_asr_model, run_asr_model
from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin, SemanticLMBuildModelMixin, build_vae, \
    build_audio_tokenizer

cfg_weight = None
infer_step = 100
extend_dur = 0
vad_len = 0

class ScriptSpeechInfer(DiTBuildModelMixin, SemanticLMBuildModelMixin):
    def __init__(self, device, dit_ckpt, lm_ckpt):
        self.device = device
        self.build_model(dit_ckpt, lm_ckpt)

    def build_model(self, dit_ckpt, lm_ckpt):
        self.asr_model = build_asr_model(self.device)

        set_hparams(config=os.path.join(dit_ckpt, 'config.yaml'), print_hparams=False, global_hparams=True)
        self.config = AttrDict(hparams)
        self.lm_hparams = set_hparams(config=os.path.join(lm_ckpt, 'config.yaml'), print_hparams=False,
                                      global_hparams=False)
        self.lm_hparams['gradient_checkpointing'] = False

        self.vae, self.hp_vae = build_vae(hparams.get('vae_ckpt'))
        self.vae.to(self.device)

        self.audio_token_feature_extractor, self.audio_tokenizer, self.audio_vocab_size = build_audio_tokenizer(
            hparams.get('audio_tokenizer', 'glm4v'))
        self.audio_tokenizer.to(self.device)

        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()
        self.lm_text_tokenizer, self.lm_vocab_size = self.build_lm_text_tokenizer()

        self.lm = self.build_lm(self.lm_hparams)
        load_ckpt(self.lm, lm_ckpt, 'lm', strict=True)
        self.eos_idx = self.lm_text_tokenizer.encode('<|endoftext|>')[0]
        self.speech_start_token = self.lm_text_tokenizer.encode('<SpeechToken_0>')[0]
        self.lm.eval()
        self.lm.to(self.device)

        self.dit = self.build_dit(hparams)
        load_ckpt(self.dit, dit_ckpt, 'dit', strict=True)
        self.dit.eval()
        self.dit.to(self.device)

        if hparams.get('load_sd_text_encoder', False):
            if hparams.get('model_size', 'base') == 'seedance_7b':
                self.build_sd_text_encoder(hparams['text'])
                self.sd_text_encoder.eval()
                self.sd_text_encoder.to(self.device)
            elif hparams.get('model_size', 'base') == 'goku_2':
                self.build_goku_text_encoder(hparams)
                self.goku_text_encoder.eval()
                self.goku_text_encoder.to(self.device)

    def generate_mask(self, input_tokens, start_token=10, end_token=11):
        batch_size, seq_len = input_tokens.shape
        mask = torch.zeros_like(input_tokens, dtype=torch.bool)

        for i in range(batch_size):
            indices = (input_tokens[i] == start_token).nonzero(as_tuple=True)[0]
            for idx in indices:
                # 找到 start_token 后面最近的 end_token
                end_idx = (input_tokens[i, idx + 1:] == end_token).nonzero(as_tuple=True)[0]
                if len(end_idx) > 0:
                    j = idx + 1 + end_idx[0].item()
                    mask[i, idx + 1:j] = 1
        return mask

    @torch.no_grad()
    def run_sd_text_encoder(self, captions: list):
        special_tokens = self.config.text.special_tokens
        token0 = special_tokens[0].token
        token1 = special_tokens[1].token
        captions = [re.sub(r'<W>(.*?)</W>', rf"{token0}\1{token1}", cur_t) for cur_t in captions]
        captions_out = self.sd_text_encoder(captions, special_tokens)
        captions_cmask = self.generate_mask(captions_out.input_token_ids,
                                            start_token=special_tokens[0].token_id,
                                            end_token=special_tokens[1].token_id)
        return captions_out, captions_cmask

    def run_goku_text_encoder(self, captions: list):
        special_token_ids = self.goku_special_token_ids
        inputs = self.goku_tokenizer(
            captions,
            max_length=hparams['text_max_token_length'],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_masks = inputs.input_ids.cuda(), inputs.attention_mask.cuda()
        encoder_hidden_states = self.goku_text_encoder(
            input_ids, return_dict=False,
            attention_mask=attention_masks,
        )[0]
        captions_cmask = self.generate_mask(input_ids,
                                            start_token=special_token_ids[0],
                                            end_token=special_token_ids[1])
        return encoder_hidden_states, captions_cmask, attention_masks

    @torch.inference_mode()
    def forward(self, text, ref_audio=None, ref_text=None, local_prompt=None, global_prompt=None, cfg_w=None,
                extend_len=0.0, start_time=0.2, end_time=0.2, num_step=100):
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        if ref_audio is not None:
            ref_wav = torch.from_numpy(np.concatenate([ref_audio, np.zeros(0, dtype=np.float16)]))[None, :].to(
                self.device)
            ref_wav_lens = torch.LongTensor([ref_wav.shape[1] // fm_wav * fm_wav]).to(self.device)
            ref_wav = ref_wav[:, :ref_wav_lens[0]]

            if ref_text is None:
                with tempfile.TemporaryDirectory(dir='/dev/shm/') as temp_dir:
                    temp_path = os.path.join(temp_dir, 'audio.wav')
                    sf.write(temp_path, ref_audio, 24000, 'PCM_16')
                    asr_result = run_asr_model(temp_path, self.asr_model)
                    print('asr_result', asr_result)
                    ref_text = asr_result['text']
                    ref_text = ref_text + '.'
            if hparams.get('audio_tokenizer', 'glm4v') == 'glm4v':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    from modules.tts.semantic_encoders.glm4_tokenizer.call_utils import extract_speech_token_v2
                    ref_semantic_tokens, ref_semantic_mask = extract_speech_token_v2(
                        self.audio_tokenizer, self.audio_token_feature_extractor,
                        wavs=ref_wav, sample_rate=hparams['audio_sample_rate'], wav_lens=ref_wav_lens,
                        device=self.device
                    )
                    ref_semantic_tokens = ref_semantic_tokens.clone().detach()  # [1, T]
            lm_input_tokens = \
                self.lm_text_tokenizer('<BOT>' + ref_text + text + '<BOS>', padding=True, return_tensors='pt')[
                    'input_ids'].to(self.device)
            lm_input_tokens = torch.cat([lm_input_tokens, ref_semantic_tokens + self.speech_start_token], dim=1)
            if cfg_w is None:
                cfg_w = [2, 2, 2, 5]
        else:
            ref_text = None
            ref_wav = None
            ref_semantic_tokens = None
            lm_input_tokens = \
                self.lm_text_tokenizer('<BOT>' + text + '<BOS>', padding=True, return_tensors='pt')[
                    'input_ids'].to(self.device)
            if cfg_w is None:
                cfg_w = [5, 2, 2, 0]
        print('| CFG: ', cfg_w)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            lm_output_tokens = self.lm.generate(lm_input_tokens, max_new_tokens=2048, do_sample=True, top_p=0.99,
                                                top_k=20, repetition_penalty=1.1, eos_token_id=self.eos_idx)
        print('[SEMANTIC OUTPUT]',
              self.lm_text_tokenizer.decode(list(lm_output_tokens[:, lm_input_tokens.size(1):].cpu().numpy()[0])))
        semantic_tokens = lm_output_tokens[:, lm_input_tokens.size(1):-1] - self.speech_start_token
        if ref_semantic_tokens is not None:
            semantic_tokens = torch.cat([ref_semantic_tokens, semantic_tokens], dim=1)
        tgt_len = semantic_tokens.shape[1] * 2 + int(extend_len * 24000 / self.hp_vae['hop_size'] / self.hp_vae['vae_stride']) # includ ref

        # caption
        caption_emb, caption_lens = None, None
        if global_prompt is not None or local_prompt is not None:
            caption = ''
            if global_prompt is not None:
                caption = caption + global_prompt
            if ref_text is not None:
                caption = caption + f'<W>{ref_text}</W>As the person speaks.'
            if local_prompt is not None:
                caption = caption + local_prompt
            print('| caption:', caption)
            if 'seedance' in self.config.model_size:
                sd_text_output, captions_cmask = self.run_sd_text_encoder([caption])
                text_embs = sd_text_output.embeddings * sd_text_output.masks[:, :, None]
                caption_lens = sd_text_output.masks.sum(-1)
            elif 'goku' in self.config.model_size:
                gk_text_embs, captions_cmask, text_att_mask = self.run_goku_text_encoder([caption])
                text_embs = gk_text_embs * text_att_mask[..., None]
                caption_lens = text_att_mask.sum(-1)
            caption_emb = torch.cat(
                [text_embs, captions_cmask[:, :, None]], -1)

        if ref_text is not None:
            with torch.inference_mode():
                lat_ctx = self.vae.encode_latent(ref_wav)
                ctx_mask = torch.ones_like(lat_ctx[:, :, 0:1])

                lat = torch.nn.functional.pad(
                    lat_ctx, (0, 0, 0, tgt_len - lat_ctx.size(1)), mode='constant', value=0)
                ctx_mask = torch.nn.functional.pad(
                    ctx_mask, (0, 0, 0, tgt_len - ctx_mask.size(1)), mode='constant', value=0)
                text_inputs = self.dit_text_tokenizer(ref_text + text, padding=True, return_tensors='pt').to(
                    self.device)
        else:
            lat_ctx = torch.zeros(1, 0, 32).to(self.device)
            lat = torch.zeros(1, tgt_len, 32).to(self.device)
            ctx_mask = torch.zeros(1, tgt_len, 1).to(self.device)
            text_inputs = self.dit_text_tokenizer(text, padding=True, return_tensors='pt').to(
                self.device)
        txt_tokens = text_inputs['input_ids']
        txt_mask = text_inputs['attention_mask'].bool()
        txt_tokens[~txt_mask] = self.cfg_mask_text_token

        vad_mask = torch.zeros_like(lat[:, :, :1])
        vad_mask[:, int(start_time * 25):-int(end_time * 25)] = 1.0
        vad_mask = torch.cat([vad_mask] * 5, dim=0)

        txt_tokens = torch.cat([
            txt_tokens,
            txt_tokens,
            torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
            torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
            torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
        ], dim=0)
        txt_mask = torch.cat([txt_mask] * 5, dim=0)

        if caption_emb is not None:
            caption_emb = torch.cat([
                caption_emb,
                torch.zeros_like(caption_emb),
                caption_emb,
                torch.zeros_like(caption_emb),
                torch.zeros_like(caption_emb),
            ], dim=0)
            caption_lens = torch.cat([caption_lens] * 5, dim=0).to(torch.long)

        lat = torch.cat([
            lat,
            torch.zeros_like(lat),
            torch.zeros_like(lat),
            lat,
            torch.zeros_like(lat)
        ], dim=0)
        ctx_mask = torch.cat([ctx_mask] * 5, dim=0)

        inputs = {
            'txt_tokens': txt_tokens if not hparams.get('drop_xt', False) else None,
            'txt_mask': txt_mask,
            'ctx_mask': ctx_mask,
            'lat_ctx': lat,
            'semantic_tokens': semantic_tokens if not hparams.get('drop_st', False) else None,
            "caption_emb": caption_emb,
            "caption_lens": caption_lens,  # B
            'vad_mask': vad_mask
        }
        global cfg_weight, infer_step, extend_dur, vad_len
        cfg_weight, infer_step, extend_dur, vad_len = cfg_w, num_step, extend_len, [start_time, end_time]

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            x = self.dit.inference(inputs, timesteps=num_step, seq_cfg_w=cfg_w)
            x[:, :lat_ctx.shape[1]] = lat_ctx
            wav_pred = self.vae.decode(x)[0, 0].to(torch.float32)

            hop_size = self.hp_vae['hop_size']
            vae_stride = self.hp_vae['vae_stride']
            # Trim prompt wav
            wav_pred = wav_pred[lat_ctx.size(1) * vae_stride * hop_size:]
            # clamp the maximum value
            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())

            wav_pred = wav_pred.cpu().numpy()

        return wav_pred

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    from inference.utils.upload_tos_utils import main as upload_tos_html
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv

        load_dotenv('.env.local')
    kill_void()
    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--dit_ckpt", help="Path to model", type=str,
                        default='checkpoints/250622_scriptspeech_dit_singlespk_01')
    parser.add_argument("--lm_ckpt", help="Path to model", type=str, default='checkpoints/250709_ss_lm_singlelocal')
    args = parser.parse_args()
    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dit_ckpt = args.dit_ckpt
    lm_ckpt = args.lm_ckpt
    infer_ins = ScriptSpeechInfer('cuda', dit_ckpt=dit_ckpt, lm_ckpt=lm_ckpt)
    out_path = f'{cfg["out_path"]}/{os.path.basename(lm_ckpt)}_{os.path.basename(dit_ckpt)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(out_path, exist_ok=True)

    for idx, sample in enumerate(cfg['samples']):
        text = sample['text']
        set_seed(len(text) + idx) # seed
        if 'prompt_audio' in sample:
            audio = sample['prompt_audio']
            audio, _ = librosa.load(audio, sr=24000)
        else:
            audio = None
        local_prompt = None
        global_prompt = None
        if hparams.get('load_sd_text_encoder', False):
            if hparams.get('use_local', False):
                local_prompt = sample['local_prompt']
            if hparams.get('use_global', False):
                global_prompt = sample['global_prompt']
        wav = infer_ins.forward(text, ref_audio=audio, local_prompt=local_prompt, global_prompt=global_prompt,
                                cfg_w=cfg.get('cfg_w', None),
                                extend_len=cfg.get('extend_len', 0),
                                num_step=cfg.get('num_step', 100),
                                start_time=cfg.get('vad_len', 0.2), end_time=cfg.get('vad_len', 0.2)
                                )
        print(f'save wav at {out_path}/out_{idx}.wav')
        sf.write(f'{out_path}/out_{idx}.wav', wav, 24000, 'PCM_16')
    desc = (f"Inference setting: cfg weight: {cfg_weight}, inference step: {infer_step}, "
            f"extend duration: {extend_dur}, vad length (silence duration at bugin and tail): {vad_len}")
    upload_tos_html(yml=args.config, out_path=out_path, title_name=os.path.basename(out_path), extra_desc=desc)
    # CUDA_VISIBLE_DEVICES=0 python inference/tts/scriptspeech_infer.py

