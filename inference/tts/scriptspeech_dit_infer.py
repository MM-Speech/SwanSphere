import os

import torch
import soundfile as sf
import librosa

from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams

from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin

class ScriptSpeechDiTInfer(DiTBuildModelMixin):
    def __init__(self, device, ckpt):
        self.device = device
        self.build_model(ckpt)

    def build_model(self, ckpt):
        set_hparams(config=os.path.join(ckpt, 'config.yaml'), print_hparams=False)
        self._build_model()
        self.vae.to(self.device)
        self.audio_tokenizer.to(self.device)
        load_ckpt(self.dit, ckpt, 'dit', strict=True)
        self.dit.eval()
        self.dit.to(self.device)

    @torch.no_grad()
    def forward(self, text, audio, ref_text=None, ref_audio=None):
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        ref_wav = torch.from_numpy(ref_audio)[None, :].to(self.device)
        ref_wav_lens = torch.LongTensor([ref_wav.shape[1] // fm_wav * fm_wav]).to(self.device)
        ref_wav = ref_wav[:, :ref_wav_lens[0]]

        wav = torch.from_numpy(audio)[None, :].to(self.device)
        wav_lens = torch.LongTensor([wav.shape[1] // fm_wav * fm_wav]).to(self.device)
        wav = wav[:, :wav_lens[0]]

        if hparams.get('audio_tokenizer', 'glm4v') == 'glm4v':
            from modules.tts.semantic_encoders.glm4_tokenizer.call_utils import extract_speech_token_v2
            ref_semantic_tokens, ref_semantic_mask = extract_speech_token_v2(
                self.audio_tokenizer, self.audio_token_feature_extractor, 
                wavs=ref_wav, sample_rate=hparams['audio_sample_rate'], wav_lens=ref_wav_lens, device=self.device
            )
            ref_semantic_tokens = ref_semantic_tokens.clone().detach()  # [1, T]

            semantic_tokens, semantic_mask = extract_speech_token_v2(
                self.audio_tokenizer, self.audio_token_feature_extractor, 
                wavs=wav, sample_rate=hparams['audio_sample_rate'], wav_lens=wav_lens, device=self.device
            )
            semantic_tokens = semantic_tokens.clone().detach()  # [1, T]

        semantic_tokens = torch.cat([ref_semantic_tokens, semantic_tokens], dim=1)
        tgt_len = semantic_tokens.shape[1] * 2  # includ ref
            
        with torch.inference_mode():
            lat_ctx = self.vae.encode_latent(ref_wav)
            ctx_mask = torch.ones_like(lat_ctx[:, :, 0:1])
            lat = torch.nn.functional.pad(
                lat_ctx, (0,0,0,tgt_len-lat_ctx.size(1)), mode='constant', value=0)
            ctx_mask = torch.nn.functional.pad(
                ctx_mask, (0,0,0,tgt_len-ctx_mask.size(1)), mode='constant', value=0)

        ref_text_tokens = self.dit_text_tokenizer(ref_text, return_tensors='pt')['input_ids'].to(self.device)
        text_inputs = self.dit_text_tokenizer(ref_text + text, padding=True, return_tensors='pt').to(self.device)
        txt_tokens = text_inputs['input_ids']
        txt_mask = text_inputs['attention_mask'].bool()
        
        txt_tokens = torch.cat([
            txt_tokens,
            txt_tokens,
            torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
            torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
            torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device),
        ], dim=0)
        semantic_tokens = torch.cat([
            semantic_tokens,
            torch.full(semantic_tokens.size(), self.cfg_mask_audio_token, device=self.device),
            semantic_tokens,
            torch.full(semantic_tokens.size(), self.cfg_mask_audio_token, device=self.device),
            torch.full(semantic_tokens.size(), self.cfg_mask_audio_token, device=self.device),
        ], dim=0)
        lat = torch.cat([
            lat,
            torch.zeros_like(lat),
            torch.zeros_like(lat),
            lat,
            torch.zeros_like(lat)
        ], dim=0)
        ctx_mask = torch.cat([ctx_mask] * 5, dim=0)
        txt_mask = torch.cat([txt_mask] * 5, dim=0)

        inputs = {
            'txt_tokens': txt_tokens,
            'txt_mask': txt_mask,
            'ctx_mask': ctx_mask,
            'lat_ctx': lat,
            'semantic_tokens': semantic_tokens
        }

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            x = self.dit.inference(inputs, timesteps=32, seq_cfg_w=[2, 2, 0, 4])

            x[:, :lat_ctx.shape[1]] = lat_ctx
            wav_pred = self.vae.decode(x)[0,0].to(torch.float32)
            
            hop_size = self.hp_vae['hop_size']
            vae_stride = self.hp_vae['vae_stride']
            # Trim prompt wav
            wav_pred = wav_pred[lat_ctx.size(1)*vae_stride*hop_size:]
            # clamp the maximum value
            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())

            wav_pred = wav_pred.cpu().numpy()

        return wav_pred


if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
    kill_void()

    ckpt = 'checkpoints/250622_scriptspeech_dit_singlespk_01'

    infer_ins = ScriptSpeechDiTInfer('cuda', ckpt=ckpt)

    text1 = '造型方面，其实很有意思。第一次试妆的时候，很不一样。'
    audio1 = '/mnt/bn/sa-ag-data/liruiqi/code/MegaHuman/inference_api/prompt_api/celebrity_liuyifei.wav'
    audio1, _ = librosa.load(audio1, sr=24000)

    text2 = '现在，已经有超过六千万的用户，在土巴兔 A P P 上获得了这份报价。'
    audio2 = '/mnt/bn/sa-ag-data/liruiqi/code/MegaHuman/inference_api/prompt_api/口播_王宏源.wav'
    audio2, _ = librosa.load(audio2, sr=24000)

    wav = infer_ins.forward(text1, audio1, text2, audio2)

    sf.write('infer_out/tts/out.wav', wav, 24000, 'PCM_16')


    # CUDA_VISIBLE_DEVICES=0 python inference/tts/scriptspeech_dit_infer.py

