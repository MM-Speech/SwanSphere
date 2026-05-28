import os
import tempfile
from pathlib import Path

# Monkey patch collections
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import torch
import soundfile as sf
import librosa
from tqdm import tqdm
import numpy as np

from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams

from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin, SemanticLMBuildModelMixin, build_vae, build_audio_tokenizer

class ScriptSpeechInfer(DiTBuildModelMixin, SemanticLMBuildModelMixin):
    def __init__(self, device, dit_ckpt, lm_ckpt):
        self.device = device
        self.build_model(dit_ckpt, lm_ckpt)

    def build_model(self, dit_ckpt, lm_ckpt):
        set_hparams(config=os.path.join(dit_ckpt, 'config.yaml'), print_hparams=False)
        self.lm_hparams = set_hparams(config=os.path.join(lm_ckpt, 'config.yaml'), print_hparams=False, global_hparams=False)
        self.lm_hparams['gradient_checkpointing'] = False

        self.vae, self.hp_vae = build_vae(hparams.get('vae_ckpt'))
        self.vae.to(self.device)

        self.audio_token_feature_extractor, self.audio_tokenizer, self.audio_vocab_size = build_audio_tokenizer(hparams.get('audio_tokenizer', 'glm4v'))
        self.audio_tokenizer.to(self.device)

        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()
        self.lm_text_tokenizer, self.lm_vocab_size = self.build_lm_text_tokenizer()

        self.lm = self.build_lm(self.lm_hparams)
        load_ckpt(self.lm, lm_ckpt, 'lm', strict=True, mmap=True)
        self.eos_idx = self.lm_text_tokenizer.encode('<|endoftext|>')[0]
        self.speech_start_token = self.lm_text_tokenizer.encode('<SpeechToken_0>')[0]
        self.lm.eval()
        self.lm.to(self.device)

        self.dit = self.build_dit(hparams)
        load_ckpt(self.dit, dit_ckpt, 'dit', strict=True, mmap=True)
        self.dit.eval()
        self.dit.to(self.device)


    @torch.no_grad()
    def forward(self, text, ref_audio=None, ref_text=None):
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        ref_wav = torch.from_numpy(ref_audio)[None, :].to(self.device)
        ref_wav_lens = torch.LongTensor([ref_wav.shape[1] // fm_wav * fm_wav]).to(self.device)
        ref_wav = ref_wav[:, :ref_wav_lens[0]]

        if hparams.get('audio_tokenizer', 'glm4v') == 'glm4v':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                from modules.tts.semantic_encoders.glm4_tokenizer.call_utils import extract_speech_token_v2
                ref_semantic_tokens, ref_semantic_mask = extract_speech_token_v2(
                    self.audio_tokenizer, self.audio_token_feature_extractor, 
                    wavs=ref_wav, sample_rate=hparams['audio_sample_rate'], wav_lens=ref_wav_lens, device=self.device
                )
                ref_semantic_tokens = ref_semantic_tokens.clone().detach()  # [1, T]

        lm_input_tokens = self.lm_text_tokenizer('<BOT>' + ref_text + text + '<BOS>', padding=True, return_tensors='pt')['input_ids'].to(self.device)
        lm_input_tokens = torch.cat([lm_input_tokens, ref_semantic_tokens + self.speech_start_token], dim=1)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            lm_output_tokens = self.lm.generate(lm_input_tokens, max_new_tokens=2048, do_sample=True, temperature=0.7, top_p=0.99, top_k=5, repetition_penalty=1.1, eos_token_id=self.eos_idx)
        # print('[SEMANTIC OUTPUT]', self.lm_text_tokenizer.decode(list(lm_output_tokens[:, lm_input_tokens.size(1):].cpu().numpy()[0])))
        semantic_tokens = lm_output_tokens[:, lm_input_tokens.size(1):-1] - self.speech_start_token

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

import os
import tempfile
from pathlib import Path

# Monkey patch collections
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import torch
import soundfile as sf
import librosa
from tqdm import tqdm
import numpy as np

from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams

from inference.tts.scriptspeech_dialogue_infer import ScriptSpeechInfer


if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
    kill_void()

    # dit_ckpt = 'checkpoints/250710_scriptspeech_dit_dialogue_textconcat'
    dit_ckpt = 'checkpoints/250717_scriptspeech_dit_2spk_textconcat'

    # lm_ckpt = 'checkpoints/250711_scriptspeech_semanticlm_dialogue'
    lm_ckpt = 'checkpoints/250717_scriptspeech_semanticlm_2spk'

    infer_ins = ScriptSpeechInfer('cuda', dit_ckpt=dit_ckpt, lm_ckpt=lm_ckpt)

    dialogue_infer = True

    if dialogue_infer:

        ref_audios = [
            ('user/prompts/dzq_enhanced.wav', 'user/prompts/jay_promptvn.wav'),
            ('user/prompts/jialing_promptvn.wav', 'user/prompts/dehua_promptvn.wav'),
            ('user/prompts/dzq_enhanced.wav', 'user/prompts/jay_promptvn.wav'),
            ('user/prompts/jialing_promptvn.wav', 'user/prompts/dehua_promptvn.wav'),
        ]
        ref_texts = [
            ('什么等下等下，我看到一个留言说什么邓紫曦都四十二岁了，你算错了吧！我十年前上我是歌手的时候，二十二岁啊兄弟，十年啊，二十二加十，三十二，懂吗？懂吗？三十二不是四十二，你算错了，谁教你的数学谁是你的数学老师。',
            '对我来讲是一种荣幸但是也是压力蛮大的，不过我觉得是一种，呃，很好的一个挑战。'),
            ('跟观众分享我人生的感悟因为我们都是只活一次，我们也都是第一次活，我们也不知道该怎么活着。',
            '所以我觉得这些成功的电影它都很真诚，而且很有生命力，它就跟当年的那个林浩的那个一模一样。'),
            ('什么等下等下，我看到一个留言说什么邓紫曦都四十二岁了，你算错了吧！我十年前上我是歌手的时候，二十二岁啊兄弟，十年啊，二十二加十，三十二，懂吗？懂吗？三十二不是四十二，你算错了，谁教你的数学谁是你的数学老师。',
            '对我来讲是一种荣幸但是也是压力蛮大的，不过我觉得是一种，呃，很好的一个挑战。'),
            ('跟观众分享我人生的感悟因为我们都是只活一次，我们也都是第一次活，我们也不知道该怎么活着。',
            '所以我觉得这些成功的电影它都很真诚，而且很有生命力，它就跟当年的那个林浩的那个一模一样。')
        ]
        texts = [
            # '<SPK>0</SPK>我的老朋友，你真的很不简单,这段时间悄悄看着你，发现你跟人相处时，总是带着春风般的亲切劲儿，让人打心眼里觉得温暖。你对生活那股子热乎劲儿，谁看了都受感染。每天乐呵呵的，好像心里揣着小太阳，跟你待在一起，连心情都跟着亮堂起来。这种积极的日子态度，可不就是咱们身边的正能量担嘛。',
            '<SPK>0</SPK>杰伦哥！好久不见！最近你发的新歌里的钢琴独奏太赞了！我练了三天才勉强弹顺！<SPK>1</SPK>哟现在年轻人还听这么复古的编曲啊，我以为你们都在搞电子乐呢。不过你弹错了一个半音。<SPK>0</SPK>哇这都能听出来哈哈哈！<SPK>1</SPK>但是还是弹的不错哟。',
            '<SPK>0</SPK>哎呀，这不是我偶像华哥吗？哈哈哈！<SPK>1</SPK>贾玲您好，久仰大名！<SPK>0</SPK>别别别您这句久仰大名我可受不起！我才是从小看您的电影，听您的歌长大的。<SPK>1</SPK>那看来我真的是老一辈了！',
            '<SPK>0</SPK>嗨杰伦哥！好久不见！最近你发的新歌里的钢琴独奏太赞了！我练了三天才勉强弹顺！<SPK>1</SPK>哟现在年轻人还听这么复古的编曲啊，我以为你们都在搞电子乐呢。不过你弹错了一个半音。<SPK>0</SPK>哇这都能听出来哈哈哈！<SPK>1</SPK>但是还是弹的不错哟。<SPK>0</SPK>杰伦哥太厉害了吧！那个半音藏得好深，我回头一定找出来改。对了，这段钢琴旋律写得好抓耳，是先有旋律还是先有歌词呀？<SPK>1</SPK>这次是先有钢琴动机，慢慢延伸出整个旋律，歌词是后面填的。<SPK>0</SPK>原来如此！怪不得钢琴部分这么突出。我弹的时候总觉得情感差点，是不是该多琢磨歌词？<SPK>1</SPK>可以试试，把歌词的画面感融进指尖，音色会不一样。<SPK>0</SPK>学到了！对了，新歌里那个转音设计，唱起来是不是也很难？<SPK>1</SPK>你试试就知道了，那个转音得像绕弯子，不能太生硬。<SPK>0</SPK>哈哈，等我钢琴练好了，说不定挑战一下演唱部分！到时候可别笑我。',
            '<SPK>0</SPK>这不是我偶像华哥吗？哈哈哈！<SPK>1</SPK>贾玲您好，久仰大名！<SPK>0</SPK>别别别您这句久仰大名我可受不起！我才是从小看您的电影，听您的歌长大的。<SPK>1</SPK>那看来我真的是老一辈了！<SPK>0</SPK>华哥您可别这么说，您在我心里永远年轻！您的歌我现在还天天循环呢。<SPK>1</SPK>真的吗？那太感谢你支持了。最近有听我哪首歌？<SPK>0</SPK>忘情水啊！每次听都有不一样的感觉，百听不厌。<SPK>1</SPK>那首歌确实有些年头了，没想到还有人这么喜欢。<SPK>0</SPK>当然喜欢啦！对了华哥，您接下来有新电影计划吗？<SPK>1</SPK>正在看一些剧本，还没定下来。你呢？最近有什么新作品？<SPK>0</SPK>我刚拍完一部喜剧，希望到时候您能去看看，给我提提意见。<SPK>1</SPK>一定一定，你的喜剧我很期待，肯定很精彩。',
        ]

        out_dir = 'infer_out/tts_dialogue'
        out_dir = os.path.join(out_dir, f"{Path(lm_ckpt).stem}#{Path(dit_ckpt).stem}")
        os.makedirs(out_dir, exist_ok=True)

        for i in range(len(ref_audios)):
            audio_1, _ = librosa.load(ref_audios[i][0], sr=24000)
            audio_2, _ = librosa.load(ref_audios[i][1], sr=24000)
            ref_audio = np.concatenate([audio_1, audio_2])
            ref_text = '<SPK>0</SPK>' + ref_texts[i][0] + '<SPK>1</SPK>' + ref_texts[i][1]
            text = texts[i]

            wav = infer_ins.forward(text, ref_audio=ref_audio, ref_text=ref_text)

            save_name = Path(ref_audios[i][0]).stem + '+' + Path(ref_audios[i][1]).stem + '+' + text.replace('<SPK>', '[').replace('</SPK>', ']')[:10]
            save_name = save_name
            sf.write(f"{out_dir}/{save_name}.wav", wav, 24000, 'PCM_16')

    else:

        ref_audios = [
            'user/prompts/dzq_enhanced.wav',
        ]
        ref_texts = [
            '什么等下等下，我看到一个留言说什么邓紫曦都四十二岁了，你算错了吧！我十年前上我是歌手的时候，二十二岁啊兄弟，十年啊，二十二加十，三十二，懂吗？懂吗？三十二不是四十二，你算错了，谁教你的数学谁是你的数学老师。',
        ]
        texts = [
            '<SPK>0</SPK>我的老朋友，你真的很不简单,这段时间悄悄看着你，发现你跟人相处时，总是带着春风般的亲切劲儿，让人打心眼里觉得温暖。你对生活那股子热乎劲儿，谁看了都受感染。每天乐呵呵的，好像心里揣着小太阳，跟你待在一起，连心情都跟着亮堂起来。这种积极的日子态度，可不就是咱们身边的正能量担嘛。',
        ]

        out_dir = 'infer_out/tts'
        out_dir = os.path.join(out_dir, f"{Path(lm_ckpt).stem}#{Path(dit_ckpt).stem}")
        os.makedirs(out_dir, exist_ok=True)

        for i in range(len(ref_audios)):
            ref_audio, _ = librosa.load(ref_audios[i], sr=24000)
            ref_text = '<SPK>0</SPK>' + ref_texts[i]
            text = texts[i]

            wav = infer_ins.forward(text, ref_audio=ref_audio, ref_text=ref_text)

            save_name = Path(ref_audios[i]).stem + '+' + text.replace('<SPK>', '[').replace('</SPK>', ']')[:10]
            save_name = save_name
            sf.write(f"{out_dir}/{save_name}.wav", wav, 24000, 'PCM_16')

    # CUDA_VISIBLE_DEVICES=0 python inference/tts/scriptspeech_dialogue_infer.py
