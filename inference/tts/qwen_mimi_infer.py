import os

import torch
import soundfile as sf
import librosa

from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams

class QwenMimiInfer:
    def __init__(self, device, ckpt):
        self.build_model(device, ckpt)
        self.device = device

    def build_model(self, device, ckpt):
        set_hparams(config=os.path.join(ckpt, 'config.yaml'), print_hparams=False)

        from transformers import MimiModel, AutoFeatureExtractor
        self.audio_tokenizer = MimiModel.from_pretrained("kyutai/mimi").to(device)
        self.audio_tokenizer.eval()

        from transformers import AutoTokenizer, Qwen2Tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.text_tokenizer.add_tokens(['<BOT>'], special_tokens=True)
        self.text_tokenizer.add_tokens(['<BOS>'], special_tokens=True)
        semantic_token_num = 2048 * 1
        self.text_tokenizer.add_tokens([f'<Reserved_TTS_{i}>' for i in range(semantic_token_num)], special_tokens=True)

        from transformers import AutoConfig, AutoModelForCausalLM
        from utils.nn.embedding import resize_embedding_layer
        backbone_name = hparams.get('backbone_name', "Qwen/Qwen3-0.6B")
        backbone = AutoModelForCausalLM.from_pretrained(backbone_name, attn_implementation="flash_attention_2")
        resize_embedding_layer(backbone, len(self.text_tokenizer))

        decoder_name = hparams.get('decoder_name', "Qwen/Qwen3-0.6B")
        decoder_config = AutoConfig.from_pretrained(decoder_name)
        decoder_config.head_dim = 64
        decoder_config.hidden_size = 512
        decoder_config.intermediate_size = 2048
        decoder_config.num_attention_heads = 8
        decoder_config.num_hidden_layers = 16
        decoder_config.num_key_value_heads = 8
        if 'qwen3' in decoder_name.lower():
            from transformers.models.qwen3 import Qwen3ForCausalLM
            decoder = Qwen3ForCausalLM._from_config(decoder_config, attn_implementation="flash_attention_2")
        elif 'qwen2' in decoder_name.lower():
            from transformers import Qwen2ForCausalLM
            decoder = Qwen2ForCausalLM._from_config(decoder_config, attn_implementation="flash_attention_2")
        resize_embedding_layer(decoder, 2048 * 31 + 2)

        from modules.tts.codeclm.qwen_tts import Qwen3TTSMimiModel
        self.model = Qwen3TTSMimiModel(
            backbone=backbone,
            decoder=decoder,
            text_acoustic_token=2048 * 31,
            audio_codebook_size=2048,
            acoustic_n_quantizers=31,
            semantic_start_idx=self.text_tokenizer.encode('<Reserved_TTS_0>')[0],
            eos_idx=self.text_tokenizer.encode('<|endoftext|>')[0],
            backbone_padding_idx=self.text_tokenizer.encode('<|endoftext|>')[0],
            decoder_padding_idx=2048 * 31 + 1,
            decoder_frame_ratio=hparams.get('decoder_frame_ratio', 1/16),
            decoder_proj_bias=hparams.get('decoder_proj_bias', True),
            acoustic_encoder_zero_init=hparams.get('acoustic_encoder_zero_init', True)
        )
        self.model.eval()
        self.model.to(device)

        # load_ckpt(self.model, 'checkpoints/250615_qwen3mimi_01', 'model', strict=True)
        # load_ckpt(self.model, 'checkpoints/250617_qwen3mimi_01', 'model', strict=True)
        # load_ckpt(self.model, 'checkpoints/250617_qwen3mimi_fix', 'model', strict=True)
        load_ckpt(self.model, ckpt, 'model', strict=True)

    @torch.no_grad()
    def forward(self, text, ref_audio=None, ref_text=None):
        if ref_audio is not None and ref_text is not None:
            text = '<BOT>' + ref_text + text + '<BOS>'
            audio = ref_audio
            audio = torch.from_numpy(audio)[None, None, :].to(self.device)
            audio_padding_mask = torch.ones_like(audio)[0]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                audio_tokens = self.audio_tokenizer.encode(input_values=audio, padding_mask=audio_padding_mask).audio_codes
                
            semantic_tokens = audio_tokens[:, 0:1]
            acoustic_tokens = audio_tokens[:, 1:]   # [B, 31, Ta]
            semantic_tokens = semantic_tokens + self.model.semantic_start_idx
            acoustic_tokens = acoustic_tokens + torch.arange(31)[None, :, None].to(acoustic_tokens) * 2048
            audio_tokens = torch.cat([semantic_tokens, acoustic_tokens], dim=1).transpose(1, 2)     # [B, Ta, 32]
        else:
            text = '<BOT>' + text + '<BOS>'
            audio_tokens = None

        print('input text:', text)

        text_tokens = self.text_tokenizer([text], return_tensors="pt").input_ids.to(self.device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            semantic_tokens, acoustic_tokens = self.model.generate(
                text_tokens=text_tokens, 
                audio_tokens=audio_tokens,
                max_new_tokens=256,
                topk=50, temperature=1.0
            )

        print(self.text_tokenizer.decode(semantic_tokens[0].cpu().numpy().tolist()))
        semantic_tokens = semantic_tokens[..., None] - self.model.semantic_start_idx
        acoustic_tokens = acoustic_tokens - torch.arange(31)[None, None, :].to(acoustic_tokens) * 2048
        audio_tokens = torch.cat([semantic_tokens, acoustic_tokens], dim=-1).transpose(1, 2)    # [1, 32, Ta]

        print('audio_tokens', audio_tokens)

        if audio_tokens[0, 0, 0] == -1:
            audio_tokens = audio_tokens[:, :, 1:]

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            audio_values = self.audio_tokenizer.decode(audio_tokens)[0]

        print('audio_values.shape', audio_values.shape)

        audio_values = audio_values[0, 0].float().cpu().numpy()

        return audio_values


if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')

    # ckpt = 'checkpoints/250617_qwen3mimi_fix'
    # ckpt = 'checkpoints/250618_qwen2mimi_01'
    ckpt = 'checkpoints/250619_qwen2mimi_01'

    infer_ins = QwenMimiInfer('cuda', ckpt=ckpt)

    text = '造型方面，其实很有意思。第一次试妆的时候，很不一样。'
    audio = '/mnt/bn/sa-ag-data/liruiqi/code/MegaHuman/inference_api/prompt_api/celebrity_liuyifei.wav'
    audio, _ = librosa.load(audio, sr=24000)
    ref_text = '造型方面，其实很有意思。第一次试妆的时候，很不一样。'
    wav = infer_ins.forward(text, audio, ref_text)
    sf.write('infer_out/tts/out.wav', wav, 24000, 'PCM_16')

    # text = '尤其是二零一零年，有两篇吐槽葛军的帖子在网上走红以后，就此奠定了葛军“数学帝”这个称号。'
    # wav = infer_ins.forward(text)
    # sf.write('infer_out/tts/out.wav', wav, 24000, 'PCM_16')

    # CUDA_VISIBLE_DEVICES=0 python inference/tts/qwen_mimi_infer.py

    # <Reserved_TTS_1049><Reserved_TTS_1946><Reserved_TTS_624><Reserved_TTS_1923><Reserved_TTS_409><Reserved_TTS_542><Reserved_TTS_191><Reserved_TTS_1420><Reserved_TTS_547><Reserved_TTS_517><Reserved_TTS_180><Reserved_TTS_584><Reserved_TTS_761><Reserved_TTS_2032><Reserved_TTS_2032><Reserved_TTS_610><Reserved_TTS_1772><Reserved_TTS_1437><Reserved_TTS_1193><Reserved_TTS_97><Reserved_TTS_1162><Reserved_TTS_2034><Reserved_TTS_165><Reserved_TTS_913><Reserved_TTS_1638><Reserved_TTS_490><Reserved_TTS_1217><Reserved_TTS_449><Reserved_TTS_1729><Reserved_TTS_1925><Reserved_TTS_47><Reserved_TTS_202><Reserved_TTS_900><Reserved_TTS_1510><Reserved_TTS_64><Reserved_TTS_428><Reserved_TTS_93><Reserved_TTS_1792><Reserved_TTS_1340><Reserved_TTS_1283><Reserved_TTS_1853><Reserved_TTS_1755><Reserved_TTS_457><Reserved_TTS_1378><Reserved_TTS_762><Reserved_TTS_785><Reserved_TTS_1699><Reserved_TTS_946><Reserved_TTS_523><Reserved_TTS_1173><Reserved_TTS_1170><Reserved_TTS_1971><Reserved_TTS_493><Reserved_TTS_378><Reserved_TTS_1429><Reserved_TTS_1533><Reserved_TTS_884><Reserved_TTS_904><Reserved_TTS_433><Reserved_TTS_1291><Reserved_TTS_1473><Reserved_TTS_907><Reserved_TTS_1651><Reserved_TTS_14><Reserved_TTS_977><Reserved_TTS_907><Reserved_TTS_140><Reserved_TTS_661><Reserved_TTS_1619><Reserved_TTS_474><Reserved_TTS_664><Reserved_TTS_490><Reserved_TTS_409><Reserved_TTS_523><Reserved_TTS_1423><Reserved_TTS_970><Reserved_TTS_970><Reserved_TTS_389><Reserved_TTS_877><Reserved_TTS_209><Reserved_TTS_1995><Reserved_TTS_1283><Reserved_TTS_1853><Reserved_TTS_1853><Reserved_TTS_332><Reserved_TTS_806><Reserved_TTS_1745><Reserved_TTS_577><Reserved_TTS_1876><Reserved_TTS_1420><Reserved_TTS_1063><Reserved_TTS_1274><Reserved_TTS_595><Reserved_TTS_806><Reserved_TTS_1086><Reserved_TTS_1513><Reserved_TTS_1733><Reserved_TTS_1336><Reserved_TTS_380><Reserved_TTS_1724><Reserved_TTS_608><Reserved_TTS_904><Reserved_TTS_977><Reserved_TTS_1792><Reserved_TTS_140><Reserved_TTS_1035><Reserved_TTS_312><Reserved_TTS_1814>
