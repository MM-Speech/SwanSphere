import os
import random
import re
import tempfile
from datetime import datetime, timedelta
import collections
import collections.abc
from glob import glob

import numpy as np

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from attrdict import AttrDict
from typing import List, Optional, Dict

import yaml
from argparse import ArgumentParser
import torch.distributed as dist

import torch
import soundfile as sf
import librosa
from multiprocessing import Process, set_start_method

from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint, torch_load_dist
from utils.commons.hparams import set_hparams, hparams

from modules.asr.sensevoice.sensevoice_api import build_asr_model, run_asr_model
from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin, SemanticLMBuildModelMixin, build_vae, \
    build_audio_tokenizer
from utils.commons.upload_tos_utils import send_file_to_tos

cfg_weight = None
infer_step = 100
extend_dur = 0
vad_len = 0

def merge_model_weights(model, new_ckpt_path, ignore_module=[], weight=0.5):
    """
    Args:
        model: 已经加载了原始权重的 torch.nn.Module
        new_ckpt_path: 新的 checkpoint 文件路径 (state_dict)
        weight: 融合比例，merged = weight * old + (1 - weight) * new
    """
    # 加载新的 ckpt
    if os.path.isfile(new_ckpt_path):
        base_dir = os.path.dirname(new_ckpt_path)
        ckpt_path = new_ckpt_path
        new_state_dict = torch_load_dist(new_ckpt_path, map_location='cpu', mmap=None)
    else:
        base_dir = new_ckpt_path
        new_state_dict, ckpt_path = get_last_checkpoint(new_ckpt_path)
    # new_state_dict = torch.load(new_ckpt_path, map_location="cpu")
    print(f'merge model from {ckpt_path} with weight {1 - weight}')
    # 拿到旧模型参数
    old_state_dict = model.state_dict()

    merged_state_dict = {}
    new_state_dict = new_state_dict['dit']
    for k, old_param in old_state_dict.items():
        if k in new_state_dict and old_param.shape == new_state_dict[k].shape and not any(ign in k for ign in ignore_module):
            new_param = new_state_dict[k]
            merged_state_dict[k] = weight * old_param + (1 - weight) * new_param
        else:
            # 如果没有对应权重，就保留旧的
            merged_state_dict[k] = old_param

    # 加载融合后的权重
    model.load_state_dict(merged_state_dict)

    return model

def gen_audio_html(infos: Dict[int, dict], output_fp: Optional[str] = None,
                   title_name=None, extra_desc=None):
    if output_fp is None:
        output_fp = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name

    num_per_row = 5
    total = len(infos)
    rows = (total + num_per_row - 1) // num_per_row

    with open(output_fp, 'w') as f:
        print('<html lang="en">', file=f)
        print('<head>', file=f)
        print('<meta charset="UTF-8">', file=f)
        print('<meta name="viewport" content="width=device-width, initial-scale=1.0">', file=f)
        print('<style>', file=f)
        print('''
            body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
            .container { max-width: 1280px; margin: 0 auto; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            h1 {
                  font-size: 2em;
                  margin-bottom: 0.2em;
                }
            p.description {
                  color: #666;
                  margin-bottom: 1.5em;
            }
            td { padding: 10px; border: 2px solid DodgerBlue; vertical-align: top; text-align: center; }
            audio { width: 100%; }
            .desc { margin-top: 10px; white-space: pre-wrap; text-align: left; font-size: 14px; background: #f8f8f8; padding: 8px; border-radius: 5px; }
        ''', file=f)
        print('</style>', file=f)
        print('</head>', file=f)
        print('<body>', file=f)
        if title_name is not None:
            print(f'  <h1>{title_name}</h1>', file=f)
        if extra_desc is not None:
            print(f'  <p class="description">{extra_desc}</p>', file=f)
        print('<div class="container">', file=f)
        print('<table>', file=f)

        keys = list(infos.keys())

        for row in range(rows):
            print('<tr>', file=f)
            for col in range(num_per_row):
                idx = row * num_per_row + col
                if idx >= total:
                    print('<td></td>', file=f)
                    continue
                info = infos[idx]
                tos_url = info.get('tos_url', '')
                if 'global_description' in info:
                    global_description = info.get('global_description', '')
                    fine_grained_transcription = info.get('fine_grained_transcription', '')
                    env_bgm = info.get('env_bgm', '')
                    speaker = info.get('speaker', '')
                elif 'subjects' in info:
                    subjects = info['subjects']
                    narration = info['narration']
                    visual = info['visual']
                elif 'caption' in info:
                    caption = info['caption']
                print('<td>', file=f)
                print(f'<audio controls preload="none">', file=f)
                print(f'  <source src="{tos_url}" type="audio/mpeg">', file=f)
                print('  Your browser does not support the audio element.', file=f)
                print('</audio>', file=f)
                print('<div class="desc">', file=f)
                if 'global_description' in info:
                    print(f'Global Prompt: {global_description}\n', file=f)
                    print(f'BGM Prompt: {env_bgm}\n', file=f)
                    print(f'Local Prompt: {fine_grained_transcription.replace("<", "&lt;").replace(">", "&gt;")}\n', file=f)
                    print(f'Speaker: {speaker}', file=f)
                elif 'subjects' in info:
                    print(f'subjects: {subjects}\n', file=f)
                    print(f'narration: {narration.replace("<", "&lt;").replace(">", "&gt;")}\n', file=f)
                    print(f'visual: {visual}\n', file=f)
                else:
                    print(f'caption: {caption.replace("<", "&lt;").replace(">", "&gt;")}\n', file=f)
                print('</div>', file=f)
                print('</td>', file=f)
            print('</tr>', file=f)

        print('</table>', file=f)
        print('</div>', file=f)
        print('</body>', file=f)
        print('</html>', file=f)

    return output_fp

def upload_tos_html(yml=None, out_path=None, title_name=None, extra_desc=None):
    sub_dir = 'scriptspeech_gqj'
    if yml is None:
        yml = './egs/inference/inference_na_sample0728.yaml'
    if out_path is None:
        out_path = 'infer_out/tts/250625_scriptspeech_semanticlm_singlespk_01_250828_scriptspeech_dit_s2_gokul_c_vm_1_20250901_081432'

    from collections import defaultdict
    infos = defaultdict(dict)

    with open(yml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        samples = cfg['samples']
    files = glob(f'{out_path}/*.wav')
    # assert len(files) == len(samples)
    for idx, sample in enumerate(samples):
            wav = os.path.join(out_path, f"out_{idx}.wav")
            if not os.path.exists(wav):
                continue
            tos_url = send_file_to_tos(wav, sub_dir=sub_dir)
            # tos_url = f"https://tosv-sg.tiktok-row.org/obj/sa-ag-sg-research-sg/scriptspeech_gqj/tts_250625_scriptspeech_semanticlm_singlespk_01_250723_scriptspeech_dit_s2_c_1_20250724_070952_out_{idx}.wav"
            print("tos_url: ", tos_url)
            infos[idx]['tos_url'] = tos_url
            infos[idx].update(sample)

    # 调用gen_html函数时传递names参数
    html_path = gen_audio_html(infos, title_name=title_name, extra_desc=extra_desc)
    print(f"生成的HTML文件路径：{html_path}")
    html_tos = send_file_to_tos(html_path, sub_dir=sub_dir)
    print(f"生成的HTML文件路径：{html_tos}")
    return html_tos

class ScriptSpeechInfer(DiTBuildModelMixin, SemanticLMBuildModelMixin):
    def __init__(self, device, dit_ckpt, lm_ckpt, merge_ckpt=None, merge_weight=0.5):
        self.device = device
        self.build_model(dit_ckpt, lm_ckpt, merge_ckpt, merge_weight)

    def build_model(self, dit_ckpt, lm_ckpt, merge_ckpt=None, merge_weight=0.5):
        # self.asr_model = build_asr_model(self.device)

        set_hparams(config=os.path.join(dit_ckpt, 'config.yaml'), print_hparams=False, global_hparams=True)
        hparams["exp_name"] = 'infer'
        self.config = AttrDict(hparams)
        self.lm_hparams = set_hparams(config=os.path.join(lm_ckpt, 'config.yaml'), print_hparams=False,
                                      global_hparams=False)
        self.lm_hparams['gradient_checkpointing'] = False

        self.vae, self.hp_vae = build_vae(hparams.get('vae_ckpt'))
        self.vae.to(self.device)

        self.audio_token_feature_extractor, self.audio_tokenizer, self.audio_vocab_size = build_audio_tokenizer(
            hparams.get('audio_tokenizer', 'glm4v'))
        self.audio_tokenizer.to(self.device)

        self.dit_text_tokenizer, self.dit_vocab_size = self .build_dit_text_tokenizer()
        self.dit = self.build_dit(hparams)
        load_ckpt(self.dit, dit_ckpt, 'dit', strict=False)
        if merge_ckpt is not None:
            self.dit = merge_model_weights(self.dit, merge_ckpt, ['cross', 'caption_proj'], merge_weight)
        self.dit.eval()
        self.dit.to(self.device)

        self.use_caption = hparams.get('load_sd_text_encoder', False)

        if self.use_caption:
            if hparams.get('model_size', 'base') == 'seedance_7b':
                self.build_sd_text_encoder(hparams['text'])
                self.sd_text_encoder.eval()
                self.sd_text_encoder.to(self.device)
            elif 'goku' in hparams.get('model_size', 'base'):
                self.build_goku_text_encoder(hparams)
                self.goku_text_encoder.eval()
                if int(os.environ.get("WORLD_SIZE", 1)) > 1:
                    self.goku_text_encoder.to(self.device)
        else:
            # 可选：给个占位，避免误访问
            self.sd_text_encoder = None
            self.goku_text_encoder = None

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
        if int(os.environ.get("WORLD_SIZE", 1)) == 1:
            self.dit.cpu()
            self.goku_text_encoder.to(self.device)
        encoder_hidden_states = self.goku_text_encoder(
            input_ids, return_dict=False,
            attention_mask=attention_masks,
        )[0]
        if int(os.environ.get("WORLD_SIZE", 1)) == 1:
            self.goku_text_encoder.cpu()
            self.dit.to(self.device)
        captions_cmask = self.generate_mask(input_ids,
                                            start_token=special_token_ids[0],
                                            end_token=special_token_ids[1])
        return encoder_hidden_states, captions_cmask, attention_masks

    @torch.no_grad()
    def forward(self, text, ref_audio=None, ref_text=None, prompt=None, cfg_w=None, negative_prompt=None,
                infer_length=5.0, start_time=0.2, end_time=0.2, num_step=100):
        speech = len(text) > 0
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
                    asr_result = run_asr_model(temp_path, self.asr_model, use_itn=False)
                    print('asr_result', asr_result)
                    ref_text = asr_result['text']
                    ref_text = ref_text + '.'
            if cfg_w is None:
                cfg_w = [2, 2, 2, 5]
        else:
            ref_text = None
            ref_wav = None
            if cfg_w is None:
                cfg_w = [5, 10, 5, 0] # all text caption
        print('| CFG: ', cfg_w)

        tgt_len = int(infer_length * 24000 / self.hp_vae['hop_size'] / self.hp_vae['vae_stride']) # includ ref

        # caption
        caption = prompt
        print('| caption:', caption)

        use_caption = getattr(self, "use_caption", False) and (caption is not None)

        caption_emb = None
        caption_lens = None
        uc_caption_emb = None
        uc_caption_lens = None

        if use_caption:
            if 'seedance' in self.config.model_size:
                sd_text_output, captions_cmask = self.run_sd_text_encoder([caption])
                text_embs = sd_text_output.embeddings * sd_text_output.masks[:, :, None]
                caption_lens = sd_text_output.masks.sum(-1)

                # 把 text_emb + mask 拼成一条 caption_emb
                caption_emb = torch.cat(
                    [text_embs, captions_cmask[:, :, None]], -1
                )
                # 如果想保持 CFG 的 5 条分支结构，可以简单复制一份当 uc
                uc_caption_emb = caption_emb.clone()
                uc_caption_lens = caption_lens.clone()

            elif 'goku' in self.config.model_size:
                gk_text_embs, captions_cmask, text_att_mask = self.run_goku_text_encoder([caption])
                text_embs = gk_text_embs * text_att_mask[..., None]
                caption_lens = text_att_mask.sum(-1)

                # negative prompt
                if negative_prompt is None:
                    negative_prompt = '<I>distorted audio</I><I>background static</I>...'  # 你的原始字符串
                uc_gk_text_embs, uc_captions_cmask, uc_text_att_mask = self.run_goku_text_encoder([negative_prompt])
                uc_text_embs = uc_gk_text_embs * uc_text_att_mask[..., None]
                uc_caption_lens = uc_text_att_mask.sum(-1)

                caption_emb = torch.cat(
                    [text_embs, captions_cmask[:, :, None]], -1
                )
                uc_caption_emb = torch.cat(
                    [uc_text_embs, uc_captions_cmask[:, :, None]], -1
                )

        # 只有在真的有 caption 时，才复制 5 路做 seq_cfg_w
        if caption_emb is not None:
            caption_emb = torch.cat([
                caption_emb,
                uc_caption_emb,
                caption_emb,
                uc_caption_emb,
                uc_caption_emb,
            ], dim=0)
            caption_lens = torch.cat([
                caption_lens,
                uc_caption_lens,
                caption_lens,
                uc_caption_lens,
                uc_caption_lens,
            ], dim=0).long()
        else:
            caption_emb = None
            caption_lens = None

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
        txt_tokens = text_inputs['input_ids'].clone()
        txt_mask = text_inputs['attention_mask'].bool()
        txt_tokens[~txt_mask] = self.cfg_mask_text_token

        vad_mask = torch.zeros_like(lat[:, :, :1])
        if not self.config.get('drop_vad', False) and speech:
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

        # semantic_tokens = torch.cat([
        #     semantic_tokens,
        #     torch.full(semantic_tokens.size(), self.cfg_mask_audio_token, device=self.device),
        #     semantic_tokens,
        #     torch.full(semantic_tokens.size(), self.cfg_mask_audio_token, device=self.device),
        #     torch.full(semantic_tokens.size(), self.cfg_mask_audio_token, device=self.device),
        # ], dim=0) TODO? replace by caption

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
            'semantic_tokens': None,
            "caption_emb": caption_emb,
            "caption_lens": caption_lens,  # B
            'vad_mask': vad_mask
        }
        global cfg_weight, infer_step, extend_dur, vad_len
        cfg_weight, infer_step, extend_dur, vad_len = cfg_w, num_step, infer_length, [start_time, end_time]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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

def extract_w_text(s: str) -> str:
    # 提取所有 <w>...</w> 中间的内容
    s = s.replace('<W>', '<w>').replace('</W>', '</w>')
    matches = re.findall(r"<w>(.*?)</w>", s, flags=re.DOTALL)
    # 用 ". " 拼接
    return ". ".join(m.strip() for m in matches if m.strip())

def worker(rank, world_size, args, cfg, out_path):
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    if world_size > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'  # 本机通信地址
        os.environ['MASTER_PORT'] = '10198'      # 通信端口
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(rank)
        from utils.commons import trainer
        trainer.LOCAL_RANK = rank
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, device_id=torch.device(rank), timeout=timedelta(seconds=3000))


    dit_ckpt = args.dit_ckpt
    lm_ckpt = args.lm_ckpt
    infer_ins = ScriptSpeechInfer(device, dit_ckpt=dit_ckpt, lm_ckpt=lm_ckpt,
                                  merge_ckpt=args.merge_ckpt, merge_weight=args.merge_weight)

    # 每个 rank 单独处理属于自己的样本
    os.makedirs(out_path, exist_ok=True)
    negative_prompt = cfg.get('negative_prompt', None)
    for idx, sample in enumerate(cfg['samples']):
        if idx % world_size != rank:
            continue
        print(f"[Rank {rank}] Processing sample {idx}: {sample}")
        if 'env_bgm' in sample:
            env_bgm = sample['env_bgm']
            fine_grained_transcription = sample['fine_grained_transcription']
            global_description = sample['global_description']
            speaker = sample['speaker']
            speech = sample['speech']
            text = extract_w_text(fine_grained_transcription)
            fine_grained_transcription = fine_grained_transcription.replace('<tag>', '<TAG>').replace('</tag>', '</TAG>').replace('<w>', '<W>').replace('</w>', '</W>')
            caption = f'{global_description} The voice is very clean and clear. {env_bgm} {speaker if speaker != "" else "There is no one speaking in the scene."} {fine_grained_transcription}'
        elif 'subjects' in sample:
            subjects = sample['subjects']
            narration = sample['narration']
            visual = sample['visual']
            text = extract_w_text(narration)
            # if len(subjects) == 0:
            #     subjects_str = 'No subjects. '
            # else:
            #     subjects_str = ''
            #     for sub in subjects:
            #         subjects_str = subjects_str + sub + ' '
            caption = f'Subjects: {subjects} Visual: {visual} Narration: {narration}'.replace('<tag>', '<TAG>').replace('</tag>',
                                                                                                       '</TAG>').replace('<w>', '<W>').replace('</w>', '</W>')
        else:
            caption = sample['caption']
            text = extract_w_text(caption)
        set_seed(len(text) + idx)


        if 'prompt_audio' in sample:
            audio = sample['prompt_audio']
            audio, _ = librosa.load(audio, sr=24000)
        else:
            audio = None
        wav = infer_ins.forward(text, ref_audio=audio, prompt=caption,
                                cfg_w=cfg.get('cfg_w', None),
                                infer_length=cfg.get('infer_length', 0),
                                num_step=cfg.get('num_step', 100),
                                start_time=cfg.get('vad_len', 0.2), end_time=cfg.get('vad_len', 0.2),
                                negative_prompt=negative_prompt)
        if wav is not None:
            print(f'save wav at {out_path}/out_{idx}.wav')
            sf.write(f'{out_path}/out_{idx}.wav', wav, 24000, 'PCM_16')

    print(f"[Rank {rank}] Finished all samples.")

if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv

        load_dotenv('.env.local')

    kill_void()

    try:
        set_start_method('spawn')  # 多进程启动方式，Linux/Windows 通用
    except RuntimeError:
        pass

    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--dit_ckpt", help="Path to model", type=str,
                        default='checkpoints/250622_scriptspeech_dit_singlespk_01')
    parser.add_argument("--lm_ckpt", help="Path to model", type=str,
                        default='checkpoints/250709_ss_lm_singlelocal')
    parser.add_argument("--merge_ckpt", help="Path to merge model", type=str)
    parser.add_argument("--merge_weight", help="Weight to merge model", type=float)
    args = parser.parse_args()
    # 读取 config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_path = f'{cfg["out_path"]}/{os.path.basename(args.lm_ckpt)}_{os.path.basename(args.dit_ckpt)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # 启动多进程，每个进程绑定一张 GPU
    processes = []
    gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    for rank in range(gpus):
        p = Process(target=worker, args=(rank, gpus, args, cfg, out_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All ranks finished. 可以做后处理或上传结果")
    # upload_tos_html(out_path)  # 可选，上传结果
    desc = (f"Inference setting: cfg weight: {cfg.get('cfg_w', None)}, inference step: {infer_step}, "
            f"extend duration: {extend_dur}, vad length (silence duration at bugin and tail): {vad_len}")
    upload_tos_html(yml=args.config, out_path=out_path, title_name=os.path.basename(out_path), extra_desc=desc)
    # CUDA_VISIBLE_DEVICES=0 python inference/tts/scriptspeech_infer.py

