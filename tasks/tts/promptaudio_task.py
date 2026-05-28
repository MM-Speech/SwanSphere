import os
import random
import re

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import random_split
import numpy as np
from copy import deepcopy
import soundfile as sf

from utils.commons.base_task import BaseTask
from utils.commons.ckpt_utils import load_ckpt,get_last_checkpoint
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.hparams import hparams, set_hparams
from utils.commons.os_utils import kill_void
from utils.commons.dataset_utils import data_loader, build_dataloader
from utils.commons.trainer import LOCAL_RANK
from utils.commons.io import print_once
from utils.nn.schedulers import WarmupSchedule, CosineSchedule, CosineAnnealingWarmRestartsWithWarmup
from utils.nn.seq_utils import sequence_mask, add_prefix
from utils.nn.model_utils import print_arch, num_params, unwrap_model
from utils.nn.ema import EMAModel, restore_ema
from utils.commons.tensor_utils import move_to_cpu, convert_to_np

from modules.tts.scriptspeech.build_model_utils import build_vae, build_qwen3, shard_model_in_node, DiTBuildModelMixinV2
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
from tasks.tts.task_utils.prompttts_task_utils import build_dialogue_mask_from_ids


class FastDatasetDiTBaseTask(FastDatasetMixin, BaseTask):
    def __init__(self):
        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
        self.hparams = hparams
        self.config = AttrDict(hparams)

        super().__init__()
    
    def build_ema_model(self):
        print_once(f'| Building EMA model with decay={self.config.ema_decay} ...')
        self.ema_model = deepcopy(self.dit)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_model.to(self.trainer.device)
    
    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        if hparams.get('use_ema', False):
            self.ema_update(self.ema_model, self.dit, self.config.ema_decay)
        
    @torch.no_grad()
    def ema_update(self, ema_model, model, decay):
        ema_params = dict(unwrap_model(ema_model).named_parameters())
        for n, p in unwrap_model(model).named_parameters():
            p_ema = ema_params[n]
            src = p.detach()
            if src.dtype != torch.float32:
                src = src.float()
            if p_ema.dtype != torch.float32:
                p_ema.data = p_ema.data.float()
            p_ema.mul_(decay).add_(src.to(p_ema.device), alpha=1.0 - decay)
            
    def load_model(self):
        """
        当同时提供 hparams['load_ckpt_audio'] 与 hparams['load_ckpt_speech'] 时：
        - feed_forward.*    ← 来自 speech
        - feed_forward_audio.* ← 来自 audio 的 feed_forward.*（键名映射）
        - 其他参数：audio 与 speech 同形状则逐元素平均；否则择其一；都无则保留初始化
        其他情况保持原逻辑，直接调用 load_ckpt。
        """

        lc = hparams.get('load_ckpt', '')
        audio = hparams.get('load_ckpt_audio', '')
        speech = hparams.get('load_ckpt_speech', '')

        # ---------- 工具函数（仅在本函数内使用） ----------
        def _resolve_single_ckpt(path_or_dir):
            """返回 (checkpoint_obj, ckpt_path_str)。兼容文件与目录。"""
            if isinstance(path_or_dir, dict) and ('state_dict' in path_or_dir or 'global_step' in path_or_dir):
                # 已是 ckpt 对象
                return path_or_dir, '<in-memory>'
            if os.path.isfile(path_or_dir):
                ckpt_path = path_or_dir
                ckpt = torch_load_dist(ckpt_path, map_location='cpu', mmap=True)
                return ckpt, ckpt_path
            # 视为目录
            base = path_or_dir
            last_path = f'{base}/model_only_last.ckpt'
            if os.path.exists(last_path):
                ckpt = torch_load_dist(last_path, map_location='cpu', mmap=True)
                return ckpt, last_path
            ckpt, ckpt_path = get_last_checkpoint(base, None)
            return ckpt, ckpt_path

        def _state_dict_all(ckpt):
            sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            return {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in sd.items()}

        def _extract_sub(sd_all, name):
            """
            抽取子模块 state_dict（相对键），兼容两种保存方式：
            1) sd_all[name] 是 dict
            2) sd_all 是扁平的，键以 'name.' 开头
            """
            if name in sd_all and isinstance(sd_all[name], dict):
                sub = sd_all[name]
                return {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in sub.items()}
            prefix = name + '.'
            sub = {k[len(prefix):]: v for k, v in sd_all.items() if k.startswith(prefix)}
            return {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in sub.items()}

        def _avg_tensors(va, vs, target_dtype):
            return ((va.to(torch.float32) + vs.to(torch.float32)) * 0.5).to(target_dtype)

        # ---------- 融合分支 ----------
        if audio and speech:
            model_name = 'dit'

            ckpt_a, path_a = _resolve_single_ckpt(audio)
            ckpt_s, path_s = _resolve_single_ckpt(speech)
            if ckpt_a is None or ckpt_s is None:
                print("| WARN: audio/speech ckpt resolve failed, fallback to default load_ckpt path.")
                # 退回到原有双路径直传（若你的老 load_ckpt 支持 dict/list，可用；否则不会触发）
                spec = {'audio': audio, 'speech': speech}
                load_ckpt(self.dit, spec, model_name, strict=False, mmap=True)
                return

            sd_all_a = _state_dict_all(ckpt_a)
            sd_all_s = _state_dict_all(ckpt_s)
            sd_a = _extract_sub(sd_all_a, model_name)
            sd_s = _extract_sub(sd_all_s, model_name)

            cur_sd = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in self.dit.state_dict().items()}
            merged = {}

            # 统计信息
            n_avg = n_audio_only = n_speech_only = n_ffn_audio = n_ffn_speech = n_fallback = 0

            for k, cur_param in cur_sd.items():
                # 1) feed_forward_audio.*  ← audio 的 feed_forward.*
                if '.feed_forward_audio.' in k:
                    src = k.replace('.feed_forward_audio.', '.feed_forward.')
                    val = None
                    if src in sd_a and sd_a[src].shape == cur_param.shape:
                        val = sd_a[src]
                    elif k in sd_a and sd_a[k].shape == cur_param.shape:  # 兜底：audio 里恰好已有同名分路
                        val = sd_a[k]
                    if val is None:
                        merged[k] = cur_param
                        n_fallback += 1
                        print(f"| WARN: audio→feed_forward_audio 缺失或不匹配，保持初始化: {k}")
                    else:
                        merged[k] = val
                        n_ffn_audio += 1
                    continue

                # 2) feed_forward.*       ← speech 的 feed_forward.*
                if '.feed_forward.' in k:
                    vs = sd_s.get(k, None)
                    if vs is not None and vs.shape == cur_param.shape:
                        merged[k] = vs
                        n_ffn_speech += 1
                    else:
                        # 兜底：可尝试 audio 的同名键，否则保持初始化
                        va = sd_a.get(k, None)
                        if va is not None and va.shape == cur_param.shape:
                            merged[k] = va
                            n_audio_only += 1
                            print(f"| NOTE: speech 缺失/不匹配，用 audio 代替: {k}")
                        else:
                            merged[k] = cur_param
                            n_fallback += 1
                            print(f"| WARN: speech→feed_forward 缺失或不匹配，保持初始化: {k}")
                    continue

                # 3) 其他键：平均或择一
                va = sd_a.get(k, None)
                vs = sd_s.get(k, None)
                if (va is not None) and (vs is not None) and (va.shape == vs.shape == cur_param.shape):
                    merged[k] = _avg_tensors(va, vs, cur_param.dtype)
                    n_avg += 1
                elif vs is not None and vs.shape == cur_param.shape:
                    merged[k] = vs
                    n_speech_only += 1
                elif va is not None and va.shape == cur_param.shape:
                    merged[k] = va
                    n_audio_only += 1
                else:
                    merged[k] = cur_param
                    n_fallback += 1
                    _va = None if va is None else tuple(va.shape)
                    _vs = None if vs is None else tuple(vs.shape)
                    print(f"| WARN: missing/mismatch -> keep init: {k} | audio={_va} speech={_vs} cur={tuple(cur_param.shape)}")

            # 组装“伪”checkpoint，交给原始 load_ckpt 去做最后的 load_state_dict 等流程
            gstep = max(
                (ckpt_a.get('global_step', 0) if isinstance(ckpt_a, dict) else 0),
                (ckpt_s.get('global_step', 0) if isinstance(ckpt_s, dict) else 0),
            )
            fused_ckpt = {'state_dict': {model_name: merged}, 'global_step': gstep}

            # 使用原函数加载（不改其实现）
            load_ckpt(self.dit, '', model_name, strict=False, mmap=True, checkpoint=fused_ckpt)

            print(f"| Mixed-loaded '{model_name}' from audio='{path_a}' & speech='{path_s}'.")
            print(f"| Stats: avg={n_avg}, audio_only={n_audio_only}, speech_only={n_speech_only}, "
                f"ffn_audio={n_ffn_audio}, ffn_speech={n_ffn_speech}, fallback={n_fallback}")
            return

        # ---------- 保持原有逻辑 ----------
        if isinstance(lc, (dict, list, tuple)):
            load_ckpt(self.dit, lc, 'dit', strict=False, mmap=True)
        elif isinstance(lc, str) and lc != '':
            load_ckpt(self.dit, lc, 'dit', strict=False, mmap=True)

                
    def build_optimizer(self):
        if not hparams.get('disable_weight_decay_on_bias_and_norm_and_embed', True):
            optimizer = AdamW(unwrap_model(self.dit).parameters(), **self.config.optimizer)
        
        else:
            def has_name(names, param_name):
                if not isinstance(names, list):
                    names = [names]
                for name in names:
                    if name in param_name:
                        return True
                return False
                
            decay_params = []
            no_decay_params = []
            for name, param in unwrap_model(self.dit).named_parameters():
                if not param.requires_grad:
                    continue
                if param.dim() == 1 or has_name(['bias', 'norm', 'text_embedder', 'tone_embed', 'ph_embed'], name):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
                    
            print_once(f"| Weight decay is canceld for {len(no_decay_params)} params")

            optimizer_groups = [
                {'params': decay_params, 'weight_decay': self.config.optimizer.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_groups, **self.config.optimizer)

        return optimizer
    
    def build_scheduler(self, optimizer):
        return CosineAnnealingWarmRestartsWithWarmup(
            optimizer, lr_max=hparams['optimizer']['lr'], warmup_updates=hparams.get('warmup_updates', 5000), 
            total_updates=1000000, initial_period=hparams.get('scheduler_initial_period', 10000), 
            period_mult=hparams.get('scheduler_period_mult', 1.2), lr_min=hparams.get('scheduler_lr_min', 1.0e-5)
        )

    def fsdp_optm2model(self):
        return [self.dit]

    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding
        import modules.flow_matching.llama
        import modules.asr.llama.llama_seq2seq
        import modules.tts.llama_dit.llama_moe

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                # Linear, Sequential, Conv1d, Conv2d, Embedding,
                modules.flow_matching.llama.TransformerBlock,
                modules.asr.llama.llama_seq2seq.DecoderBlock,
                modules.asr.llama.llama_seq2seq.EncoderBlock,
                modules.tts.llama_dit.llama_moe.TransformerBlock,
                get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy
    
    def on_epoch_start(self):
        super().on_epoch_start()
        kill_void()

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        infer_steps = self.hparams.get('infer_steps', 12)
        outputs = self._validation_step(sample, batch_idx, infer_steps)
        return outputs

    def _validation_step(self, sample, batch_idx, infer_steps):
        outputs = {}
        if self.trainer.proc_rank == 0:
            pass
        return outputs

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        infer_steps = hparams['infer_steps']
        return self._validation_step(sample, batch_idx, infer_steps)


class PromptAudioTask(DiTBuildModelMixinV2, FastDatasetDiTBaseTask):
    def build_model(self):
        # 原逻辑
        self._build_model()
        self.vae.to(self.trainer.device)

        # === 新增：确保 caption tokenizer 拥有所需特殊 token，并缓存其 id ===
        self._ensure_caption_special_tokens()

        if hparams.get('use_ema', False):
            self.build_ema_model()
            return {'trainable': [self.dit, self.ema_model], 'others': []}
        return {'trainable': [self.dit], 'others': []}

    # ------------------------ 新增/改：caption 标注支持 ------------------------

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.1:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'diff_loss': 1.0,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])
        
        if self.trainer.proc_rank == 0 and self.global_step < 10:
            save_dir = f'{hparams["work_dir"]}/sample_batches/step_{self.global_step}'
            os.makedirs(save_dir, exist_ok=True)
            sample = convert_to_np(move_to_cpu(sample))
            for i in range(sample['nsamples']):
                sf.write(f"{save_dir}/{i}.wav", sample['wavs'][i, :sample['wav_lengths'][i]], 24000, 'PCM_16')
                sf.write(f"{save_dir}/{i}_ctx.wav", sample['ctx_wavs'][i, :sample['wav_lengths'][i]], 24000, 'PCM_16')
                np.save(f"{save_dir}/{i}.npy", sample, allow_pickle=True)
            del sample['wavs']
            del sample['ctx_wavs']

        return total_loss, loss_output

    def _ensure_caption_special_tokens(self):
        """
        确保 caption_tokenizer 有我们要用的特殊 tag；并缓存它们的 id。
        """

        tok = self.caption_tokenizer
        self._caption_special_ids = {
            "S1_START": tok.convert_tokens_to_ids("<S1>"),
            "S1_END": tok.convert_tokens_to_ids("</S1>"),
            "AUDIO_START": tok.convert_tokens_to_ids("<Audio>"),
            "AUDIO_END": tok.convert_tokens_to_ids("</Audio>"),
            "BGM_START": tok.convert_tokens_to_ids("<BGM>"),
            "BGM_END": tok.convert_tokens_to_ids("</BGM>"),
            "TAG_START": tok.convert_tokens_to_ids("<TAG>"),
            "TAG_END": tok.convert_tokens_to_ids("</TAG>"),
            "GP_START": tok.convert_tokens_to_ids("<GPROMPT>"),
            "GP_END": tok.convert_tokens_to_ids("</GPROMPT>"),
        }

    @torch.no_grad()
    def _build_caption_mark_from_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        将带 tag 的 caption token 序列转为 mark（0=其它/填充, 1=文本, 2=音效, 3=BGM）
        input_ids: [B, T], attention_mask: [B, T]
        return: LongTensor [B, T]
        """
        ids = self._caption_special_ids
        start2label = {
            ids["S1_START"]: 1,
            ids["TAG_START"]: 1,
            ids["GP_START"]: 1,
            ids["AUDIO_START"]: 2,
            ids["BGM_START"]: 3,
        }
        endset = {
            ids["S1_END"],
            ids["TAG_END"],
            ids["GP_END"],
            ids["AUDIO_END"],
            ids["BGM_END"],
        }

        B, T = input_ids.shape
        device = input_ids.device
        mark = torch.zeros((B, T), dtype=torch.long, device=device)

        for b in range(B):
            cur = 0
            stack = []
            for t in range(T):
                tok = input_ids[b, t].item()
                if tok in start2label:
                    stack.append(cur)
                    cur = start2label[tok]
                    # tag token 本身记 0
                    mark[b, t] = 0
                    continue
                if tok in endset:
                    # 结束 tag，本身记 0，并恢复上一个状态
                    mark[b, t] = 0
                    cur = stack.pop() if len(stack) else 0
                    continue
                # 普通内容：按当前段落类型打标
                mark[b, t] = cur

        # 去掉 padding 部分
        return mark * attention_mask.long()

    # --------------------------------------------------------------------------

    @torch.no_grad()
    def run_caption_encoder(self, captions, device):
        """
        改动：在返回 encoder_hidden_states 与 attention_mask 的同时，
        额外返回多类别 caption_text_mark（0/1/2/3）。
        """
        inputs = self.caption_tokenizer(
            captions,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)             # [B, T]
        attention_masks = inputs.attention_mask.to(device)  # [B, T]

        encoder_hidden_states = self.caption_encoder(
            input_ids, return_dict=False,
            attention_mask=attention_masks,
        )[0]  # [B, T, H]

        # 多类别 mark：文本=1、音效=2、BGM=3（其它/填充=0）
        caption_text_mark = self._build_caption_mark_from_ids(input_ids, attention_masks)

        # 与原接口保持一致（上游已经写成三元返回）
        return encoder_hidden_states, attention_masks, caption_text_mark

    # --------------------------------------------------------------------------

    def run_model(self, sample, infer=False, infer_steps=None):
        """
        改动要点（与需求严格一致）：
        - audio prompt = ctx 组（按 prompt_ratio 划分的“另一组”）；ctx 组保留原始 caption（cap_from_ds）。
        - prompt 组：caption = <GPROMPT>{global}</GPROMPT> + {local_prompt(将<tag>标准化为<TAG>)}；
                    不再拼接原 cap_from_ds。
        - 其余编码/CFG/对齐/损失逻辑保持不变。
        """
        model_out = {}
        losses_out = {}
        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out

        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        ctx_wavs = sample["ctx_wavs"]
        ctx_mask = sample["ctx_mask"].float()
        if len(ctx_mask.shape) == 2:
            ctx_mask = ctx_mask[:, :, None]

        text = sample['text']
        ph_tokens = sample["ph_tokens"]
        tone_tokens = sample["tone"]
        mel2ph = sample['mel2ph']

        bsz, device = wavs.shape[0], wavs.device

        # 来自数据集的字段
        cap_from_ds = sample['caption'] if 'caption' in sample else [''] * bsz
        global_prompts = sample.get('global_prompt', [''] * bsz)
        local_prompts = sample.get('local_prompt', [''] * bsz)

        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']

        # 英文调类规范（沿用原逻辑）
        en_tone_idx = ~((tone_tokens == 4) | ((11 <= tone_tokens) & (tone_tokens <= 15)) | (tone_tokens == 0))
        tone_tokens[en_tone_idx] = 3

        # ===== audio encode =====
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                lat = self.vae.encode_latent(wavs)
                lat_ctx = self.vae.encode_latent(ctx_wavs)
                lat_ctx = torch.nn.functional.pad(
                    lat_ctx, (0, 0, 0, lat.size(1) - lat_ctx.size(1)), mode='constant', value=0
                )

        # ===== text tokenize =====
        text_inputs = self.dit_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
        txt_tokens = text_inputs['input_ids']   # [B, T]
        txt_mask = text_inputs['attention_mask'].bool()
        txt_tokens[~txt_mask] = self.cfg_mask_text_token
        txt_lens = txt_mask.int().sum(1)

        # loss_mask 先算（与原一致）
        loss_mask = sequence_mask(lat_lens)[:, :, None] * (1 - ctx_mask)

        # ===== 文本长度对齐检查 =====
        if txt_tokens.shape[1] > lat.shape[1]:
            print(f'|Warning: text lengths [{txt_tokens.shape[1]}] > wav latent [{lat.shape[1]}], clipping...')
            line = txt_tokens[torch.argmax(txt_mask.sum(1))]
            print(self.dit_text_tokenizer.decode(line.detach().cpu().numpy().tolist()))

            line_idxs = np.argsort(txt_lens.cpu().numpy()).tolist()
            for line_idx in reversed(line_idxs):
                if txt_lens[line_idx] > lat.shape[1]:
                    loss_mask[line_idx] = 0.0
                else:
                    break

            txt_tokens = txt_tokens[:, :lat.shape[1]]
            txt_mask = txt_mask[:, :lat.shape[1]]
            txt_lens = txt_mask.sum(1).int()

        # ========= 任务划分：prompt 组 vs ctx 组 =========
        # 约定：ctx 组 == audio prompt（保留原 caption）
        prompt_ratio = float(hparams.get('prompt_ratio', 0.8))
        prompt_ratio = min(max(prompt_ratio, 0.0), 1.0)
        n_prompt = int(round(bsz * prompt_ratio))
        perm = torch.randperm(bsz).tolist()
        prompt_indices = perm[:n_prompt]
        ctx_indices = perm[n_prompt:]
        if len(prompt_indices) == 0:
            prompt_indices = list(range(bsz))
            ctx_indices = []
        prompt_set = set(prompt_indices)

        # prompt 组不使用上下文（沿用原逻辑）
        if len(prompt_indices) > 0:
            ctx_mask[prompt_indices] = 0

        # ===== 逐样本构造 caption =====
        captions = []
        for i in range(bsz):
            cap_ds = (cap_from_ds[i] or '').strip()

            if i in prompt_set:
                # prompt 组：仅拼 global + local（不拼接 cap_ds）
                gp = (global_prompts[i] or '').strip()
                gp = f"<GPROMPT>{gp}</GPROMPT>" if gp != '' else ''
                lp = local_prompts[i]
                if isinstance(lp, str) and lp:
                    # 将<tag>标准化为<TAG>，以配合 tokenizer 的 special tokens
                    lp = lp.replace('<tag>', '<TAG>').replace('</tag>', '</TAG>')
                else:
                    lp = ''
                final_cap = " ".join([x for x in [gp, lp] if x]).strip()
            else:
                # ctx 组（= audio prompt）：严格保留原始 caption
                final_cap = cap_ds

            captions.append(final_cap)

        # ===== caption 编码（返回 emb / mask / text_mark）=====
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                caption_embs, caption_mask, caption_text_mark = self.run_caption_encoder(captions, device)
                caption_embs = caption_embs * caption_mask[..., None]
                caption_lens = caption_mask.sum(-1)

        # ========= CFG masks =========
        lat_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        lat_cfg_mask = (lat_cfg_mask < hparams.get('lat_cfg_prob', 0.15)).long()
        lat_ctx = (lat_ctx * ctx_mask * (1 - lat_cfg_mask)[:, :, None])

        txt_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        txt_cfg_mask = (txt_cfg_mask < hparams.get('txt_cfg_prob', 0.15)).long()
        txt_tokens = txt_tokens * (1 - txt_cfg_mask) + self.cfg_mask_text_token * txt_cfg_mask

        caption_cfg_mask = torch.rand_like(caption_embs[:, 0].float())[:, None]
        caption_cfg_mask = (caption_cfg_mask < hparams.get('caption_cfg_prob', 0.15)).long()
        caption_embs = caption_embs * (1 - caption_cfg_mask)

        if not hasattr(self, 'cfg_mask_token_phone'):
            self.cfg_mask_token_phone = 302 - 1
        if not hasattr(self, 'cfg_mask_token_tone'):
            self.cfg_mask_token_tone = 32 - 1
        no_ph_mask = ph_tokens[:, 0] == self.cfg_mask_token_phone
        if no_ph_mask.float().mean() < 0.15:
            ph_cfg_mask = torch.rand_like(ph_tokens[:, 0].float())[:, None]
            ph_cfg_mask = (ph_cfg_mask < hparams.get('ph_cfg_prob', 0.15)).long()
            ph_tokens = ph_tokens * (1 - ph_cfg_mask) + self.cfg_mask_token_phone * ph_cfg_mask
            tone_tokens = tone_tokens * (1 - ph_cfg_mask) + self.cfg_mask_token_tone * ph_cfg_mask

        inputs = {
            "phone": ph_tokens,
            "tone": tone_tokens,
            "txt_tokens": txt_tokens.long(),
            "txt_lens": txt_lens,
            "txt_mask": txt_mask,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": lat_ctx,
            "ctx_mask": ctx_mask,
            "caption_emb": caption_embs,
            "caption_lens": caption_lens,           # [B]
            "caption_text_mark": caption_text_mark, # LongTensor[B, T]
            "vad_mask": sample['vad_mask'] if hparams['add_vad_mask'] else None,
            "mel2ph": mel2ph,
        }
        if hparams.get('use_sparse_dur', False):
            inputs['mel2ph_sparse'] = sample['mel2ph_sparse']

        if not infer:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                model_outputs, target = self.dit(inputs)

            loss = F.mse_loss(model_outputs.float(), target.float(), reduction='none')
            loss = (loss * loss_mask).sum() / loss_mask.sum() / target.shape[-1]
            losses_out['diff_loss'] = loss
            losses_out['bs'] = loss_mask.shape[0]
            losses_out['ntokens'] = sum(lat_lens)

            # 监控项
            losses_out['monitor/orig_loss'] = loss.detach()
            losses_out['monitor/pred'] = (model_outputs.detach() * loss_mask).sum() / loss_mask.sum()
            losses_out['monitor/tgt'] = (target * loss_mask).sum() / loss_mask.sum()
            losses_out['monitor/loss_mask'] = loss_mask.sum()
            losses_out['monitor/ctx_mask'] = ctx_mask.sum()

            if loss_mask.sum() <= 3:
                if self.trainer.proc_rank_local == 0:
                    print(f"\n| CRITICAL: loss_mask.sum() = {loss_mask.sum()}. Resetting loss.")
                losses_out['diff_loss'] = 0.0

            return losses_out, model_out
        else:
            return losses_out, model_out
