import os
import random
import re

from attrdict import AttrDict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np

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

from modules.tts.scriptspeech.build_model_utils import build_vae, build_qwen3, shard_model_in_node, DiTBuildModelMixinV2
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin
from tasks.tts.scriptspeech_task import ScriptSpeechBaseTask
import json
import time
import math
from pathlib import Path
import torch.distributed as dist


class PromptAudioTask(DiTBuildModelMixinV2, ScriptSpeechBaseTask):
    def build_model(self):
        """
        1) 构建模型与 VAE；
        2) 校验并缓存 caption 特殊 token；
        3) 不再在这里立即构建 EMA（避免与后续 FSDP 分片形状不一致）。
        改为延迟初始化：见 _lazy_init_ema() & on_after_optimization()。
        """
        self._build_model()
        self.vae.to(self.trainer.device)

        # === 确保 caption tokenizer 的特殊标签存在，并缓存其 id ===
        self._ensure_caption_special_tokens()

        # === EMA 延迟初始化所需配置 ===
        self.model_ema = None
        self._ema_params = None          # 固定 EMA 用到的参数列表（保持顺序与对象恒定）
        if hparams.get('use_ema', False):
            self._ema_cfg = dict(
                decay=hparams.get('ema_decay', 0.9999),
                update_after_step=hparams.get('ema_update_after_step', 100),
            )
        else:
            self._ema_cfg = None

        # 只把可训练模块交给优化器/训练器管理；EMA 以后再创建
        return {'trainable': [self.dit], 'others': []}


    def _lazy_init_ema(self):
        """
        在 FSDP 包装完成之后再创建 EMA。
        - 只在第一次需要时创建一次；
        - 固定住一份参数对象列表，以保证与 shadow_params 的一一对应关系；
        - 注意：不要过滤掉 numel()==0 的 shard（FSDP 某些 rank 可能确实为空 shard），
        这样在 step() 时形状仍然一致（[0] 对 [0]），不会再报错。
        """
        if not hparams.get('use_ema', False):
            return
        if self.model_ema is not None:
            return

        base = unwrap_model(self.dit)
        # 固定参数对象列表与顺序；不要做 requires_grad 或 numel 过滤
        self._ema_params = tuple(p for p in base.parameters())

        self.model_ema = EMAModel(
            self._ema_params,
            decay=self._ema_cfg.get('decay', 0.9999),
            update_after_step=self._ema_cfg.get('update_after_step', 100),
        )
        self.model_ema.to(self.trainer.device)
        print_once("| EMA initialized lazily after FSDP wrapping.")


    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        if hparams.get('use_ema', False):
            # 确保在 FSDP 包装完成之后再去创建 EMA
            self._lazy_init_ema()
            # 用初始化时固定下来的同一批参数对象进行更新，避免顺序/形状不一致
            self.model_ema.step(self._ema_params)

    
    def load_model(self):
        """
        当同时提供 hparams['load_ckpt_audio'] 与 hparams['load_ckpt_speech'] 时：
        - feed_forward.*         ← 来自 speech
        - feed_forward_audio.*   ← 来自 audio 的 feed_forward.*（键名映射）
        - 其他参数：若 audio 与 speech 同形状则逐元素平均；否则择其一；都无则保留初始化

        其他情况保持原逻辑，直接调用 load_ckpt。

        说明：
        - 这里不处理/加载 EMA；EMA 在训练阶段用 _lazy_init_ema() 延迟创建，
        以规避 FSDP 分片导致的形状不一致。
        """
        lc = hparams.get('load_ckpt', '')
        audio = hparams.get('load_ckpt_audio', '')
        speech = hparams.get('load_ckpt_speech', '')

        # ---------- 工具函数（仅在本函数内使用） ----------
        def _resolve_single_ckpt(path_or_dir):
            """返回 (checkpoint_obj, ckpt_path_str)。兼容文件与目录，也可直接传入 ckpt dict。"""
            if isinstance(path_or_dir, dict) and ('state_dict' in path_or_dir or 'global_step' in path_or_dir):
                return path_or_dir, '<in-memory>'
            if isinstance(path_or_dir, str) and path_or_dir != '':
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
            return None, str(path_or_dir)

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

        # ---------- 双 ckpt 融合分支 ----------
        if audio and speech:
            model_name = 'dit'

            ckpt_a, path_a = _resolve_single_ckpt(audio)
            ckpt_s, path_s = _resolve_single_ckpt(speech)
            if ckpt_a is None or ckpt_s is None:
                print("| WARN: audio/speech ckpt resolve failed, fallback to default load_ckpt path.")
                # 回退（如果你的 load_ckpt 支持 dict/list，这里也可传递）
                spec = {'audio': audio, 'speech': speech}
                load_ckpt(self.dit, spec, model_name, strict=False, mmap=True)
                return

            sd_all_a = _state_dict_all(ckpt_a)
            sd_all_s = _state_dict_all(ckpt_s)
            sd_a = _extract_sub(sd_all_a, model_name)
            sd_s = _extract_sub(sd_all_s, model_name)

            # 以当前模型的 state_dict 作为模板，仅写入存在且形状匹配的键
            cur_sd = {k.replace('module.', '').replace('_orig_mod.', ''): v
                    for k, v in self.dit.state_dict().items()}
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
                    elif k in sd_a and sd_a[k].shape == cur_param.shape:  # 兜底：audio 中已有同名分路
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
                        # 兜底：尝试 audio 的同名键，否则保持初始化
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

            # 组装“伪”checkpoint，交给原始 load_ckpt 做最终的 load_state_dict 流程
            gstep = max(
                (ckpt_a.get('global_step', 0) if isinstance(ckpt_a, dict) else 0),
                (ckpt_s.get('global_step', 0) if isinstance(ckpt_s, dict) else 0),
            )
            fused_ckpt = {'state_dict': {'dit': merged}, 'global_step': gstep}

            load_ckpt(self.dit, '', 'dit', strict=False, mmap=True, checkpoint=fused_ckpt)

            print(f"| Mixed-loaded 'dit' from audio='{path_a}' & speech='{path_s}'.")
            print(f"| Stats: avg={n_avg}, audio_only={n_audio_only}, speech_only={n_speech_only}, "
                f"ffn_audio={n_ffn_audio}, ffn_speech={n_ffn_speech}, fallback={n_fallback}")
            return

        # ---------- 单一 ckpt 分支（保持原有逻辑） ----------
        lc_is_obj = isinstance(lc, (dict, list, tuple))
        if lc_is_obj:
            load_ckpt(self.dit, lc, 'dit', strict=False, mmap=True)
        elif isinstance(lc, str) and lc != '':
            load_ckpt(self.dit, lc, 'dit', strict=False, mmap=True)


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
        # optimizer = AdamW(unwrap_model(self.dit).parameters(), **self.config.optimizer)
        
        decay_params = []
        no_decay_params = []
        for name, param in unwrap_model(self.dit).named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 1 or "bias" in name or "norm" in name:
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

    def fsdp_optm2model(self):
        return [self.dit]

    def fsdp_wrap_policy(self):
        from torch.nn import Linear, Sequential, Conv1d, Conv2d, Embedding
        import modules.flow_matching.llama
        import modules.asr.llama.llama_seq2seq
        import modules.tts.llama_dit.llama_ca

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                # Linear, Sequential, Conv1d, Conv2d, Embedding,
                modules.flow_matching.llama.TransformerBlock,
                modules.asr.llama.llama_seq2seq.DecoderBlock,
                modules.asr.llama.llama_seq2seq.EncoderBlock,
                modules.tts.llama_dit.llama_ca.TransformerBlock,
                get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.1:
            kill_void()
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'diff_loss': 1.0,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])

        return total_loss, loss_output

    # ========================= 新增：caption 标注支持 =========================

    def _ensure_caption_special_tokens(self):
        """
        确保 caption_tokenizer 有我们要用的特殊 tag；并缓存它们的 id。
        不在此处修改词表，仅转换已有的 token 到 id；若未训练过相应特殊 token，
        convert_tokens_to_ids 可能返回未识别的 id（由上游 tokenizer 决定）。
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
        将带 tag 的 caption token 序列转为 mark：
        0 = 其它/填充
        1 = 文本（仅 <S1>…</S1> 内）
        2 = 音效（<Audio>…</Audio>）
        3 = BGM（<BGM>…</BGM>）

        规则：
        - <S1>…</S1> 内标 1
        - <GPROMPT>…</GPROMPT> 与 <TAG>…</TAG> 内强制标 0（不继承外层状态）
        - <Audio>…</Audio> 标 2
        - <BGM>…</BGM> 标 3
        - 所有起止 tag token 本身标 0
        """
        ids = self._caption_special_ids

        # 进入这些 tag 时切换到对应 label
        start2label = {
            ids["S1_START"]: 1,
            ids["AUDIO_START"]: 2,
            ids["BGM_START"]: 3,
        }
        # 进入这些 tag 时强制切到 0（不继承外层状态）
        start_zero = {ids["GP_START"], ids["TAG_START"]}

        endset = {
            ids["S1_END"],
            ids["AUDIO_END"],
            ids["BGM_END"],
            ids["GP_END"],
            ids["TAG_END"],
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
                    mark[b, t] = 0
                    continue

                if tok in start_zero:
                    stack.append(cur)
                    cur = 0  # 强制为 0，不继承外层
                    mark[b, t] = 0
                    continue

                if tok in endset:
                    mark[b, t] = 0
                    cur = stack.pop() if stack else 0
                    continue

                # 普通内容：按当前段落类型打标
                mark[b, t] = cur

        # 去掉 padding 部分
        return mark * attention_mask.long()


    def run_caption_encoder(self, captions, device):
        """
        与数据集/模型接口对齐：返回 (encoder_hidden_states, attention_mask, caption_text_mark)。
        其中 caption_text_mark 为多类别标注：文本=1、音效=2、BGM=3（其它/填充=0）。
        """
        inputs = self.caption_tokenizer(
            captions,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)            # [B, T]
        attention_masks = inputs.attention_mask.to(device) # [B, T]

        encoder_hidden_states = self.caption_encoder(
            input_ids, return_dict=False,
            attention_mask=attention_masks,
        )[0]

        caption_text_mark = self._build_caption_mark_from_ids(input_ids, attention_masks)
        return encoder_hidden_states, attention_masks, caption_text_mark

    def run_model(self, sample, infer=False, infer_steps=None):
        """
        精简版 run_model：
        - 不使用 global/local prompt，caption 直接来自 sample['caption']。
        - 不使用 audio ctx（ctx_wavs/ctx_mask），以零张量占位；loss_mask 覆盖全部有效时长。
        - 保留 caption 的多类别标注：文本=1、音效=2、BGM=3（其它/填充=0）。
        - 其他编码/CFG/对齐/损失逻辑与原版保持一致。
        """
        model_out = {}
        losses_out = {}
        if infer:
            return losses_out, model_out
        if 'wavs' not in sample:
            return losses_out, model_out

        # ---------- 取样本字段 ----------
        wavs = sample["wavs"].float()
        wav_lengths = sample["wav_lengths"]
        text = sample['text']
        ph_tokens = sample["ph_tokens"]
        tone_tokens = sample["tone"]
        mel2ph = sample['mel2ph']

        bsz, device = wavs.shape[0], wavs.device
        lat_lens = wav_lengths // hparams['hop_size'] // hparams['vae_stride']

        # 英文调类规范（沿用原逻辑）
        en_tone_idx = ~((tone_tokens == 4) | ((11 <= tone_tokens) & (tone_tokens <= 15)) | (tone_tokens == 0))
        tone_tokens[en_tone_idx] = 3

        # ---------- Audio 编码 ----------
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                lat = self.vae.encode_latent(wavs)  # [B, T, D]

        # ---------- 不使用 ctx，占位为 0 ----------
        # ctx_mask: [B, T, 1]，全 0；lat_ctx 与 lat 同形，全部为 0
        ctx_mask = torch.zeros((bsz, lat.size(1), 1), device=device, dtype=lat.dtype)
        lat_ctx = torch.zeros_like(lat)

        # 先算 loss_mask（保持与原逻辑一致，后续若文本过长会按样本清零）
        loss_mask = sequence_mask(lat_lens).to(lat.dtype)[:, :, None] * (1.0 - ctx_mask)

        # ---------- 文本 Tokenize（仅 text prompt 控制） ----------
        text_inputs = self.dit_text_tokenizer(text, padding=True, return_tensors="pt").to(device)
        txt_tokens = text_inputs['input_ids']   # [B, T_txt]
        txt_mask = text_inputs['attention_mask'].bool()
        txt_tokens[~txt_mask] = self.cfg_mask_text_token
        txt_lens = txt_mask.int().sum(1)

        # 文本长度对齐（若文本超过 latent 长度，裁剪并将超长样本的 loss_mask 清零）
        if txt_tokens.shape[1] > lat.shape[1]:
            print(f'|Warning: text lengths [{txt_tokens.shape[1]}] > wav latent [{lat.shape[1]}], clipping...')
            line = txt_tokens[torch.argmax(txt_mask.sum(1))]
            print(self.dit_text_tokenizer.decode(line.detach().cpu().numpy().tolist()))

            line_idxs = np.argsort(txt_lens.cpu().numpy()).tolist()
            for line_idx in reversed(line_idxs):
                if txt_lens[line_idx] > lat.shape[1]:
                    loss_mask[line_idx] = 0.0  # 该样本不参与损失
                else:
                    break

            txt_tokens = txt_tokens[:, :lat.shape[1]]
            txt_mask = txt_mask[:, :lat.shape[1]]
            txt_lens = txt_mask.sum(1).int()

        # ---------- 读取/规范化 caption 并编码（保留多类别标注） ----------
        cap_from_ds = sample['caption'] if 'caption' in sample else [''] * bsz

        # 轻量大小写规范化：将常见小写 tag 规范到 tokenizer 的特殊 token，以便正确打标
        def _norm_caption(s: str) -> str:
            s = (s or '').strip()
            if not s:
                return s
            s = s.replace('<tag>', '<TAG>').replace('</tag>', '</TAG>')
            s = s.replace('<audio>', '<Audio>').replace('</audio>', '</Audio>')
            s = s.replace('<bgm>', '<BGM>').replace('</bgm>', '</BGM>')
            # 允许用户数据里偶尔出现小写 gprompt
            s = s.replace('<gprompt>', '<GPROMPT>').replace('</gprompt>', '</GPROMPT>')
            return s

        captions = [_norm_caption(c) for c in cap_from_ds]

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                # 返回：encoder_hidden_states, attention_mask, caption_text_mark(0/1/2/3)
                caption_embs, caption_mask, caption_text_mark = self.run_caption_encoder(captions, device)
                caption_embs = caption_embs * caption_mask[..., None]
                caption_lens = caption_mask.sum(-1)

        # ---------- CFG（与原逻辑一致） ----------
        lat_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        lat_cfg_mask = (lat_cfg_mask < hparams.get('lat_cfg_prob', 0.15)).long()
        lat_ctx = lat_ctx * ctx_mask * (1 - lat_cfg_mask)[:, :, None]

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

        # ---------- 组装模型输入（与下游一致） ----------
        inputs = {
            "phone": ph_tokens,
            "tone": tone_tokens,
            "txt_tokens": txt_tokens.long(),
            "txt_lens": txt_lens,
            "txt_mask": txt_mask,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": lat_ctx,             # 保留接口：此处为 0
            "ctx_mask": ctx_mask,           # 保留接口：此处为 0
            "caption_emb": caption_embs,
            "caption_lens": caption_lens,           # [B]
            "caption_text_mark": caption_text_mark, # LongTensor[B, T]，文本=1/音效=2/BGM=3
            "vad_mask": sample['vad_mask'] if hparams['add_vad_mask'] else None,
            "mel2ph": mel2ph,
        }
        if hparams.get('use_sparse_dur', False):
            inputs['mel2ph_sparse'] = sample['mel2ph_sparse']

        # ---------- 前向与损失 ----------
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            model_outputs, target = self.dit(inputs)

        loss = F.mse_loss(model_outputs.float(), target.float(), reduction='none')
        loss = (loss * loss_mask).sum() / loss_mask.sum() / target.shape[-1]

        # 统计项
        losses_out['diff_loss'] = loss
        losses_out['bs'] = loss_mask.shape[0]
        losses_out['ntokens'] = sum(lat_lens)
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
