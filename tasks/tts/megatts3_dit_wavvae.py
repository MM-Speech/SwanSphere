import random

import torch
import torch.nn.functional as F
import torchdiffeq
from attrdict import AttrDict
from torch.nn import Linear, Sequential, Conv1d, Conv2d
from torch.optim import AdamW
from transformers import Wav2Vec2Model

from modules.flow_matching.logitnormal import LogitNormalTrainingTimesteps
from modules.flow_matching.vp_cfm import ConditionalFlowMatcher
from modules.tts.llama_dit.llama import TransformerBlock, FeedForward, Attention
from tasks.tts.dataset_utils.dataset_mixin import TTSDatasetMixin
from utils.commons.base_task import BaseTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.import_utils import import_module_bystr
from utils.commons.hparams import hparams, set_hparams
from utils.nn.schedulers import WarmupSchedule, CosineSchedule
from utils.nn.seq_utils import sequence_mask, weights_nonzero_speech


class MegaTTS3DiTTask(TTSDatasetMixin, BaseTask):
    def __init__(self):
        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        self.val_dataset_cls = import_module_bystr(hparams['val_dataset_cls'])
        self.processer_fn = import_module_bystr(hparams['processer_fn'])
        self.build_fast_dataloader = import_module_bystr(hparams['build_fast_dataloader'])
        self.hparams = hparams
        self.config = AttrDict(hparams)

        self.cfg_mask_token_phone = 302 - 1
        self.cfg_mask_token_tone = 32 - 1
        super().__init__()

    def build_model(self):
        self.build_vae()
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        self.timestep_sampler = LogitNormalTrainingTimesteps()

        from modules.tts.llama_dit.speechdit import Diffusion, ModelArgs
        config = ModelArgs()
        config.target_type = 'epsilon' if hparams.get('use_ddpm', False) else 'velocity'
        config.target_type = 'vector_field' if hparams.get('use_vpcfm', False) else config.target_type
        config.use_expand_ph = hparams.get('use_expand_ph', False)
        config.zero_xt_prompt = self.zero_xt_prompt = hparams.get('zero_xt_prompt', False)
        config.do_checkpoint = hparams.get('do_checkpoint', False)
        if hparams.get('use_small_model', False):
            config.encoder_n_layers = 12
            config.encoder_n_heads = 12
            config.encoder_dim = 768 
        if hparams.get('use_dit_1b', False):
            config.encoder_n_layers = 28
            config.encoder_n_heads = 16
            config.encoder_dim = 1536 
        config.max_seq_len = 16384
        config.in_channels = config.out_channels = hparams['latent_dim']
        self.dit = Diffusion(config)
        self.vae_stride = hparams.get('vae_stride', 8)

        return {'trainable': [self.dit], 'others': [self.vae]}

    def build_vae(self):
        self.hp_vae = set_hparams(f'/mnt/bn/sa-ag-data/liruiqi/code/MegaHuman/checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4/config.yaml', global_hparams=False)
        from modules.tts.wavvae.decoder.wavvae_v3 import WavVAE_V3
        self.vae = WavVAE_V3(hparams=self.hp_vae)
        load_ckpt(self.vae, f'/mnt/bn/sa-ag-data/liruiqi/code/MegaHuman/checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4', 'model_gen', strict=True)
        self.vae.eval()

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.dit, hparams['load_ckpt'], 'dit', strict=False)

    def build_optimizer(self):
        optimizer = AdamW(self.dit.parameters(), **self.config.optimizer)
        return optimizer

    def build_scheduler(self, optimizer):
        if hparams.get('warmup_updates', 5000) <= 10:
            return None
        return CosineSchedule(
            optimizer, hparams['optimizer']['lr'], hparams.get('warmup_updates', 5000), total_updates=1000000)

    def fsdp_optm2model(self):
        return [self.dit]

    def fsdp_wrap_policy(self):
        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            dit_blocks = (
                # BaseTransformerLayer,
                # SelfAttention,
                # LayerNorm,
                # MLP,
                Linear,
                Sequential,
                Conv1d,
                Conv2d
            )
            return recurse or isinstance(module, dit_blocks)

        return custom_auto_wrap_policy

    ##########################
    # training and validation
    ##########################

    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'diff_loss': 1.0,
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if
                          isinstance(v, torch.Tensor) and v.requires_grad])

        return total_loss, loss_output

    def mse_loss(self, decoder_output, target, *args, **kwargs):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        mse_loss = F.mse_loss(decoder_output, target, reduction='none')
        weights = weights_nonzero_speech(target)
        mse_loss = (mse_loss * weights).sum() / weights.sum()
        return mse_loss

    def run_model(self, sample, infer=False, infer_steps=None):
        """
        render or train on a single-frame
        :param sample: a batch of data
        :param infer: bool, run in infer mode
        :return: losses (if not infer), model_out
        """
        model_out = {}
        losses_out = {}

        wavs = sample["wavs"].float()
        ctx_wavs = sample["ctx_wavs"].float()
        txt_tokens = sample["txt_tokens"]  # [B, T_t]
        tone_tokens = sample["tone"]  # [B, T_t]
        txt_lens = sample["txt_lengths"]  # [B, T_t]
        ctx_mask = sample["ctx_mask"]
        mel2ph = sample['mel2ph']
        lat_lens = ((mel2ph > 0).sum(-1) // hparams['vae_stride']).long()

        """ Disable the English tone (set them to 3)"""
        en_tone_idx = ~((sample["tone"] == 4) | ( (11 <= sample["tone"]) & (sample["tone"] <= 15)) | (sample["tone"] == 0))
        sample["tone"][en_tone_idx] = 3
    
        with torch.inference_mode():
            lat = self.vae.encode_latent(wavs)
            lat_ctx = self.vae.encode_latent(ctx_wavs)
            lat_ctx = torch.nn.functional.pad(lat_ctx, (0,0,0,lat.size(1)-lat_ctx.size(1)), mode='constant', value=0)
        
        # CFG Mask
        lat_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        lat_cfg_mask = (lat_cfg_mask < 0.15).long()
        txt_cfg_mask = torch.rand_like(txt_tokens[:, 0].float())[:, None]
        txt_cfg_mask = (txt_cfg_mask < 0.15).long()
        txt_tokens = txt_tokens * (1 - txt_cfg_mask) + self.cfg_mask_token_phone * txt_cfg_mask
        tone_tokens = tone_tokens * (1 - txt_cfg_mask) + self.cfg_mask_token_tone * txt_cfg_mask
        loss_mask = sequence_mask(lat_lens)[:, :, None] * (1-ctx_mask)
        text_mel_mask = sequence_mask(lat_lens)

        inputs = {
            "phone": txt_tokens,
            "tone": tone_tokens,
            "txt_lens": txt_lens,
            "lat": lat,
            "lat_lens": lat_lens,
            "lat_ctx": (lat_ctx * ctx_mask * (1 - lat_cfg_mask)[:, :, None]),
            "ctx_mask": ctx_mask,
            "text_mel_mask": text_mel_mask,
            "mel2ph": mel2ph,
        }

        if not infer:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                model_outputs, target = self.dit(inputs)

            loss = F.mse_loss(model_outputs.float(), target.float(), reduction='none')
            loss = (loss * loss_mask).sum() / loss_mask.sum() / target.shape[-1]
            losses_out['diff_loss'] = loss
            losses_out['bs'] = loss_mask.shape[0]
            losses_out['ntokens'] = sum(lat_lens)
            return losses_out, model_out
        else:
            return losses_out, model_out

    @torch.no_grad()
    def inference(self, inputs, local_cond, lengths, timesteps=20, cfg_w=1.0, **kwargs):
        ctx_mask = inputs['ctx_mask']
        _, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
        if cfg_w != 1:
            traj = torchdiffeq.odeint(
                lambda t, x: self.dit(x * (1 - ctx_mask), (t[None] * 1000).int(),
                                      concat_cond=local_cond, img=inputs.get('images'), seqlen=lengths),
                torch.randn([1, frm_len, self.config.model_config.network_config.params.out_channels], device=device),
                torch.linspace(0, 1, timesteps + 1, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
        else:
            local_cond = local_cond[:1]
            ctx_mask = ctx_mask[:1]
            traj = torchdiffeq.odeint(
                lambda t, x: self.dit(x * (1 - ctx_mask), (t[None] * 1000).int(),
                                      concat_cond=local_cond, img=inputs.get('images')),
                torch.randn([1, frm_len, self.config.model_config.network_config.params.out_channels], device=device),
                torch.linspace(0, 1, timesteps + 1, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
        x = traj[-1]

        return x

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        infer_steps = self.hparams.get('infer_steps', 12)
        outputs = self._validation_step(sample, batch_idx, infer_steps)
        return outputs

    def _validation_step(self, sample, batch_idx, infer_steps):
        outputs = {}
        if self.trainer.proc_rank == 0:
            # self.vae.eval()
            # with torch.inference_mode():
            #     with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            #         lat = self.vae.get_latent(sample["mels"])
            #         lat_lens = latent_lengths.clamp(max=lat.size(1))
            # mel = self.vae.decode(lat)
            pass
            # outputs['losses'], _ = self.run_model(sample)
            # _, model_out = self.run_model(sample, infer=True, infer_steps=infer_steps)
            # outputs = tensors_to_scalars(outputs)
            # output_ldm = model_out['ldm_out']
            # T = output_ldm.shape[1]
            # ldm = sample['kps'][:, :T]  # [B, T, nkp, kp_dim] [0, 1]
            # B, T, nkp, kp_dim = ldm.shape
            # output_ldm = self.denormalize_ldm(output_ldm)
            # recon_ldm = model_out['recon_ldm']
            # recon_ldm = self.denormalize_ldm(recon_ldm)

            # results_dir = f"{hparams['work_dir']}/results/{self.global_step}_infersteps{infer_steps}_cfg{hparams['cfg_w']}"
            # os.makedirs(results_dir, exist_ok=True)
            # n_ctx = model_out['ctx_mask'][0, :, 0].sum().long().item()
            # writer_kp = imageio.get_writer(f"{results_dir}/{batch_idx:06d}_kp.sil.mp4", fps=25)
            # writer_gt = imageio.get_writer(f"{results_dir}/{batch_idx:06d}_gt.sil.mp4", fps=25)
            # writer_pred = imageio.get_writer(f"{results_dir}/{batch_idx:06d}_pred.sil.mp4", fps=25)
            # for i in range(T):
            #     img = self.draw_ldm(recon_ldm[0, i])
            #     writer_gt.append_data(img)
            #     img = self.draw_ldm(ldm[0, i])
            #     writer_kp.append_data(img)
            #     if i < n_ctx:
            #         writer_pred.append_data(img)
            #     else:
            #         img = self.draw_ldm(
            #             output_ldm[0, i], color=(255, 255, 0),
            #         )
            #         writer_pred.append_data(img)
            # writer_gt.close()
            # writer_kp.close()
            # writer_pred.close()
        return outputs

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        infer_steps = hparams['infer_steps']
        return self._validation_step(sample, batch_idx, infer_steps)
