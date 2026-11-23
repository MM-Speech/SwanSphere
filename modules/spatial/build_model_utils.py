"""Model construction shared by DiTAR v8 training and inference."""
from utils.commons.hparams import hparams

class DiTARBuildModelMixinS:

    def build_model(self):
        self._build_model()
        return {'trainable': [self.model], 'others': []}

    def _build_model(self):
        self.build_ditar(hparams)
        if hparams.get('use_caption_encoder', False):
            raise NotImplementedError

    def build_ditar(self, hparams):
        model_ver = int(hparams.get('model_ver', -1))
        if model_ver != 8:
            raise ValueError(f'Only DiTAR v8 is supported, got model_ver={model_ver}')
        if hparams.get('ditar_dpo_enabled', False):
            from modules.spatial.ditar_v8_dpo import DiTARModel, ModelArgs
        else:
            from modules.spatial.ditar_v8 import DiTARModel, ModelArgs
        config = ModelArgs()
        config.in_channels = config.out_channels = hparams['latent_dim']
        config.video_input_dim = hparams.get('video_input_dim', 2 * hparams.get('d_clip', 512))
        config.do_checkpoint = hparams.get('gradient_checkpointing', False)
        config.use_caption_encoder = bool(hparams.get('use_caption_encoder', False))
        config.patch_size = hparams.get('ditar_patch_size', 4)
        config.use_lm_audio_history = hparams.get('use_lm_audio_history', True)
        config.latent_frame_rate = float(hparams.get('latent_frame_rate', 21.5))
        config.video_frame_rate = float(hparams.get('video_frame_rate', 4.0))
        config.ctx_n_patches = hparams.get('ditar_ctx_n_patches', 1)
        config.dit_history_embedding_dim = hparams.get('dit_history_embedding_dim', 64)
        config.dit_history_dropout_prob = hparams.get('dit_history_dropout_prob', 0.3)
        config.training_patch_keep_ratio = hparams.get('training_patch_keep_ratio', -1.0)
        config.focal_loss_alpha = hparams.get('stop_loss_alpha', 0.99)
        config.focal_loss_gamma = hparams.get('stop_loss_gamma', 1.5)
        config.cfg_unconditional_dropout = hparams.get('cfg_unconditional_dropout', 0.15)
        config.dit_ctx_corrupt_prob = hparams.get('dit_ctx_corrupt_prob', 0.5)
        config.dit_ctx_noise_ratio_min = hparams.get('dit_ctx_noise_ratio_min', 0.0)
        config.dit_ctx_noise_ratio_max = hparams.get('dit_ctx_noise_ratio_max', 0.25)
        config.dit_ctx_noise_warmup_steps = hparams.get('dit_ctx_noise_warmup_steps', 5000)
        from modules.spatial.audio_semantic_encoder import default_audio_semantic_embedding_dim, normalize_audio_semantic_encoder
        config.use_audio_semantic_supervision = hparams.get('use_audio_semantic_supervision', False)
        config.audio_semantic_encoder = normalize_audio_semantic_encoder(hparams.get('audio_semantic_encoder', 'm2d_clap'))
        config.audio_semantic_embedding_dim = int(hparams.get('audio_semantic_embedding_dim', default_audio_semantic_embedding_dim(config.audio_semantic_encoder)))
        config.use_lm_history_bottleneck = hparams.get('use_lm_history_bottleneck', True)
        config.lm_history_bottleneck_dim = hparams.get('lm_history_bottleneck_dim', 16)
        config.lm_history_noise_prob = hparams.get('lm_history_noise_prob', 0.5)
        config.lm_history_noise_gamma_min = hparams.get('lm_history_noise_gamma_min', 0.0)
        config.lm_history_noise_gamma_max = hparams.get('lm_history_noise_gamma_max', 0.5)
        config.lm_history_noise_infer_gamma = hparams.get('lm_history_noise_infer_gamma', 0.15)
        self.model = DiTARModel(config)
        return self.model
