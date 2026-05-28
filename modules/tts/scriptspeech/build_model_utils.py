import os
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
# from transformers import Qwen2Model, AutoTokenizer
from omegaconf import DictConfig
import torch
from attrdict import AttrDict
import json

from utils.commons.hparams import hparams, set_hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.io import print_once

from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy,BackwardPrefetch, CPUOffload

in_node_device_mesh = None
# local_shard_size = 8
def get_in_node_device_mesh():
    global in_node_device_mesh
    # global local_shard_size
    if in_node_device_mesh == None:
        if 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            world_size = 1
        # assert world_size % local_shard_size == 0
        # in_node_device_mesh = init_device_mesh('cuda', (world_size // local_shard_size, local_shard_size))
        in_node_device_mesh = init_device_mesh('cuda', (1, world_size))
    return in_node_device_mesh


def shard_model_in_node(model: torch.nn.Module, transformer_cls, device) -> FSDP:
    transformer_class = transformer_cls
    layer_wrap = ModuleWrapPolicy([transformer_class,])
    device_mesh = get_in_node_device_mesh()
    sharding_strategy = ShardingStrategy.HYBRID_SHARD
    param_dtype = torch.bfloat16
    print_once(f'using {param_dtype} param_dtype for FSDP for model qwen...')
    model = FSDP(
        model,
        auto_wrap_policy=layer_wrap,
        device_id=device,
        sharding_strategy=sharding_strategy,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=torch.float,
            buffer_dtype=param_dtype,
        ),
        sync_module_states=False,
        limit_all_gathers=True,
        use_orig_params=True,
        device_mesh=device_mesh,
    )
    # FSDP.set_state_dict_type(model, StateDictType.LOCAL_STATE_DICT)
    # why do we need this? Shilong found this would cause error when saving checkpoint? is it ralated to pytorch version?
    #TODO(shuo) why do this sync?
    torch.cuda.synchronize()
    return model

def get_class_from_module(module_name, class_name):
    import importlib
    # Import the module dynamically
    module = importlib.import_module(module_name)
    # Get the class from the module
    cls = getattr(module, class_name)
    return cls


def build_vae(vae_ckpt):
    return None, None
    hp_vae = set_hparams(os.path.join(vae_ckpt, 'config.yaml'), global_hparams=False)
    from modules.tts.wavvae.decoder.wavvae_v3 import WavVAE_V3
    vae = WavVAE_V3(hparams=hp_vae)
    load_ckpt(vae, vae_ckpt, 'model_gen', strict=True)
    vae.eval()
    return vae, hp_vae

def build_audio_tokenizer(audio_tokenizer_type):
    return None, None, None
    if audio_tokenizer_type == 'glm4v':
        from modules.tts.semantic_encoders.glm4_tokenizer.feature_extraction_whisper import WhisperFeatureExtractorV2
        from modules.tts.semantic_encoders.glm4_tokenizer.modeling_whisper import WhisperVQEncoder
        audio_token_feature_extractor = WhisperFeatureExtractorV2.from_pretrained(hparams.get('glm_path', "checkpoints/glm-4-voice-tokenizer"))
        audio_tokenizer = WhisperVQEncoder.from_pretrained(hparams.get('glm_path', "checkpoints/glm-4-voice-tokenizer")).eval()
        for param in audio_tokenizer.parameters():
            param.requires_grad = False
            param.grad = None
        audio_vocab_size = audio_tokenizer.config.quantize_vocab_size + 1     # the last should be padding/mask/cfg mask
        return audio_token_feature_extractor, audio_tokenizer, audio_vocab_size

def build_wavlm(ckpt='pretrained_models/wavlm/WavLM-Large.pt', init_pretrained=True, freeze=True):
    from modules.asr.wavlm.WavLM import WavLM, WavLMConfig
    if init_pretrained:
        checkpoint = torch.load(ckpt)
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])
    else:
        cfg = json.load(open('pretrained_models/wavlm/WavLM-Large-Config.json'))
        cfg = WavLMConfig(cfg)
        model = WavLM(cfg)
    audio_encoder_dim = cfg.encoder_embed_dim
    audio_encoder = model
    audio_encoder_hopsize = 320
    audio_encoder_sample_rate = 16000
    if freeze:
        for param in audio_encoder.parameters():
            param.requires_grad = False
            param.grad = None
    return audio_encoder, audio_encoder_dim, audio_encoder_hopsize, audio_encoder_sample_rate


class DiTBuildModelMixin:
    def build_model(self):  # interface to BaseTask()
        self._build_model()
        if hparams.get('trained_module', 'all') == 'crossatt':
            self.training_crossatt()
        return {'trainable': [self.dit], 'others': [self.vae, self.audio_tokenizer]}
    
    def _build_model(self):  # interface to infer
        self.vae, self.hp_vae = build_vae(hparams.get('vae_ckpt'))
        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()
        self.audio_token_feature_extractor, self.audio_tokenizer, self.audio_vocab_size = build_audio_tokenizer(hparams.get('audio_tokenizer', 'glm4v'))
        self.build_dit(hparams)
        if hparams.get('load_sd_text_encoder', False):
            if hparams.get('model_size', 'base') == 'seedance_7b':
                self.build_sd_text_encoder(hparams['text'])
            elif 'goku' in hparams.get('model_size', 'base'):
                self.build_goku_text_encoder(hparams)

    def training_crossatt(self):
        for param in self.dit.parameters():
            param.requires_grad = False

        # 仅解冻特定模块
        for name, param in self.dit.named_parameters():
            if (
                    'caption_proj' in name or
                    'cross_attention' in name or
                    'cross_attention_norm' in name
            ):
                param.requires_grad = True

    def build_dit(self, hparams):
        if hparams.get('dit_version', 1) == 1:
            from modules.tts.scriptspeech.dit import Diffusion, ModelArgs
        elif hparams.get('dit_version', 1) == 2:
            from modules.tts.scriptspeech.dit_na import Diffusion, ModelArgs
        elif hparams.get('dit_version', 1) == 3:
            from modules.tts.scriptspeech.dit_na_ph import Diffusion, ModelArgs
        elif hparams.get('dit_version', 1) == 4:
            from modules.tts.spatial.dit import Diffusion, ModelArgs
        elif hparams.get('dit_version', 1) == 5:
            from modules.tts.spatial.dit_global_cond import Diffusion, ModelArgs
        elif hparams.get('dit_version', 1) == 6:
            from modules.tts.spatial.dit_v2 import Diffusion, ModelArgs
        elif hparams.get('dit_version', 1) == 7:
            from modules.tts.spatial.dit_v3 import Diffusion, ModelArgs
        else:
            raise NotImplementedError
        print(f'| Building DIT model version {hparams.get("dit_version", 1)}')
        config = ModelArgs()
        config.do_checkpoint = hparams.get('do_checkpoint', False)
        if hparams.get('model_size', 'base') == 'small':
            print('| use small model')
            config.encoder_n_layers = 12
            config.encoder_n_heads = 12
            config.encoder_dim = 768 
        elif hparams.get('model_size', 'base') == '1b':
            print('| use base model')
            config.encoder_n_layers = 28
            config.encoder_n_heads = 16
            config.encoder_dim = 1536
        elif hparams.get('model_size', 'base') == 'seedance_7b':
            print('| use seedance_7b model')
            config.encoder_n_layers = 32
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        elif hparams.get('model_size', 'base') == 'goku_2':
            print('| use goku_2 model')
            config.encoder_n_layers = 16
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        elif hparams.get('model_size', 'base') == 'goku_2_large':
            print('| use goku_2_large model')
            config.encoder_n_layers = 32
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        else:
            print('| use base model')

        config.in_channels = config.out_channels = hparams['latent_dim']
        config.vocab_size = self.dit_vocab_size
        config.audio_tokenizer = hparams.get('audio_tokenizer', 'glm4v')
        config.audio_vocab_size = self.audio_vocab_size
        config.caption_dim = hparams.get('caption_encoder_dim', 5120 + 1)
        # config.cfg_mask_text_token = self.cfg_mask_text_token = self.dit_text_tokenizer.encode('<MASK>')[0]
        # config.cfg_mask_audio_token = self.cfg_mask_audio_token = config.audio_vocab_size - 1
        # config.cfg_mask_text_token = self.cfg_mask_text_token = self.dit_text_tokenizer.encode('<MASK>')[0]
        config.cfg_mask_ph_token = self.cfg_mask_ph_token = 302 - 1
        # config.text_fill_token = self.text_fill_token = self.dit_text_tokenizer.encode('<FILL>')[0]
        config.ph_fill_token = self.ph_fill_token = 302 - 2
        config.use_caption_text_mark = hparams.get('use_caption_text_mark', False)
        config.use_caption_pool_in_adaln = hparams.get('use_caption_pool_in_adaln', False)
        self.dit = Diffusion(config)
        self.dit.text_tokenizer = self.dit_text_tokenizer
        self.vae_stride = hparams.get('vae_stride', 8)
        return self.dit

    def build_dit_text_tokenizer(self):
        return None, None
        from transformers import AutoTokenizer, Qwen2Tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        text_tokenizer.add_tokens([
            '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>', 
            '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>', 
            '<S1>', '</S1>','<S2>', '</S2>', 
            '<Audio>', '</Audio>','<BGM>','</BGM>', 
            '<W>', '</W>', '<FILL>', '<PAD>',
            '<S3>', '</S3>','<S4>', '</S4>', 
            '<|laughter|>', '<|breathe|>', '<|cry|>'
        ], special_tokens=True)
        vocab_size = len(text_tokenizer)
        return text_tokenizer, vocab_size

    def build_sd_text_encoder(self, config):
        # seedance text encoder
        from projects.video_diffusion_v3.models.text.encoder import TextEncoder
        # Create text model.
        config = DictConfig(config)
        dtype = getattr(torch, config.dtype)
        self.sd_text_encoder = TextEncoder(config)
        self.sd_text_encoder.requires_grad_(False).eval()
        self.sd_text_encoder.to(dtype=dtype)

        # Compile text model if needed.
        if config.compile:
            self.sd_text_encoder = torch.compile(self.sd_text_encoder)

        self.sd_text_encoder = self.sd_text_encoder
        return self.sd_text_encoder

    def build_goku_text_encoder(self, config):
        return 
        from transformers import T5EncoderModel, T5Tokenizer
        dtype = torch.bfloat16

        if config['use_qwen']:
            qwen_name = config['pretrained_text_encoder_qwen']

            # 分支1：包含 Qwen2 -> 保持现有逻辑不变
            if 'qwen2' in qwen_name.lower():
                from transformers import Qwen2Model, AutoTokenizer
                self.goku_text_encoder = Qwen2Model.from_pretrained(qwen_name, torch_dtype=dtype)
                self.goku_text_encoder.requires_grad_(False).eval()

                self.goku_tokenizer = AutoTokenizer.from_pretrained(
                    qwen_name,
                    max_length=config['text_max_token_length'],
                    revision=None
                )
                vocal_size = len(self.goku_tokenizer)
                self.goku_special_token_ids = [vocal_size + 11, vocal_size + 12]
                self.goku_tokenizer.add_tokens([
                    '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>',
                    '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>',
                    '<S1>', '</S1>', '<S2>', '</S2>',
                    '<Audio>', '</Audio>', '<BGM>', '</BGM>',
                    '<W>', '</W>', '<FILL>', '<PAD>',
                    '<S3>', '</S3>', '<S4>', '</S4>',
                    '<|laughter|>', '<|breathe|>', '<|cry|>'
                ], special_tokens=True)

                transformer_cls = get_class_from_module(
                    "transformers.models.qwen2.modeling_qwen2",
                    "Qwen2DecoderLayer"
                )
                if int(os.environ.get("WORLD_SIZE", 1)) > 1:
                    self.goku_text_encoder = shard_model_in_node(
                        self.goku_text_encoder,
                        transformer_cls,
                        int(os.environ.get("LOCAL_RANK", 0))
                    )

            # 分支2：包含 Qwen3 -> 使用已有的 build_qwen3 逻辑
            elif 'qwen3' in qwen_name.lower():
                # build_qwen3 内部已根据环境优先选择 FA3/FA2/SDPA，并完成必要的 tokenizer 加 special tokens 与 embedding resize
                lm, tok = build_qwen3(hparams, qwen_name, dtype=dtype)
                # 仅取解码器主体做为 encoder 使用（与其他代码保持一致）
                self.goku_text_encoder = lm.model
                self.goku_text_encoder.requires_grad_(False).eval()
                self.goku_tokenizer = tok

                # 保留原有的 special token 索引占位逻辑
                vocal_size = len(self.goku_tokenizer)
                self.goku_special_token_ids = [vocal_size + 11, vocal_size + 12]

                transformer_cls = get_class_from_module(
                    "transformers.models.qwen3.modeling_qwen3",
                    "Qwen3DecoderLayer"
                )
                if int(os.environ.get("WORLD_SIZE", 1)) > 1:
                    self.goku_text_encoder = shard_model_in_node(
                        self.goku_text_encoder,
                        transformer_cls,
                        int(os.environ.get("LOCAL_RANK", 0))
                    )

            else:
                raise ValueError(
                    f"Unknown Qwen backbone: {qwen_name}. Expect the name to contain 'Qwen2' or 'Qwen3'."
                )

        else:
            # 保持原 T5 逻辑不变
            self.goku_text_encoder = T5EncoderModel.from_pretrained(config['pretrained_text_encoder'], torch_dtype=dtype)
            self.goku_text_encoder.requires_grad_(False).eval()
            self.goku_tokenizer = T5Tokenizer.from_pretrained(
                config['pretrained_text_encoder'],
                max_length=config['text_max_token_length']
            )
            vocal_size = len(self.goku_tokenizer)
            self.goku_special_token_ids = [vocal_size + 11, vocal_size + 12]
            self.goku_tokenizer.add_tokens([
                '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>',
                '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>',
                '<S1>', '</S1>', '<S2>', '</S2>',
                '<Audio>', '</Audio>', '<BGM>', '</BGM>',
                '<W>', '</W>', '<FILL>', '<PAD>',
                '<S3>', '</S3>', '<S4>', '</S4>',
                '<|laughter|>', '<|breathe|>', '<|cry|>'
            ], special_tokens=True)


class DiTBuildModelMixinV2:
    def build_model(self):
        self._build_model()
        self.vae.to(self.trainer.device)
        return {'trainable': [self.dit], 'others': []}

    def _build_model(self):  # interface to infer
        self.vae, self.hp_vae = build_vae(hparams.get('vae_ckpt'))
        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()
        self.build_dit(hparams)
        if hparams.get('use_caption_encoder', False):
            caption_encoder_name = hparams.get('caption_encoder_name')
            self.caption_encoder, self.caption_tokenizer = build_qwen3(hparams, caption_encoder_name)
            self.caption_encoder = self.caption_encoder.model
            self.caption_encoder.requires_grad_(False).eval()
            if hparams.get('use_fsdp', True):
                transformer_cls = get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
                self.caption_encoder = shard_model_in_node(self.caption_encoder, transformer_cls, int(os.environ.get("LOCAL_RANK", 0)))

    def build_dit(self, hparams):
        if hparams.get('dit_version', 'v2') == '2spk':
            from modules.tts.scriptspeech.dit_ph_multi import ModelArgs
        elif hparams.get('dit_version', 'v2') == 'audio':
            from modules.tts.scriptspeech.dit_prompt import ModelArgs
        elif hparams.get('dit_version', 'v2') == 'avgen':
            from modules.tts.scriptspeech.dit_v2 import ModelArgs
        # elif hparams.get('dit_version', 'base') == 3:
        #     from modules.tts.scriptspeech.dit_ph_audio_old import ModelArgs
        # elif hparams.get('dit_version', 'base') == 3.1:
        #     from modules.tts.scriptspeech.dit_ph_audio_qf import ModelArgs
        elif hparams.get('dit_version', 'base') == 'v3':
            from modules.tts.scriptspeech.dit_ph_v3 import ModelArgs
            print_once(f'| Use DiT version v3')
            config = ModelArgs()
        else:
            from modules.tts.scriptspeech.dit_ph import ModelArgs
        config = ModelArgs()
        config.do_checkpoint = hparams.get('do_checkpoint', False)

        if hparams.get('model_size', 'base') == 'small':
            print_once('| use small model')
            config.encoder_n_layers = 12
            config.encoder_n_heads = 12
            config.encoder_dim = 768 
        elif hparams.get('model_size', 'base') == '1b':
            print_once('| use base 1b model')
            config.encoder_n_layers = 28
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        elif hparams.get('model_size', 'base') == 'seedance_7b':
            print_once('| use seedance_7b model')
            config.encoder_n_layers = 32
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        elif hparams.get('model_size', 'base') == 'goku_2':
            print_once('| use goku_2 model')
            config.encoder_n_layers = 16
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        elif hparams.get('model_size', 'base') == 'goku_2_32layer':
            print_once('| use goku_2 model with 32 layers')
            config.encoder_n_layers = 32
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        else:
            print_once('| use base model')
        config.in_channels = config.out_channels = hparams['latent_dim']
        config.vocab_size = self.dit_vocab_size
        config.cfg_mask_text_token = self.cfg_mask_text_token = self.dit_text_tokenizer.encode('<MASK>')[0]
        config.text_inject_method = hparams.get('dit_text_inject_method', 'left-prefill')
        config.add_vad_mask = hparams.get('add_vad_mask', False)
        config.caption_dim = hparams.get('caption_encoder_dim', 5120 + 1)
        config.do_checkpoint = hparams.get('gradient_checkpointing', False)
        config.crossattn_n_layers = hparams.get('caption_crossattn_n_layers', 16)
        config.use_qk_norm = hparams.get('use_qk_norm', False)
        config.use_sparse_dur = hparams.get('use_sparse_dur', False)
        config.use_caption_encoder = hparams.get('use_caption_encoder', False)
        config.use_caption_pool_in_adaln = hparams.get('use_caption_pool_in_adaln', False)
        config.use_caption_text_mark = hparams.get('use_caption_text_mark', False)
        config.use_dynamic_cross_gate = hparams.get('use_dynamic_cross_gate', False)
        config.use_llama = hparams.get('use_llama', True)
        config.drop_st = hparams.get('drop_st',True)
        config.sparse_ph_idx = 302 - 2
        config.sparse_tone_idx = 32 - 2
        if hparams.get('dit_version', 'v2') == '2spk':
            from modules.tts.scriptspeech.dit_ph_multi import Diffusion
        # elif hparams.get('dit_version', 'base') == 3:
        #     from modules.tts.scriptspeech.dit_ph_audio_old import Diffusion
        #     self.dit = Diffusion(config)
        # elif hparams.get('dit_version', 'base') == 3.1:
        #     from modules.tts.scriptspeech.dit_ph_audio_qf import Diffusion
        #     self.dit = Diffusion(config)
        elif hparams.get('dit_version', 'v2') == 'audio':
            from modules.tts.scriptspeech.dit_prompt import Diffusion
        elif hparams.get('dit_version', 'v2') == 'v3':
            from modules.tts.scriptspeech.dit_ph_v3 import Diffusion
        elif hparams.get('dit_version', 'v2') == 'avgen':
            from modules.tts.scriptspeech.dit_v2 import Diffusion
        else:
            from modules.tts.scriptspeech.dit_ph import Diffusion
        self.dit = Diffusion(config)
        self.dit.text_tokenizer = self.dit_text_tokenizer
        self.vae_stride = hparams.get('vae_stride', 8)

        return self.dit

    def build_dit_text_tokenizer(self):
        from transformers import AutoTokenizer, Qwen2Tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained("pretrained_models/Qwen3-0.6B")
        text_tokenizer.add_tokens([
            '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>', 
            '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>', 
            '<S1>', '</S1>','<S2>', '</S2>', 
            '<Audio>', '</Audio>','<BGM>','</BGM>', 
            '<W>', '</W>', '<FILL>', '<PAD>',
            '<S3>', '</S3>','<S4>', '</S4>', 
            '<|laughter|>', '<|breathe|>', '<|cry|>'
        ], special_tokens=True)
        vocab_size = len(text_tokenizer)
        return text_tokenizer, vocab_size
    
    
class DiTBuildModelMixinV4(DiTBuildModelMixinV2):
    def build_dit(self, hparams):
        from modules.tts.scriptspeech.dit_ph_v4 import Diffusion, ModelArgs
        print_once(f'| Use DiT version v4')
        config = ModelArgs()
        if hparams.get('model_size', 'base') == 'small':
            print_once('| use small model')
            config.encoder_n_layers = 12
            config.encoder_n_heads = 12
            config.encoder_dim = 768 
        elif hparams.get('model_size', 'base') == '1b':
            print_once('| use base 1b model')
            config.encoder_n_layers = 28
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        elif hparams.get('model_size', 'base') == 'goku_2_32layer':
            print_once('| use goku_2 model with 32 layers')
            config.encoder_n_layers = 32
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        else:
            print_once('| use base model')
        config.in_channels = config.out_channels = hparams['latent_dim']
        config.vocab_size = self.dit_vocab_size
        config.cfg_mask_text_token = self.cfg_mask_text_token = self.dit_text_tokenizer.encode('<MASK>')[0]
        config.cfg_mask_ph_token = self.cfg_mask_ph_token = 302 - 1
        config.text_fill_token = self.text_fill_token = self.dit_text_tokenizer.encode('<FILL>')[0]
        config.ph_fill_token = self.ph_fill_token = 302 - 2
        config.add_vad_mask = hparams.get('add_vad_mask', False)
        config.caption_dim = hparams.get('caption_encoder_dim', 5120 + 1)
        config.do_checkpoint = hparams.get('gradient_checkpointing', False)
        config.crossattn_n_layers = hparams.get('caption_crossattn_n_layers', 16)
        config.use_qk_norm = hparams.get('use_qk_norm', False)
        config.use_sparse_dur = hparams.get('use_sparse_dur', False)
        config.use_caption_text_mark = hparams.get('use_caption_text_mark', False)
        config.use_caption_pool_in_adaln = hparams.get('use_caption_pool_in_adaln', False)
        config.use_dynamic_cross_gate = hparams.get('use_dynamic_cross_gate', False)
        config.use_dur = hparams.get('use_dur', True)
        config.sparse_ph_idx = 302 - 2
        config.sparse_tone_idx = 32 - 2
        self.dit = Diffusion(config)
        self.dit.text_tokenizer = self.dit_text_tokenizer
        self.vae_stride = hparams.get('vae_stride', 8)

        return self.dit

    def build_dit_text_tokenizer(self):
        from transformers import AutoTokenizer, Qwen2Tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        text_tokenizer.add_tokens([
            '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>', 
            '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>', 
            '<S1>', '</S1>','<S2>', '</S2>', 
            '<Audio>', '</Audio>','<BGM>','</BGM>', 
            '<W>', '</W>', '<FILL>', '<PAD>',
            '<S3>', '</S3>','<S4>', '</S4>', 
            '<|laughter|>', '<|breathe|>', '<|cry|>'
        ], special_tokens=True)
        vocab_size = len(text_tokenizer)
        return text_tokenizer, vocab_size


class DiTBuildModelMixinV5(DiTBuildModelMixinV4):
    def _build_model(self):  # interface to infer
        self.vae, self.hp_vae = build_vae(hparams.get('vae_ckpt'))
        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()
        self.build_dit(hparams)
        if hparams.get('use_caption_encoder', False):
            caption_encoder_name = hparams.get('caption_encoder_name')
            self.caption_encoder, self.caption_tokenizer = build_qwen3(hparams, caption_encoder_name)
            self.caption_encoder = self.caption_encoder.model
            self.caption_encoder.requires_grad_(False).eval()
            if hparams.get('use_fsdp', True):
                transformer_cls = get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
                self.caption_encoder = shard_model_in_node(self.caption_encoder, transformer_cls, int(os.environ.get("LOCAL_RANK", 0)))
        
        self.audio_encoder, _, _, _ = build_wavlm()

    def build_dit(self, hparams):
        from modules.tts.scriptspeech.dit_ph_v5 import Diffusion, ModelArgs
        print_once(f'| Use DiT version v5')
        config = ModelArgs()
        if hparams.get('model_size', 'base') == 'small':
            print_once('| use small model')
            config.encoder_n_layers = 12
            config.encoder_n_heads = 12
            config.encoder_dim = 768 
        elif hparams.get('model_size', 'base') == '1b':
            print_once('| use base 1b model')
            config.encoder_n_layers = 28
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        elif hparams.get('model_size', 'base') == 'goku_2_32layer':
            print_once('| use goku_2 model with 32 layers')
            config.encoder_n_layers = 32
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        else:
            print_once('| use base model')
        config.in_channels = config.out_channels = hparams['latent_dim']
        config.vocab_size = self.dit_vocab_size
        config.cfg_mask_text_token = self.cfg_mask_text_token = self.dit_text_tokenizer.encode('<MASK>')[0]
        config.cfg_mask_ph_token = self.cfg_mask_ph_token = 302 - 1
        config.text_fill_token = self.text_fill_token = self.dit_text_tokenizer.encode('<FILL>')[0]
        config.ph_fill_token = self.ph_fill_token = 302 - 2
        config.add_vad_mask = hparams.get('add_vad_mask', False)
        config.caption_dim = hparams.get('caption_encoder_dim', 5120 + 1)
        config.do_checkpoint = hparams.get('gradient_checkpointing', False)
        config.crossattn_n_layers = hparams.get('caption_crossattn_n_layers', 16)
        config.use_qk_norm = hparams.get('use_qk_norm', False)
        config.use_sparse_dur = hparams.get('use_sparse_dur', False)
        config.use_caption_text_mark = hparams.get('use_caption_text_mark', False)
        config.use_caption_pool_in_adaln = hparams.get('use_caption_pool_in_adaln', False)
        config.use_dynamic_cross_gate = hparams.get('use_dynamic_cross_gate', False)
        config.audio_encoder_dim = 1024
        config.use_dur = hparams.get('use_dur', True)
        config.sparse_ph_idx = 302 - 2
        config.sparse_tone_idx = 32 - 2
        self.dit = Diffusion(config)
        self.dit.text_tokenizer = self.dit_text_tokenizer
        self.vae_stride = hparams.get('vae_stride', 8)

        return self.dit


class SemanticLMBuildModelMixin:
    def build_model(self):
        self._build_model()
        self.audio_tokenizer.to(self.trainer.device)
        return {'trainable': [self.lm], 'others': []}

    def _build_model(self):
        self.audio_token_feature_extractor, self.audio_tokenizer, self.audio_vocab_size = build_audio_tokenizer(hparams.get('audio_tokenizer', 'glm4v'))
        self.lm_text_tokenizer, self.lm_vocab_size = self.build_lm_text_tokenizer(hparams)
        self.build_lm(hparams)
        self.eos_idx = self.padding_idx = self.lm_text_tokenizer.encode('<|endoftext|>')[0]
        self.speech_start_token = self.lm_text_tokenizer.encode('<SpeechToken_0>')[0]

    def build_lm(self, hparams):
        from utils.nn.embedding import resize_embedding_layer
        lm, _ = build_qwen3(hparams, hparams.get('backbone_name', "pretrained_models/Qwen3-0.6B"))
        resize_embedding_layer(lm, self.lm_vocab_size)

        self.lm = lm
        if hparams.get('gradient_checkpointing', False):
            print_once('| Enabling gradient checkpointing...')
            # https://github.com/huggingface/transformers/issues/23018
            self.lm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self.lm_hparams.get('use_ctx_mask'):
            self.mask_proj = torch.nn.Linear(1, 1024)
            torch.nn.init.zeros_(self.mask_proj.weight)
            torch.nn.init.zeros_(self.mask_proj.bias)

        return lm

    def build_lm_text_tokenizer(self, hp=None):
        from transformers import AutoTokenizer, Qwen2Tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained("pretrained_models/Qwen3-0.6B")
        text_tokenizer.add_tokens([
            '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>', 
            '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>', 
            '<S1>', '</S1>','<S2>', '</S2>', 
            '<Audio>', '</Audio>','<BGM>','</BGM>', 
            '<W>', '</W>', '<FILL>', '<PAD>',
            '<S3>', '</S3>','<S4>', '</S4>', 
            '<|laughter|>', '<|breathe|>', '<|cry|>'
        ], special_tokens=True)
        if hp is not None and hp.get('use_ctx_mask', False):
            text_tokenizer.add_tokens(['<S1>', '</S1>'], special_tokens=True)
        text_tokenizer.add_tokens([f'<SpeechToken_{i}>' for i in range(self.audio_vocab_size)], special_tokens=True)
        vocab_size = len(text_tokenizer)
        return text_tokenizer, vocab_size


def build_qwen3(hparams, model_name="Qwen/Qwen3-0.6B", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import importlib
    from utils.nn.embedding import resize_embedding_layer
    if fa3_hopper_installed := importlib.util.find_spec('flash_attn_3') is not None:
        print_once('| use fa3_hopper')
        attn_implementation = 'flash_attention_3'
    elif fa2_installed := importlib.util.find_spec('flash_attn') is not None:
        print_once('| use fa2')
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = 'sdpa'
    backbone_name = model_name
    # if hparams.get('load_ckpt', '') != '' or not hparams.get('init_pretrained', True):
    #     from transformers import Qwen3ForCausalLM, Qwen3Config
    #     lm = Qwen3ForCausalLM._from_config(Qwen3Config.from_pretrained(backbone_name), attn_implementation=attn_implementation, torch_dtype=dtype)
    # else:
    lm = AutoModelForCausalLM.from_pretrained(backbone_name, attn_implementation=attn_implementation, torch_dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(backbone_name, max_length=hparams.get('text_max_token_length', 800), revision=None)
    tokenizer.add_tokens([
        '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>', 
        '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>', 
        '<S1>', '</S1>','<S2>', '</S2>', 
        '<Audio>', '</Audio>','<BGM>','</BGM>', 
        '<W>', '</W>', '<FILL>', '<PAD>',
        '<S3>', '</S3>','<S4>', '</S4>', 
        '<|laughter|>', '<|breathe|>', '<|cry|>'
    ], special_tokens=True)
    resize_embedding_layer(lm, len(tokenizer))

    return lm, tokenizer
