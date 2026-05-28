import os
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

from omegaconf import DictConfig
import torch
from attrdict import AttrDict

from modules.tts.scriptspeech.build_model_utils import build_qwen3

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
    hp_vae = set_hparams(os.path.join(vae_ckpt, 'config.yaml'), global_hparams=False)
    from modules.tts.wavvae.decoder.wavvae_v3 import WavVAE_V3
    vae = WavVAE_V3(hparams=hp_vae)
    load_ckpt(vae, vae_ckpt, 'model_gen', strict=True)
    vae.eval()
    return vae, hp_vae


class DiTARBuildModelMixin:
    def build_model(self):
        self._build_model()
        self.vae.to(self.trainer.device)
        return {'trainable': [self.model], 'others': []}

    def _build_model(self):  # interface to infer
        self.vae, self.hp_vae = build_vae(hparams.get('vae_ckpt'))
        self.ditar_text_tokenizer, self.ditar_vocab_size = self.build_ditar_text_tokenizer()
        self.build_ditar(hparams)
        if hparams.get('use_caption_encoder', False):
            caption_encoder_name = hparams.get('caption_encoder_name')
            self.caption_encoder, self.caption_tokenizer = build_qwen3(hparams, caption_encoder_name)
            self.caption_encoder = self.caption_encoder.model
            self.caption_encoder.requires_grad_(False).eval()
            if hparams.get('use_fsdp', True):
                transformer_cls = get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
                self.caption_encoder = shard_model_in_node(self.caption_encoder, transformer_cls, int(os.environ.get("LOCAL_RANK", 0)))

    def build_ditar(self, hparams):
        from modules.tts.ditar.model import DiTARModel, ModelArgs
        config = ModelArgs()
        config.in_channels = config.out_channels = hparams['latent_dim']
        config.text_vocab_size = self.ditar_vocab_size
        config.caption_dim = hparams.get('caption_encoder_dim', 5120 + 1)
        config.decoder_use_caption = hparams.get('ditar_decoder_use_caption', True)
        config.do_checkpoint = hparams.get('gradient_checkpointing', False)
        config.patch_size = hparams.get('ditar_patch_size', 4)
        config.ctx_n_patches = hparams.get('ditar_ctx_n_patches', 2)
        config.training_patch_keep_ratio = hparams.get('training_patch_keep_ratio', -1.0)
        config.focal_loss_alpha = hparams.get('stop_loss_alpha', 0.99)
        config.focal_loss_gamma = hparams.get('stop_loss_gamma', 1.5)
        self.model = DiTARModel(config)
        self.model.text_tokenizer = self.ditar_text_tokenizer
        self.vae_stride = hparams.get('vae_stride', 8)

        return self.model

    def build_ditar_text_tokenizer(self):
        from transformers import AutoTokenizer, Qwen2Tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        text_tokenizer.add_tokens([
            '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>', 
            '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>'
        ], special_tokens=True)
        vocab_size = len(text_tokenizer)
        return text_tokenizer, vocab_size
    
    
    
class DiTARBuildModelMixinS:
    def build_model(self):
        self._build_model()
        # self.vae.to(self.trainer.device)
        return {'trainable': [self.model], 'others': []}
    
    
    def _build_model(self):  # interface to infer
        # self.vae, self.hp_vae = build_vae(hparams.get('vae_ckpt'))
        # self.ditar_text_tokenizer, self.ditar_vocab_size = self.build_ditar_text_tokenizer()
        self.build_ditar(hparams)
        if hparams.get('use_caption_encoder', False):
            raise NotImplementedError
            # caption_encoder_name = hparams.get('caption_encoder_name')
            # self.caption_encoder, self.caption_tokenizer = build_qwen3(hparams, caption_encoder_name)
            # self.caption_encoder = self.caption_encoder.model
            # self.caption_encoder.requires_grad_(False).eval()
            # if hparams.get('use_fsdp', True):
            #     transformer_cls = get_class_from_module("transformers.models.qwen3.modeling_qwen3", "Qwen3DecoderLayer")
            #     self.caption_encoder = shard_model_in_node(self.caption_encoder, transformer_cls, int(os.environ.get("LOCAL_RANK", 0)))

    
    def build_ditar(self, hparams):
        if hparams.get('model_ver', -1) == 1:
            from modules.tts.ditar.ditar_v2s import DiTARModel, ModelArgs
        elif hparams.get('model_ver', -1) == 2:
            from modules.tts.ditar.ditar_v2s_v2 import DiTARModel, ModelArgs
        else:
            # from modules.tts.ditar.ditar_v2s import DiTARModel, ModelArgs
            raise NotImplementedError
        config = ModelArgs()
        config.in_channels = config.out_channels = hparams['latent_dim']
        # config.text_vocab_size = self.ditar_vocab_size
        # config.caption_dim = hparams.get('caption_encoder_dim', 5120 + 1)
        # config.decoder_use_caption = hparams.get('ditar_decoder_use_caption', True)
        config.do_checkpoint = hparams.get('gradient_checkpointing', False)
        config.patch_size = hparams.get('ditar_patch_size', 4)
        config.ctx_n_patches = hparams.get('ditar_ctx_n_patches', 2)
        config.training_patch_keep_ratio = hparams.get('training_patch_keep_ratio', -1.0)
        config.focal_loss_alpha = hparams.get('stop_loss_alpha', 0.99)
        config.focal_loss_gamma = hparams.get('stop_loss_gamma', 1.5)
        self.model = DiTARModel(config)
        # self.model.text_tokenizer = self.ditar_text_tokenizer
        # self.vae_stride = hparams.get('vae_stride', 8)

        return self.model