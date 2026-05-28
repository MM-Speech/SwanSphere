import torch
import math
import numpy as np
import math
import json

from torch import nn, sin, pow
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torchaudio import transforms as T
from alias_free_torch import Activation1d
from typing import List, Literal, Dict, Any, Callable
from einops import rearrange

from stable_audio_tools.inference.sampling import sample
from stable_audio_tools.inference.utils import prepare_audio
from stable_audio_tools.models.blocks import SnakeBeta
from stable_audio_tools.models.bottleneck import Bottleneck, DiscreteBottleneck
from stable_audio_tools.models.diffusion import ConditionedDiffusionModel, DAU1DCondWrapper, UNet1DCondWrapper, DiTWrapper
from stable_audio_tools.models.factory import create_pretransform_from_config, create_bottleneck_from_config
from stable_audio_tools.models.pretransforms import Pretransform, AutoencoderPretransform
from stable_audio_tools.models.transformer import ContinuousTransformer, TransformerBlock, RotaryEmbedding

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    
    if antialias:
        act = Activation1d(act)
    
    return act

def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

def unfold_channels_from_batch(x, channels):
    if channels == 1:
        return x.unsqueeze(1)
    x = rearrange(x, '(b c) ... -> b c ...', c = channels)
    return x

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        super().__init__()
        
        self.dilation = dilation

        padding = (dilation * (7-1)) // 2

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation, padding=padding),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        res = x
        
        if self.training:
            x = checkpoint(self.layers, x)
        else:
            x = self.layers(x)

        return x + res

class Transpose(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, **kwargs):
        return rearrange(x, '... a b -> ... b a')

class TAAEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, type = 'encoder', transformer_depth = 3, use_snake = False, sliding_window = [31,32], checkpointing = False, conformer = False, layer_scale = True, use_dilated_conv = False):
        super().__init__()
        if type not in ['encoder', 'decoder']:
            raise ValueError(f"Unknown type {type}. Must be 'encoder' or 'decoder'")
        
        self.checkpointing = checkpointing
        
        transformer_dim = out_channels if type == 'encoder' else in_channels
        transformers = []
        transformers.append(Transpose())

        self.sliding_window = sliding_window

        for _ in range(transformer_depth):
            transformers.append(TransformerBlock(transformer_dim, 
                                                 dim_heads = 128, 
                                                 causal = False, 
                                                 zero_init_branch_outputs = True if not layer_scale else False, 
                                                 remove_norms = False, 
                                                 conformer = conformer, 
                                                 layer_scale = layer_scale, 
                                                 add_rope = True, 
                                                 attn_kwargs={'qk_norm': "ln"}, 
                                                 ff_kwargs={'mult': 4, 'no_bias': False},
                                                 norm_kwargs = {'eps': 1e-2}))
        transformers.append(Transpose())
        transformers = nn.ModuleList(transformers)

        if type == 'encoder':
            layers = []
            if use_dilated_conv:
                layers.append(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=1, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=3, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=in_channels, out_channels=in_channels, dilation=9, use_snake=use_snake))
            layers.append(get_activation("snake" if use_snake else "none", antialias=False, channels=in_channels))
            layers.append(WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)) if stride > 1 else nn.Identity())
            layers.append(transformers)
            self.layers = nn.ModuleList(layers)
        elif type == 'decoder':
            layers = []
            layers.append(transformers)
            layers.append(get_activation("snake" if use_snake else "none", antialias=False, channels=out_channels))
            layers.append(WNConvTranspose1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)) if stride > 1 else nn.Identity())
            if use_dilated_conv:
                layers.append(ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=1, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=3, use_snake=use_snake))
                layers.append(ResidualUnit(in_channels=out_channels, out_channels=out_channels, dilation=9, use_snake=use_snake))
            self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.ModuleList):
                for transformer in layer:
                    if self.checkpointing:
                        x = checkpoint(transformer, x, self_attention_flash_sliding_window = self.sliding_window)
                    else:
                        x = transformer(x, self_attention_flash_sliding_window = self.sliding_window)
            else:
                if self.checkpointing:
                    x = checkpoint(layer, x)
                else:
                    x = layer(x)
        return x

class TAAEEncoder(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 transformer_depths = [3,3,3,3],
                 use_snake=False,
                 sliding_window = [63,64],
                 checkpointing = False,
                 conformer = False,
                 layer_scale = True,
                 use_dilated_conv = False,
                 **kwargs
        ):
        super().__init__()
          
        channel_dims = [c * channels for c in c_mults]
        channel_dims = [channel_dims[0]] + channel_dims

        self.depth = len(c_mults)

        layers = [WNConv1d(in_channels=in_channels, out_channels=channel_dims[0], kernel_size=7, padding=3, bias = True)]

        for i in range(self.depth):
            layers += [TAAEBlock(in_channels=channel_dims[i], out_channels=channel_dims[i+1], stride=strides[i], transformer_depth = transformer_depths[i], use_snake=use_snake, sliding_window = sliding_window, checkpointing = checkpointing, conformer = conformer, layer_scale = layer_scale, use_dilated_conv = use_dilated_conv, **kwargs)]

        layers += [
            get_activation("snake" if use_snake else "none", antialias=False, channels=channel_dims[-1]),
            WNConv1d(in_channels=channel_dims[-1], out_channels=latent_dim, kernel_size=3, padding=1, bias = True)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class TAAEDecoder(nn.Module):
    def __init__(self, 
                 out_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 transformer_depths = [3,3,3,3],
                 use_snake=False,
                 sliding_window = [63,64],
                 checkpointing = False,
                 conformer = False,
                 layer_scale = True,
                 use_dilated_conv = False,
                 **kwargs
        ):
        super().__init__()

        channel_dims = [c * channels for c in c_mults]
        channel_dims = [channel_dims[0]] + channel_dims

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=channel_dims[-1], kernel_size=3, padding=1, bias = True)
        ]
        
        for i in range(self.depth, 0, -1):
            layers += [TAAEBlock(in_channels=channel_dims[i], out_channels=channel_dims[i-1], stride=strides[i-1], type = 'decoder', transformer_depth = transformer_depths[i-1], use_snake=use_snake, sliding_window = sliding_window, checkpointing = checkpointing, conformer = conformer, layer_scale = layer_scale, use_dilated_conv = use_dilated_conv, **kwargs)]  

        layers += [get_activation("snake" if use_snake else "none", antialias=False, channels=channel_dims[0]),
                    WNConv1d(in_channels=channel_dims[0], out_channels=out_channels, kernel_size=7, padding=3, bias = False)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9, use_snake=use_snake),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
        )

    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(in_channels=in_channels,
                        out_channels=out_channels, 
                        kernel_size=2*stride,
                        stride=1,
                        bias=False,
                        padding='same')
            )
        else:
            upsample_layer = WNConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2))

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample_layer,
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)

class OobleckEncoder(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False
        ):
        super().__init__()
        self.in_channels = in_channels
          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
        ]
        
        for i in range(self.depth-1):
            layers += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], use_snake=use_snake)]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            WNConv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, 
                 out_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False,
                 use_nearest_upsample=False,
                 final_tanh=True):
        super().__init__()
        self.out_channels = out_channels

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3),
        ]
        
        for i in range(self.depth-1, 0, -1):
            layers += [DecoderBlock(
                in_channels=c_mults[i]*channels, 
                out_channels=c_mults[i-1]*channels, 
                stride=strides[i-1], 
                use_snake=use_snake, 
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample
                )
            ]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DACEncoderWrapper(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()

        from dac.model.dac import Encoder as DACEncoder

        latent_dim = kwargs.pop("latent_dim", None)

        encoder_out_dim = kwargs["d_model"] * (2 ** len(kwargs["strides"]))
        self.encoder = DACEncoder(d_latent=encoder_out_dim, **kwargs)
        self.latent_dim = latent_dim

        # Latent-dim support was added to DAC after this was first written, and implemented differently, so this is for backwards compatibility
        self.proj_out = nn.Conv1d(self.encoder.enc_dim, latent_dim, kernel_size=1) if latent_dim is not None else nn.Identity()

        if in_channels != 1:
            self.encoder.block[0] = WNConv1d(in_channels, kwargs.get("d_model", 64), kernel_size=7, padding=3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj_out(x)
        return x

class DACDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()

        from dac.model.dac import Decoder as DACDecoder

        self.decoder = DACDecoder(**kwargs, input_channel = latent_dim, d_out=out_channels)

        self.latent_dim = latent_dim

    def forward(self, x):
        return self.decoder(x)

class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck: Bottleneck = None,
        pretransform: Pretransform = None,
        in_channels = None,
        out_channels = None,
        soft_clip = False
    ):
        super().__init__()

        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels

        self.min_length = self.downsampling_ratio

        if in_channels is not None:
            self.in_channels = in_channels

        if out_channels is not None:
            self.out_channels = out_channels

        self.bottleneck = bottleneck

        self.encoder = encoder

        self.decoder = decoder

        self.pretransform = pretransform

        self.soft_clip = soft_clip
 
        self.is_discrete = self.bottleneck is not None and self.bottleneck.is_discrete

    def encode(self, audio, skip_bottleneck: bool = False, return_info=False, skip_pretransform=False, iterate_batch=False, **kwargs):

        info = {}

        if self.pretransform is not None and not skip_pretransform:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    audios = []
                    for i in range(audio.shape[0]):
                        audios.append(self.pretransform.encode(audio[i:i+1]))
                    audio = torch.cat(audios, dim=0)
                else:
                    audio = self.pretransform.encode(audio)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        audios = []
                        for i in range(audio.shape[0]):
                            audios.append(self.pretransform.encode(audio[i:i+1]))
                        audio = torch.cat(audios, dim=0)
                    else:
                        audio = self.pretransform.encode(audio)

        if self.encoder is not None:
            if iterate_batch:
                latents = []
                for i in range(audio.shape[0]):
                    latents.append(self.encoder(audio[i:i+1]))
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.encoder(audio)
        else:
            latents = audio

        info["pre_bottleneck_latents"] = latents

        if self.bottleneck is not None and not skip_bottleneck:
            # TODO: Add iterate batch logic, needs to merge the info dicts
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)

            info.update(bottleneck_info)
        
        if return_info:
            return latents, info

        return latents

    def decode(self, latents, skip_bottleneck: bool = False, iterate_batch=False, **kwargs):

        if self.bottleneck is not None and not skip_bottleneck:
            if iterate_batch:
                decoded = []
                for i in range(latents.shape[0]):
                    decoded.append(self.bottleneck.decode(latents[i:i+1]))
                latents = torch.cat(decoded, dim=0)
            else:
                latents = self.bottleneck.decode(latents)

        if iterate_batch:
            decoded = []
            for i in range(latents.shape[0]):
                decoded.append(self.decoder(latents[i:i+1]))
            decoded = torch.cat(decoded, dim=0)
        else:
            decoded = self.decoder(latents, **kwargs)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    decodeds = []
                    for i in range(decoded.shape[0]):
                        decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                    decoded = torch.cat(decodeds, dim=0)
                else:
                    decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        decodeds = []
                        for i in range(latents.shape[0]):
                            decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                        decoded = torch.cat(decodeds, dim=0)
                    else:
                        decoded = self.pretransform.decode(decoded)

        if self.soft_clip:
            decoded = torch.tanh(decoded)
        
        return decoded
          
    def decode_tokens(self, tokens, **kwargs):
        '''
        Decode discrete tokens to audio
        Only works with discrete autoencoders
        '''

        assert isinstance(self.bottleneck, DiscreteBottleneck), "decode_tokens only works with discrete autoencoders"

        latents = self.bottleneck.decode_tokens(tokens, **kwargs)

        return self.decode(latents, **kwargs)
  
    def preprocess_audio_for_encoder(self, audio, in_sr):
        '''
        Preprocess single audio tensor (Channels x Length) to be compatible with the encoder.
        If the model is mono, stereo audio will be converted to mono.
        Audio will be silence-padded to be a multiple of the model's downsampling ratio.
        Audio will be resampled to the model's sample rate. 
        The output will have batch size 1 and be shape (1 x Channels x Length)
        '''
        return self.preprocess_audio_list_for_encoder([audio], [in_sr])

    def preprocess_audio_list_for_encoder(self, audio_list, in_sr_list):
        '''
        Preprocess a [list] of audio (Channels x Length) into a batch tensor to be compatable with the encoder. 
        The audio in that list can be of different lengths and channels. 
        in_sr can be an integer or list. If it's an integer it will be assumed it is the input sample_rate for every audio.
        All audio will be resampled to the model's sample rate. 
        Audio will be silence-padded to the longest length, and further padded to be a multiple of the model's downsampling ratio. 
        If the model is mono, all audio will be converted to mono. 
        The output will be a tensor of shape (Batch x Channels x Length)
        '''
        batch_size = len(audio_list)
        if isinstance(in_sr_list, int):
            in_sr_list = [in_sr_list]*batch_size
        assert len(in_sr_list) == batch_size, "list of sample rates must be the same length of audio_list"
        new_audio = []
        max_length = 0
        # resample & find the max length
        for i in range(batch_size):
            audio = audio_list[i]
            in_sr = in_sr_list[i]
            if len(audio.shape) == 3 and audio.shape[0] == 1:
                # batchsize 1 was given by accident. Just squeeze it.
                audio = audio.squeeze(0)
            elif len(audio.shape) == 1:
                # Mono signal, channel dimension is missing, unsqueeze it in
                audio = audio.unsqueeze(0)
            assert len(audio.shape)==2, "Audio should be shape (Channels x Length) with no batch dimension" 
            # Resample audio
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate).to(audio.device)
                audio = resample_tf(audio)
            new_audio.append(audio)
            if audio.shape[-1] > max_length:
                max_length = audio.shape[-1]
        # Pad every audio to the same length, multiple of model's downsampling ratio
        padded_audio_length = max_length + (self.min_length - (max_length % self.min_length)) % self.min_length
        for i in range(batch_size):
            # Pad it & if necessary, mixdown/duplicate stereo/mono channels to support model
            new_audio[i] = prepare_audio(new_audio[i], in_sr=in_sr, target_sr=in_sr, target_length=padded_audio_length, 
                target_channels=self.in_channels, device=new_audio[i].device).squeeze(0)
        # convert to tensor 
        return torch.stack(new_audio) 

    def encode_audio(self, audio, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Encode audios into latents. Audios should already be preprocesed by preprocess_audio_for_encoder.
        If chunked is True, split the audio into chunks of a given maximum size chunk_size, with given overlap.
        Overlap and chunk_size params are both measured in number of latents (not audio samples) 
        # and therefore you likely could use the same values with decode_audio. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked output and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked:
            # default behavior. Encode the entire audio in parallel
            return self.encode(audio, **kwargs)
        else:
            # CHUNKED ENCODING
            # samples_per_latent is just the downsampling ratio (which is also the upsampling ratio)
            samples_per_latent = int(self.downsampling_ratio)
            total_size = audio.shape[2] # in samples
            batch_size = audio.shape[0]
            chunk_size *= samples_per_latent # converting metric in latents to samples
            overlap *= samples_per_latent # converting metric in latents to samples
            hop_size = chunk_size - overlap
            chunks = []
            for i in range(0, total_size - chunk_size + 1, hop_size):
                chunk = audio[:,:,i:i+chunk_size]
                chunks.append(chunk)
            if i+chunk_size != total_size:
                # Final chunk
                chunk = audio[:,:,-chunk_size:]
                chunks.append(chunk)
            chunks = torch.stack(chunks)
            num_chunks = chunks.shape[0]
            # Note: y_size might be a different value from the latent length used in diffusion training
            # because we can encode audio of varying lengths
            # However, the audio should've been padded to a multiple of samples_per_latent by now.
            y_size = total_size // samples_per_latent
            # Create an empty latent, we will populate it with chunks as we encode them
            y_final = torch.zeros((batch_size,self.latent_dim,y_size), dtype = chunks.dtype).to(audio.device)
            for i in range(num_chunks):
                x_chunk = chunks[i,:]
                # encode the chunk
                y_chunk = self.encode(x_chunk)
                # figure out where to put the audio along the time domain
                if i == num_chunks-1:
                    # final chunk always goes at the end
                    t_end = y_size
                    t_start = t_end - y_chunk.shape[2]
                else:
                    t_start = i * hop_size // samples_per_latent
                    t_end = t_start + chunk_size // samples_per_latent
                #  remove the edges of the overlaps
                ol = overlap//samples_per_latent//2
                chunk_start = 0
                chunk_end = y_chunk.shape[2]
                if i > 0:
                    # no overlap for the start of the first chunk
                    t_start += ol
                    chunk_start += ol
                if i < num_chunks-1:
                    # no overlap for the end of the last chunk
                    t_end -= ol
                    chunk_end -= ol
                # paste the chunked audio into our y_final output audio
                y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
            return y_final
    
    def decode_audio(self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Decode latents to audio. 
        If chunked is True, split the latents into chunks of a given maximum size chunk_size, with given overlap, both of which are measured in number of latents. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked audio and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked:
            # default behavior. Decode the entire latent in parallel
            return self.decode(latents, **kwargs)
        else:
            # chunked decoding
            hop_size = chunk_size - overlap
            total_size = latents.shape[2]
            batch_size = latents.shape[0]
            chunks = []
            for i in range(0, total_size - chunk_size + 1, hop_size):
                chunk = latents[:,:,i:i+chunk_size]
                chunks.append(chunk)
            if i+chunk_size != total_size:
                # Final chunk
                chunk = latents[:,:,-chunk_size:]
                chunks.append(chunk)
            chunks = torch.stack(chunks)
            num_chunks = chunks.shape[0]
            # samples_per_latent is just the downsampling ratio
            samples_per_latent = int(self.downsampling_ratio)
            # Create an empty waveform, we will populate it with chunks as decode them
            y_size = total_size * samples_per_latent
            y_final = torch.zeros((batch_size,self.out_channels,y_size), dtype = chunks.dtype).to(latents.device)
            for i in range(num_chunks):
                x_chunk = chunks[i,:]
                # decode the chunk
                y_chunk = self.decode(x_chunk)
                # figure out where to put the audio along the time domain
                if i == num_chunks-1:
                    # final chunk always goes at the end
                    t_end = y_size
                    t_start = t_end - y_chunk.shape[2]
                else:
                    t_start = i * hop_size * samples_per_latent
                    t_end = t_start + chunk_size * samples_per_latent
                #  remove the edges of the overlaps
                ol = (overlap//2) * samples_per_latent
                chunk_start = 0
                chunk_end = y_chunk.shape[2]
                if i > 0:
                    # no overlap for the start of the first chunk
                    t_start += ol
                    chunk_start += ol
                if i < num_chunks-1:
                    # no overlap for the end of the last chunk
                    t_end -= ol
                    chunk_end -= ol
                # paste the chunked audio into our y_final output audio
                y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
            return y_final

    def forward(self, audio_data):
        '''新加的方法'''
        latents, info = self.encode(audio_data, return_info=True)
        rec_audio = self.decode(latents)
        
        outputs = {}
        outputs['recon'] = rec_audio
        mean, scale = info["pre_bottleneck_latents"].chunk(2, dim=1)
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        outputs['mu'] = mean
        outputs['logvar'] = logvar
        outputs['kl'] = info['kl']

        return outputs
        

        
    
class DiffusionAutoencoder(AudioAutoencoder):
    def __init__(
        self,
        diffusion: ConditionedDiffusionModel,
        diffusion_downsampling_ratio,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.diffusion = diffusion

        self.min_length = self.downsampling_ratio * diffusion_downsampling_ratio

        if self.encoder is not None:
            # Shrink the initial encoder parameters to avoid saturated latents
            with torch.no_grad():
                for param in self.encoder.parameters():
                    param *= 0.5

    def decode(self, latents, steps=100):

        upsampled_length = latents.shape[2] * self.downsampling_ratio

        if self.bottleneck is not None:
            latents = self.bottleneck.decode(latents)

        if self.decoder is not None:
            latents = self.decode(latents)
    
        # Upsample latents to match diffusion length
        if latents.shape[2] != upsampled_length:
            latents = F.interpolate(latents, size=upsampled_length, mode='nearest')

        noise = torch.randn(latents.shape[0], self.io_channels, upsampled_length, device=latents.device)
        decoded = sample(self.diffusion, noise, steps, 0, input_concat_cond=latents)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    decoded = self.pretransform.decode(decoded)

        return decoded
        
# AE factories

def create_encoder_from_config(encoder_config: Dict[str, Any]):
    encoder_type = encoder_config.get("type", None)
    assert encoder_type is not None, "Encoder type must be specified"

    if encoder_type == "oobleck":
        encoder = OobleckEncoder(
            **encoder_config["config"]
        )
    
    elif encoder_type == "seanet":
        from encodec.modules import SEANetEncoder
        seanet_encoder_config = encoder_config["config"]

        #SEANet encoder expects strides in reverse order
        seanet_encoder_config["ratios"] = list(reversed(seanet_encoder_config.get("ratios", [2, 2, 2, 2, 2])))
        encoder = SEANetEncoder(
            **seanet_encoder_config
        )
    elif encoder_type == "dac":
        dac_config = encoder_config["config"]

        encoder = DACEncoderWrapper(**dac_config)
    elif encoder_type == "local_attn":
        from .local_attention import TransformerEncoder1D

        local_attn_config = encoder_config["config"]

        encoder = TransformerEncoder1D(
            **local_attn_config
        )
    elif encoder_type == "taae":
        encoder = TAAEEncoder(
            **encoder_config["config"]
        )
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")
    
    requires_grad = encoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder

def create_decoder_from_config(decoder_config: Dict[str, Any]):
    decoder_type = decoder_config.get("type", None)
    assert decoder_type is not None, "Decoder type must be specified"

    if decoder_type == "oobleck":
        decoder = OobleckDecoder(
            **decoder_config["config"]
        )
    elif decoder_type == "seanet":
        from encodec.modules import SEANetDecoder

        decoder = SEANetDecoder(
            **decoder_config["config"]
        )
    elif decoder_type == "dac":
        dac_config = decoder_config["config"]

        decoder = DACDecoderWrapper(**dac_config)
    elif decoder_type == "local_attn":
        from .local_attention import TransformerDecoder1D

        local_attn_config = decoder_config["config"]

        decoder = TransformerDecoder1D(
            **local_attn_config
        )
    elif decoder_type == "taae":
        decoder = TAAEDecoder(
            **decoder_config["config"]
        )
    else:
        raise ValueError(f"Unknown decoder type {decoder_type}")
    
    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder

def create_autoencoder_from_config(config: Dict[str, Any]):
    
    ae_config = config["model"]

    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])

    bottleneck = ae_config.get("bottleneck", None)

    latent_dim = ae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = ae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = ae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)

    pretransform = ae_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    soft_clip = ae_config["decoder"].get("soft_clip", False)

    return AudioAutoencoder(
        encoder,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=pretransform,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip
    )

def create_diffAE_from_config(config: Dict[str, Any]):
    
    diffae_config = config["model"]

    if "encoder" in diffae_config:
        encoder = create_encoder_from_config(diffae_config["encoder"])
    else:
        encoder = None

    if "decoder" in diffae_config:
        decoder = create_decoder_from_config(diffae_config["decoder"])
    else:
        decoder = None

    diffusion_model_type = diffae_config["diffusion"]["type"]

    if diffusion_model_type == "DAU1d":
        diffusion = DAU1DCondWrapper(**diffae_config["diffusion"]["config"])
    elif diffusion_model_type == "adp_1d":
        diffusion = UNet1DCondWrapper(**diffae_config["diffusion"]["config"])
    elif diffusion_model_type == "dit":
        diffusion = DiTWrapper(**diffae_config["diffusion"]["config"])

    latent_dim = diffae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = diffae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = diffae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    bottleneck = diffae_config.get("bottleneck", None)

    pretransform = diffae_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    diffusion_downsampling_ratio = None,

    if diffusion_model_type == "DAU1d":
        diffusion_downsampling_ratio = np.prod(diffae_config["diffusion"]["config"]["strides"])
    elif diffusion_model_type == "adp_1d":
        diffusion_downsampling_ratio = np.prod(diffae_config["diffusion"]["config"]["factors"])
    elif diffusion_model_type == "dit":
        diffusion_downsampling_ratio = 1

    return DiffusionAutoencoder(
        encoder=encoder,
        decoder=decoder,
        diffusion=diffusion,
        io_channels=io_channels,
        sample_rate=sample_rate,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        diffusion_downsampling_ratio=diffusion_downsampling_ratio,
        bottleneck=bottleneck,
        pretransform=pretransform
    )

def build_wavvae_old(hparams, init_pretrained=True, verbose=True):
    json_path = '/home/leike/spatial/ScriptSpeech/modules/vae/foavae.json'
    old_json_path = '/home/leike/spatial/stable-audio-tools/stable_audio_tools/checkpoints/vae_model_config.json'
    vae_path = '/home/leike/spatial/stable-audio-tools/stable_audio_tools/checkpoints/vae_model.ckpt'
    with open(json_path, "r", encoding="utf-8") as f:
        cfg: dict = json.load(f)
        
    model = create_autoencoder_from_config(cfg)
        
    if init_pretrained:
        vae_checkpoint = torch.load(vae_path, map_location='cpu')
        if "state_dict" in vae_checkpoint:
            state_dict = vae_checkpoint["state_dict"]
        else:
            state_dict = vae_checkpoint
        load_results = model.load_state_dict(state_dict, strict=True)
        
        
        if verbose:
            print(f"| loaded 'model' from '{vae_path}'.")
            missing_keys, unexpected_keys = load_results.missing_keys, load_results.unexpected_keys
            print(f"| Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    model = model.to(torch.device('cuda'))
    return model

def adapt_state_dict(model, old_state_dict):
    """
    [最终修复版] 修复了 Weight Norm 初始化为 0 导致 NaN 的问题
    """
    new_state_dict = model.state_dict()
    patched_state_dict = {}
    
    print("| Starting Model Surgery (NaN-Safe Version)...")
    
    # 动态获取层名称
    enc_layer_names = [n for n, _ in model.encoder.layers.named_children()]
    dec_layer_names = [n for n, _ in model.decoder.layers.named_children()]
    
    enc_in_name = f"encoder.layers.{enc_layer_names[0]}"
    enc_out_name = f"encoder.layers.{enc_layer_names[-1]}"
    dec_in_name = f"decoder.layers.{dec_layer_names[0]}"
    dec_out_name = f"decoder.layers.{dec_layer_names[-2]}"
    
    for key, new_param in new_state_dict.items():
        if key not in old_state_dict:
            patched_state_dict[key] = new_param
            continue
            
        old_param = old_state_dict[key]
        
        # ---------------------------------------------------------
        # 1. Encoder Input (2ch -> 4ch)
        # ---------------------------------------------------------
        # 输入层把 v 设为 0 是安全的，因为 norm 是跨 input channels 计算的，
        # 只要保留了前两个通道，总模长就不为 0。
        if key.startswith(enc_in_name) and "weight_v" in key:
            print(f"| [Surgery] Enc Input: {key}")
            patched_param = new_param.clone()
            patched_param[:, :2, :] = old_param
            patched_param[:, 2:, :] = 0 
            patched_state_dict[key] = patched_param
            continue
            
        # ---------------------------------------------------------
        # 2. Encoder Output (128 -> 256 channels) [关键修复点!]
        # ---------------------------------------------------------
        elif key.startswith(enc_out_name) and new_param.shape[0] == old_param.shape[0] * 2:
            print(f"| [Surgery] Enc Output: {key}")
            
            patched_param = new_param.clone()
            old_dim = old_param.shape[0] # 128
            half_old = old_dim // 2      # 64
            new_dim = new_param.shape[0] # 256
            half_new = new_dim // 2      # 128
            
            # 判断是 v, g 还是 bias
            if "weight_v" in key:
                # v 不能为 0，必须给随机值
                nn.init.normal_(patched_param, mean=0.0, std=0.01)
                # 覆盖旧的 Mean 方向
                patched_param[:half_old, ...] = old_param[:half_old, ...]
                # 覆盖旧的 Scale 方向
                patched_param[half_new:half_new+half_old, ...] = old_param[half_old:, ...]
                
            elif "weight_g" in key:
                # g 可以为 0，且必须为 0 (对于新增部分)
                patched_param.fill_(0) # 全部先置 0
                # 恢复旧的 Mean 幅度
                patched_param[:half_old, ...] = old_param[:half_old, ...]
                # 恢复旧的 Scale 幅度
                patched_param[half_new:half_new+half_old, ...] = old_param[half_old:, ...]
                
            elif "bias" in key:
                # bias 同理
                patched_param.fill_(0)
                patched_param[:half_old] = old_param[:half_old]
                patched_param[half_new:half_new+half_old] = old_param[half_old:]
            
            patched_state_dict[key] = patched_param
            continue

        # ---------------------------------------------------------
        # 3. Decoder Input (64 -> 128 channels)
        # ---------------------------------------------------------
        elif key.startswith(dec_in_name) and "weight_v" in key:
            print(f"| [Surgery] Dec Input: {key}")
            patched_param = new_param.clone()
            patched_param[:, :64, :] = old_param
            patched_param[:, 64:, :] = 0
            patched_state_dict[key] = patched_param
            continue

        # ---------------------------------------------------------
        # 4. Decoder Output (2 -> 4 channels)
        # ---------------------------------------------------------
        elif key.startswith(dec_out_name) and new_param.shape[0] == 4 and old_param.shape[0] == 2:
            print(f"| [Surgery] Dec Output: {key}")
            patched_param = new_param.clone()
            
            if "weight_v" in key:
                # v 用随机噪声，防止除以 0
                nn.init.normal_(patched_param, mean=0.0, std=0.01)
                patched_param[:2, ...] = old_param # 恢复前两项
                
            elif "weight_g" in key:
                # g 对于新增通道，建议设为 0 或者很小的值，保证初始输出接近 L/R
                # 这里我们设为 0，确保 Z/X 初始完全静音
                patched_param.fill_(0)
                patched_param[:2, ...] = old_param
            
            elif "bias" in key: # 虽然配置里可能是 False，防万一
                patched_param.fill_(0)
                patched_param[:2] = old_param

            patched_state_dict[key] = patched_param
            continue

        # ---------------------------------------------------------
        # Default
        # ---------------------------------------------------------
        if old_param.shape == new_param.shape:
            patched_state_dict[key] = old_param
        else:
            print(f"| [WARNING] Shape mismatch: {key}. Using Random Init.")
            patched_state_dict[key] = new_param

    return patched_state_dict

def freeze_layers(model):
    """
    冻结除首尾之外的所有层
    """
    # 动态获取层名称
    enc_names = [n for n, _ in model.encoder.layers.named_children()]
    dec_names = [n for n, _ in model.decoder.layers.named_children()]
    
    # 白名单：只训练这些层
    whitelist_prefixes = [
        f"encoder.layers.{enc_names[0]}",   # Enc Input
        f"encoder.layers.{enc_names[-1]}",  # Enc Output
        f"decoder.layers.{dec_names[0]}",   # Dec Input
        f"decoder.layers.{dec_names[-2]}",  # Dec Output (Conv)
        # f"decoder.layers.{dec_names[-1]}" # Tanh (无参数)
    ]
    
    print(f"| Whitelist layers: {whitelist_prefixes}")

    frozen_cnt = 0
    trainable_cnt = 0
    
    for name, param in model.named_parameters():
        is_trainable = False
        for prefix in whitelist_prefixes:
            if name.startswith(prefix):
                is_trainable = True
                break
        
        if is_trainable:
            param.requires_grad = True
            trainable_cnt += param.numel()
        else:
            param.requires_grad = False
            frozen_cnt += param.numel()
            
    print(f"| Freeze Report: Trainable={trainable_cnt/1e6:.2f}M, Frozen={frozen_cnt/1e6:.2f}M")

def build_wavvae(hparams, init_pretrained=True, verbose=True):
    # 路径配置
    json_path = '/home/leike/spatial/ScriptSpeech/modules/vae/foavae.json'
    vae_path = '/home/leike/spatial/stable-audio-tools/stable_audio_tools/checkpoints/vae_model.ckpt'
    
    # 1. 加载 Config
    with open(json_path, "r", encoding="utf-8") as f:
        cfg: dict = json.load(f)
        
    # 2. 初始化新模型 (4ch)
    model = create_autoencoder_from_config(cfg)
    
    # 3. 加载权重并手术
    if init_pretrained:
        if verbose: print(f"| Loading & Adapting weights from '{vae_path}'...")
        
        # 读取 checkpoint
        ckpt = torch.load(vae_path, map_location='cpu')
        old_state_dict = ckpt.get("state_dict", ckpt)
        
        # 执行手术
        new_state_dict = adapt_state_dict(model, old_state_dict)
        
        # 加载
        # 这里用 strict=True 是为了验证我们的手术字典是否完美构建了
        model.load_state_dict(new_state_dict, strict=True)
        
        # 4. 冻结
        freeze_layers(model)
        
    model = model.to(torch.device('cuda'))
    
    # 假设你已经有了 model 和 ckpt
    ckpt = torch.load(vae_path, map_location='cpu')
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    print("=== 侦探模式: 检查 Key 前缀 ===")
    print(f"你的模型里的 Key: {list(model.state_dict().keys())[0]}")
    print(f"Checkpoint里的 Key: {list(state_dict.keys())[0]}")
    
    return model
        
def build_wavvae_train(hparams, init_pretrained=True, verbose=True):
    json_path = '/home/leike/spatial/ScriptSpeech/modules/vae/foavae.json'
    vae_path = '/home/leike/spatial/ScriptSpeech/checkpoints/foavae_v2_nol1/model_ckpt_steps_52000.ckpt'
    
    # 1. 加载 Config
    with open(json_path, "r", encoding="utf-8") as f:
        cfg: dict = json.load(f)
        
    # 2. 初始化新模型 (4ch)
    model = create_autoencoder_from_config(cfg)
    
    # 3. 加载权重并手术
    if init_pretrained:
        if verbose: print(f"| Loading & Adapting weights from '{vae_path}'...")
        
        # 读取 checkpoint
        ckpt = torch.load(vae_path, map_location='cpu')
        state_dict = ckpt.get("model_gen", ckpt)
        
        # 加载
        model.load_state_dict(state_dict, strict=True)
        
        
    model = model.to(torch.device('cuda'))
    
    return model
    

if __name__ == "__main__":
    
    import json
    import torchaudio
    import tqdm
    import os
    import glob
    device = torch.device('cuda')
    
    # vae = build_wavvae(None)
    vae = build_wavvae_train(None)
    
    # audio = torch.randn(7, 2, int(44100 * 1.3))
    wavs = glob.glob('/home/leike/spatial/ScriptSpeech/results/vae/*')
    for wav_path in tqdm.tqdm(wavs):
        audio, _ = torchaudio.load(wav_path)
        audio = audio.to(device)
        # audio = audio.reshape(2, 2, -1)
        latents = vae.encode(audio)
        rec_audio = vae.decode(latents)
        rec_audio = rec_audio.squeeze().detach().cpu()
        item_name = wav_path.split('/')[-1][:-4]
        torchaudio.save(f'/home/leike/spatial/ScriptSpeech/results/vae/{item_name}_recon.wav', rec_audio, 44100)
    
    outputs = vae(audio)
    
    
    # import pdb; pdb.set_trace()
    
    # clips = pd.read_csv(args.data_csv)["id"].tolist()
    # wavs = [f"{args.data_path}/{clip}{args.audio_extension}" for clip in clips]
    # print_code = True
    
    # with torch.no_grad():
    #     for idx in tqdm.tqdm(range(0, len(wavs))):
    #         wav = wavs[idx]
    #         save_path = wav.replace(args.data_path, args.save_path).replace(
    #             args.audio_extension, ".npy"
    #         )
    #         if os.path.exists(save_path):
    #             continue
    #         if not os.path.exists(wav):
    #             continue
            
    #         audio, sr = torchaudio.load("/home/leike/spatial/ACE-Step/acestep/music_dcae/test2.wav")
    #         audio = audio.to(device)
    #         WX = audio[:2, :]
    #         YZ = audio[2:, :]
    #         audios = torch.stack([WX, YZ], dim=0)
            
    #         latents = vae.encode(audios)
    #         latents = latents.reshape(-1, latents.shape[-1])
    #         latents = latents.detach().to('cpu') 
    #         if print_code:
    #             print(latents.shape)
    #             print_code = False
    #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #         np.save(save_path, latents)

            
            
            
