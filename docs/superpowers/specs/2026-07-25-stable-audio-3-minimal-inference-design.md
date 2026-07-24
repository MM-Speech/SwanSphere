# Stable Audio 3 Minimal Inference Migration Design

## Goal

Create a self-contained Stable Audio 3 text-to-audio inference entry point under
`data_gen/stable_audio_3` in SwanSphere. Runtime code must not import from or
depend on `/mnt/bn/sa-ag-data/leike/code/stable-audio-3`.

## Scope

Vendor the smallest unchanged runtime subset of the upstream `stable_audio_3`
Python package required by `StableAudioModel.from_pretrained("medium")` and
`StableAudioModel.generate(...)`:

- package entry point and model wrapper;
- model configuration and checkpoint resolution;
- model factories and checkpoint loading;
- inference sampling and audio preparation;
- neural network model definitions, including modules imported eagerly by the
  model wrapper.

Do not migrate training, datasets, Gradio, CLI, optimized mobile backends,
documentation, caches, or compiled Python artifacts.

## Layout

```text
data_gen/stable_audio_3/
|-- quickstart.py
`-- stable_audio_3/
    |-- __init__.py
    |-- factory.py
    |-- loading_utils.py
    |-- model.py
    |-- model_configs.py
    |-- inference/
    `-- models/
```

The nested package keeps upstream absolute imports such as
`from stable_audio_3.models...` working without rewriting model internals.

## Quickstart Behavior

`quickstart.py` will:

- import only the vendored package beside the script;
- accept model name, prompt, duration, seed, steps, and output path as command
  line arguments with runnable defaults;
- require CUDA for the medium model and fail early with a clear error when CUDA
  is unavailable;
- load model weights through the upstream Hugging Face resolver, reusing the
  existing cache for the `tiger` user;
- generate one waveform and save it with the model sample rate using
  `torchaudio.save`;
- create the output parent directory when needed and print the resolved output
  path and tensor shape.

Model weights are not copied into SwanSphere. Network access is only needed on
a cache miss; the configured HTTP(S) proxy may be exported by the caller when
that happens.

## Dependencies

Run with the existing `sdaudio3` conda environment. Its PyTorch family must be
the matching CUDA 12.6 set:

- `torch==2.7.1+cu126`
- `torchvision==0.22.1+cu126`
- `torchaudio==2.7.1+cu126`

The vendored runtime also uses the dependencies already declared upstream,
including `transformers`, `huggingface-hub`, `safetensors`, `einops`, `numpy`,
`packaging`, and `tqdm`.

## Validation

Validation will cover:

1. a static import test from SwanSphere with the upstream repository removed
   from `sys.path`;
2. CLI help and argument parsing without loading model weights;
3. import of Torch, Torchvision, Torchaudio, and the `torchvision::nms` operator;
4. an end-to-end short CUDA generation that creates a readable WAV file;
5. a source-path scan proving the migrated code does not reference the original
   repository location.

No upstream source files or model weights will be modified.
