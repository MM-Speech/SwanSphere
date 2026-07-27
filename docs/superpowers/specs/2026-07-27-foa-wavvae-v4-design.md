# FOA WavVAE v4 Design

## Goal

Build an independent FOA Stable Audio VAE variant with four-channel WYZX audio, the original Stable Audio temporal downsampling factor of 2048, and 64 latent channels. The model, training task, configuration, and inference entrypoint must remain independent from v1, v2, and v3 so future changes do not couple the versions.

The implementation may run unit tests and a bounded construction or forward smoke test. It must not launch a formal training run.

## Non-goals

- Do not change v1, v2, or v3 model, task, config, inference, or checkpoint files.
- Do not add projectors, adapters, stereo-to-FOA mappings, repeated stereo channels, zero-extra channels, staged activation, or directional losses.
- Do not change the shared mixed dataset metadata or dataset weights.
- Do not resume from any v1, v2, or v3 training checkpoint.

## Files

Create the following product files:

- `modules/foa_vae/wavvae_v4.py`
- `tasks/foa_vae/wavvae_v4_task.py`
- `egs/foa_vae/wavvae_v4.yaml`
- `inference/foa_vae/infer_wavvae_v4.py`
- `egs/inference/inference_foa_vae_v4.yaml`
- `tests/foa_vae/test_wavvae_v4.py`
- `tests/foa_vae/test_wavvae_v4_task.py`
- `tests/foa_vae/test_infer_wavvae_v4.py`

No existing v1, v2, v3, dataset, or inference file is modified.

## Model Architecture

`modules/foa_vae/wavvae_v4.py` implements a self-contained thin wrapper around `diffusers.AutoencoderOobleck`. It does not import model or initialization helpers from `wavvae_v1`, `wavvae_v2`, or `wavvae_v3`.

The target architecture is fixed to:

```text
audio_channels = 4
decoder_input_channels = 64
downsampling_ratios = [2, 4, 4, 8, 8]
total downsampling = 2048
sample rate = 44100 Hz
latent frame rate = 21.533203125 Hz
```

The builder rejects configurations where input and output channels are not both four, latent channels are not 64, or the exact downsampling ratio list differs from `[2, 4, 4, 8, 8]`. The source Stable Audio config must describe a two-channel, z64, ds2048 Oobleck.

The wrapper samples the Oobleck posterior, decodes it, and returns:

```python
{
    "recon": reconstruction,
    "kl": kl_loss,
    "mu": latent_distribution.mean,
    "logvar": latent_distribution.logvar,
}
```

There are no learnable layers outside the native Oobleck model.

## Stable Audio Initialization

Instantiate the four-channel target model first, then load the original two-channel Stable Audio model. Copy every source state tensor whose shape exactly matches the target.

Exactly three target state tensors are allowed to differ in shape:

```text
encoder.conv1.weight_v: source [128, 2, 7], target [128, 4, 7]
decoder.conv2.weight_v: source [2, 128, 7], target [4, 128, 7]
decoder.conv2.weight_g: source [2, 1, 1], target [4, 1, 1]
```

Any missing key or additional shape mismatch raises an error before loading.

Initialization rules:

- Keep the target constructor's independent Kaiming-initialized four-channel direction for `encoder.conv1.weight_v`.
- Copy the source `encoder.conv1.weight_g` and bias because their shapes are unchanged.
- Initialize the four flattened rows of `decoder.conv2.weight_v` as independent random directions. They must not be repeated or zero.
- Set all four decoder output gains to the same positive value while preserving the source output layer's total squared gain:

```python
g_v4 = torch.sqrt(source_g.square().sum() / 4.0)
target_g = torch.full_like(target_g, g_v4)
```

Load the fully patched target state with `strict=True`. Verbose initialization reports the number of exact copies and the three specially initialized tensors. With `init_pretrained=False`, construct the same target architecture without loading Stable Audio; inference then loads a trained v4 checkpoint strictly.

## Independent Training Task

`tasks/foa_vae/wavvae_v4_task.py` independently implements the current v3 training recipe. It does not import or subclass `FOAWavVAEV3Task` and does not import any v1, v2, or v3 model module.

The file contains its own implementations of the tensor-only EMA checkpoint adapter, high-frequency excess dB loss, reconstruction length trimming, optimizers, warmup schedulers, discriminator control, generator loss, discriminator loss, gradient clipping, EMA update, and checkpoint restore behavior.

All v4 generator parameters are trainable from the first optimizer step. The task uses a four-channel Encodec discriminator. The discriminator is pretrained for 10,000 generator steps, after which adversarial and feature-matching weights ramp for 20,000 steps.

The generator losses are limited to:

```yaml
lambda_mrstft: 1.0
lambda_high_frequency_excess_db: 0.01
lambda_kl: 1.0e-5
lambda_mel_multiband: 0.0
lambda_adv: 0.05
lambda_feature_matching: 5.0
lambda_dis: 1.0
```

No direction, intensity, covariance, cross-spectrum, stereo-projection, or other spatial loss is present.

## Training Configuration and Length

`egs/foa_vae/wavvae_v4.yaml` uses the current `mixed_foa_dataset.yaml` and otherwise reproduces the v3 optimizer, discriminator, EMA, loss, and training settings. It changes only the task class, model architecture, and crop alignment needed by v4.

The crop configuration is:

```yaml
hop_size: 256
frames_multiple: 8
sample_size: 65536
foa_target_seconds: 1.5
```

The existing dataset alignment computes a multiple of `8 * 256 = 2048`. A 1.5-second request at 44.1 kHz contains 66,150 samples and is rounded down to 65,536 samples, producing exactly 32 v4 latent frames. This requires no dataset code or new configuration key.

The initial model source is `checkpoints/vae`. `load_ckpt` remains empty for a new run, `from_scratch` is false, and EMA remains enabled.

## Independent Inference

`inference/foa_vae/infer_wavvae_v4.py` is a standalone v4 entrypoint and does not import `infer_wavvae.py`. It directly imports `build_foa_wavvae_v4`, constructs the model with `init_pretrained=False`, and strictly loads `model_gen` from a v4 checkpoint.

The script supports a single input, a list file, or JSONL metadata; validates four input channels; resamples to the configured sample rate; falls back to ffmpeg when torchaudio cannot decode a file; pads audio to a multiple of 2048; removes inference padding after reconstruction; and writes four-channel floating-point WAV output. Loading v1, v2, or v3 checkpoints fails through strict state loading.

`egs/inference/inference_foa_vae_v4.yaml` is the default inference configuration for the new script. Existing inference code and configuration remain unchanged.

## Error Handling

- Missing Stable Audio config or weights fail before model training begins.
- Incorrect source or target architecture values raise descriptive `ValueError` exceptions.
- Unexpected state keys or shapes raise `RuntimeError` with source and target shapes.
- Inference requires a checkpoint path, four-channel input, and at least one input path.
- Inference checkpoint loading is always strict.

## Tests and Verification

`tests/foa_vae/test_wavvae_v4.py` verifies exact architecture constraints, the three-key mismatch allowlist, exact copying of same-shape weights, nonzero and nonrepeated four-channel directions, preservation of decoder total squared gain, strict loading, and absence of projector or old-version imports.

`tests/foa_vae/test_wavvae_v4_task.py` verifies that the independent task builds v4, contains the approved loss weights only, implements the 10k discriminator pretrain and 20k GAN ramp, round-trips EMA state, creates separate generator and discriminator optimizers, and resolves the training crop to 65,536 samples.

`tests/foa_vae/test_infer_wavvae_v4.py` verifies direct v4 construction, strict checkpoint loading, four-channel validation, 2048 alignment, reconstruction trimming, input collection, and output path behavior.

Final verification consists of targeted pytest runs, Python compilation of all new files, a v4 build from `checkpoints/vae`, an initialization audit, and at most a bounded no-gradient forward smoke test. It does not start `tasks/run.py` or any formal training process.
