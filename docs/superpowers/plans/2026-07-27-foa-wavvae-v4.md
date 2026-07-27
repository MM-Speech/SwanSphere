# FOA WavVAE v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent four-channel FOA Oobleck VAE that retains Stable Audio's ds2048/z64 bottleneck, training recipe, and strict checkpoint behavior without importing or modifying v1, v2, or v3 implementation files.

**Architecture:** Add a standalone v4 model with strict source-state adaptation, a standalone training task that reproduces the approved v3 loss/EMA/GAN behavior, and a standalone inference entrypoint. All product changes are new files; existing versioned model, task, dataset, and inference files remain untouched.

**Tech Stack:** Python 3.11, PyTorch, diffusers `AutoencoderOobleck`, stable-audio-tools losses/discriminator, YAML, pytest, torchaudio.

---

## File Map

- `modules/foa_vae/wavvae_v4.py`: v4 architecture, wrapper, validation, and Stable Audio initialization.
- `tasks/foa_vae/wavvae_v4_task.py`: independent generator/discriminator training, EMA, losses, and optimizers.
- `egs/foa_vae/wavvae_v4.yaml`: ds2048/z64 training configuration and approved v3 loss values.
- `inference/foa_vae/infer_wavvae_v4.py`: direct v4 reconstruction entrypoint with no version detection.
- `egs/inference/inference_foa_vae_v4.yaml`: default v4 inference settings.
- `tests/foa_vae/test_wavvae_v4.py`: model and initialization contract tests.
- `tests/foa_vae/test_wavvae_v4_task.py`: task independence, schedule, EMA, and config tests.
- `tests/foa_vae/test_infer_wavvae_v4.py`: inference input, alignment, and direct-builder tests.

### Task 1: Implement Strict v4 Model Initialization

**Files:**
- Create: `tests/foa_vae/test_wavvae_v4.py`
- Create: `modules/foa_vae/wavvae_v4.py`

- [ ] **Step 1: Write failing initialization tests**

Create tests that import the not-yet-existing v4 module, construct state-only target/source fixtures with the three real mismatches, and assert the approved initialization contract:

```python
EXPECTED_MISMATCHES = {
    "encoder.conv1.weight_v",
    "decoder.conv2.weight_v",
    "decoder.conv2.weight_g",
}

def test_adaptation_copies_equal_shapes_and_keeps_random_encoder_direction():
    target, source = make_states()
    original_encoder_v = target["encoder.conv1.weight_v"].clone()
    patched = adapt_oobleck_v4_state_dict(StateModel(target), source, verbose=False)
    assert torch.equal(patched["encoder.block.weight"], source["encoder.block.weight"])
    assert torch.equal(patched["encoder.conv1.weight_v"], original_encoder_v)
    assert patched["encoder.conv1.weight_v"][:, 2:].abs().sum() > 0

def test_decoder_gain_is_symmetric_and_energy_preserving():
    target, source = make_states()
    patched = adapt_oobleck_v4_state_dict(StateModel(target), source, verbose=False)
    gain = patched["decoder.conv2.weight_g"]
    assert torch.all(gain > 0)
    assert torch.allclose(gain, gain[:1].expand_as(gain))
    assert torch.allclose(gain.square().sum(), source["decoder.conv2.weight_g"].square().sum())

def test_unexpected_shape_mismatch_fails():
    target, source = make_states()
    target["encoder.block.weight"] = torch.randn(5, 5)
    with pytest.raises(RuntimeError, match="Unexpected v4 shape mismatch"):
        adapt_oobleck_v4_state_dict(StateModel(target), source, verbose=False)

def test_v4_has_no_old_version_imports_or_projectors():
    source = Path("modules/foa_vae/wavvae_v4.py").read_text()
    assert "wavvae_v1" not in source
    assert "wavvae_v2" not in source
    assert "wavvae_v3" not in source
    assert "Projector" not in source
```

- [ ] **Step 2: Run tests and confirm import failure**

Run:

```bash
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/envs/spat_edit/bin/python -m pytest tests/foa_vae/test_wavvae_v4.py -q
```

Expected: collection fails with `ModuleNotFoundError: modules.foa_vae.wavvae_v4`.

- [ ] **Step 3: Implement the independent model module**

Implement these public names and behavior:

```python
DEFAULT_V4_DOWNSAMPLING_RATIOS = [2, 4, 4, 8, 8]
V4_SHAPE_MISMATCH_KEYS = {
    "encoder.conv1.weight_v",
    "decoder.conv2.weight_v",
    "decoder.conv2.weight_g",
}

class FOAWavVAEV4(nn.Module):
    def forward(self, audio):
        latent_dist = self.autoencoder.encode(audio).latent_dist
        latents = latent_dist.sample()
        reconstruction = self.autoencoder.decode(latents).sample
        return {
            "recon": reconstruction,
            "kl": latent_dist.kl().mean(),
            "mu": latent_dist.mean,
            "logvar": latent_dist.logvar,
        }

def adapt_oobleck_v4_state_dict(target_model, source_state, verbose=True):
    target_state = target_model.state_dict()
    patched = {}
    observed_mismatches = set()
    for key, target_value in target_state.items():
        if key not in source_state:
            raise RuntimeError(f"Missing Stable Audio parameter for v4: {key}")
        source_value = source_state[key]
        if source_value.shape == target_value.shape:
            patched[key] = source_value
            continue
        observed_mismatches.add(key)
        if key == "encoder.conv1.weight_v":
            patched[key] = target_value
        elif key == "decoder.conv2.weight_v":
            direction = torch.empty_like(target_value)
            nn.init.orthogonal_(direction.flatten(1))
            patched[key] = direction
        elif key == "decoder.conv2.weight_g":
            gain = source_value.float().square().sum().div(target_value.shape[0]).sqrt()
            patched[key] = torch.ones_like(target_value) * gain.to(target_value)
        else:
            raise RuntimeError(
                f"Unexpected v4 shape mismatch for {key}: "
                f"source={tuple(source_value.shape)} target={tuple(target_value.shape)}"
            )
    if observed_mismatches != V4_SHAPE_MISMATCH_KEYS:
        raise RuntimeError(
            f"v4 mismatch set must be {sorted(V4_SHAPE_MISMATCH_KEYS)}, "
            f"got {sorted(observed_mismatches)}"
        )
    return patched
```

`build_foa_wavvae_v4` validates exact 4ch/z64/[2,4,4,8,8], validates the source config as 2ch/z64/[2,4,4,8,8], loads `AutoencoderOobleck.from_pretrained`, calls the adapter, and loads the patched target with `strict=True`. Export `build_foa_wavvae = build_foa_wavvae_v4` without importing old versions.

- [ ] **Step 4: Run model tests**

Run the Task 1 pytest command again. Expected: all model tests pass.

- [ ] **Step 5: Commit model and tests**

```bash
git add modules/foa_vae/wavvae_v4.py tests/foa_vae/test_wavvae_v4.py
git commit -m "feat: add independent foa wavvae v4 model"
```

### Task 2: Add the v4 Training Configuration

**Files:**
- Create: `egs/foa_vae/wavvae_v4.yaml`
- Create: `tests/foa_vae/test_wavvae_v4_task.py`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_v4_training_config_contract():
    config = yaml.safe_load(Path("egs/foa_vae/wavvae_v4.yaml").read_text())
    assert config["task_cls"] == "tasks.foa_vae.wavvae_v4_task.FOAWavVAEV4Task"
    assert config["foa_vae"] == {
        "pretrained_model_dir": "checkpoints/vae",
        "input_channels": 4,
        "output_channels": 4,
        "latent_channels": 64,
        "downsampling_ratios": [2, 4, 4, 8, 8],
    }
    assert config["losses"] == {
        "lambda_mrstft": 1.0,
        "lambda_high_frequency_excess_db": 0.01,
        "lambda_kl": 1.0e-5,
        "lambda_mel_multiband": 0.0,
        "lambda_adv": 0.05,
        "lambda_feature_matching": 5.0,
        "lambda_dis": 1.0,
    }
    assert compute_target_num_samples(44100, 1.5, 8, 256, True) == 65536
```

- [ ] **Step 2: Run the config test and confirm missing-file failure**

```bash
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/envs/spat_edit/bin/python -m pytest tests/foa_vae/test_wavvae_v4_task.py::test_v4_training_config_contract -q
```

Expected: failure because `egs/foa_vae/wavvae_v4.yaml` does not exist.

- [ ] **Step 3: Create the v4 YAML**

Start from the approved values in `wavvae_v3.yaml`, then set:

```yaml
task_cls: tasks.foa_vae.wavvae_v4_task.FOAWavVAEV4Task
hop_size: 256
frames_multiple: 8
sample_size: 65536
foa_target_seconds: 1.5

foa_vae:
  pretrained_model_dir: checkpoints/vae
  input_channels: 4
  output_channels: 4
  latent_channels: 64
  downsampling_ratios: [2, 4, 4, 8, 8]

losses:
  lambda_mrstft: 1.0
  lambda_high_frequency_excess_db: 0.01
  lambda_kl: 1.0e-5
  lambda_mel_multiband: 0.0
  lambda_adv: 0.05
  lambda_feature_matching: 5.0
  lambda_dis: 1.0
```

Copy all other optimizer, EMA, discriminator, spectral, and high-frequency settings exactly from the current v3 YAML. Do not add spatial losses.

- [ ] **Step 4: Run the config test**

Expected: the config contract test passes.

- [ ] **Step 5: Commit the configuration**

```bash
git add egs/foa_vae/wavvae_v4.yaml tests/foa_vae/test_wavvae_v4_task.py
git commit -m "config: add foa wavvae v4 training recipe"
```

### Task 3: Implement the Independent v4 Training Task

**Files:**
- Create: `tasks/foa_vae/wavvae_v4_task.py`
- Modify: `tests/foa_vae/test_wavvae_v4_task.py`

- [ ] **Step 1: Add failing task independence and schedule tests**

```python
def test_v4_task_is_independent():
    source = Path("tasks/foa_vae/wavvae_v4_task.py").read_text()
    assert "wavvae_v1" not in source
    assert "wavvae_v2" not in source
    assert "wavvae_v3" not in source
    assert "from modules.foa_vae.wavvae_v4 import" in source

def test_gan_ramp_contract():
    task = object.__new__(FOAWavVAEV4Task)
    task.global_step = 9999
    assert task._gan_ramp(10000) == 0.0
    task.global_step = 20000
    assert task._gan_ramp(10000) == pytest.approx(0.5)
    task.global_step = 30000
    assert task._gan_ramp(10000) == 1.0

def test_tensor_state_ema_round_trip():
    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    source = TensorStateEMAModel([parameter], decay=0.9)
    state = source.state_dict()
    target = TensorStateEMAModel([nn.Parameter(torch.zeros(2))], decay=0.1)
    target.load_state_dict(state, strict=True)
    assert target.decay == pytest.approx(0.9)
    assert torch.equal(target.shadow_params[0], source.shadow_params[0])
```

- [ ] **Step 2: Run task tests and confirm import failure**

```bash
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/envs/spat_edit/bin/python -m pytest tests/foa_vae/test_wavvae_v4_task.py -q
```

Expected: collection fails because `tasks.foa_vae.wavvae_v4_task` does not exist.

- [ ] **Step 3: Implement the standalone task**

Create concrete independent definitions named `TensorStateEMAModel`,
`HighFrequencyExcessDBLoss`, `trim_to_shortest`, and `FOAWavVAEV4Task`.
Their signatures and checkpoint-facing state names must match the approved design
so training and inference checkpoints remain deterministic.

Reproduce the approved behavior directly in this file and import only v4:

```python
from modules.foa_vae.wavvae_v4 import build_foa_wavvae_v4, trainable_param_report

def build_model(self):
    init_pretrained = (
        not hparams.get("from_scratch", False)
        and not hparams.get("load_ckpt", "")
        and not hparams.get("resume_from", "")
    )
    self.model_gen = build_foa_wavvae_v4(hparams=hparams, init_pretrained=init_pretrained)
    self.model_gen.requires_grad_(True)
```

The generator step computes MRSTFT, high-frequency excess, optional mel, KL, adversarial, and feature-matching losses. The discriminator step uses detached reconstruction. Preserve the 10k pretrain, 20k ramp, two AdamW optimizers, warmup schedules, gradient clipping, checkpoint restore, and EMA update. Do not import any versioned task or model other than v4.

- [ ] **Step 4: Run task tests**

Expected: config, independence, schedule, and EMA tests pass.

- [ ] **Step 5: Commit the task**

```bash
git add tasks/foa_vae/wavvae_v4_task.py tests/foa_vae/test_wavvae_v4_task.py
git commit -m "feat: add independent foa wavvae v4 task"
```

### Task 4: Implement Standalone v4 Inference

**Files:**
- Create: `inference/foa_vae/infer_wavvae_v4.py`
- Create: `egs/inference/inference_foa_vae_v4.yaml`
- Create: `tests/foa_vae/test_infer_wavvae_v4.py`

- [ ] **Step 1: Write failing standalone inference tests**

```python
def test_v4_inference_is_independent_and_direct():
    source = Path("inference/foa_vae/infer_wavvae_v4.py").read_text()
    assert "infer_wavvae import" not in source
    assert "build_foa_wavvae_v4" in source
    assert "resolve_model_module" not in source

def test_padding_and_trimming_contract():
    wav = torch.randn(4, 5000)
    padded, original = pad_to_multiple(wav, 2048)
    assert original == 5000
    assert padded.shape[-1] == 6144

def test_v4_default_config_contract():
    config = yaml.safe_load(Path("egs/inference/inference_foa_vae_v4.yaml").read_text())
    assert config["model_config"] == "egs/foa_vae/wavvae_v4.yaml"
    assert config["expected_channels"] == 4
    assert config["pad_to_multiple"] == 2048
    assert config["load_ckpt_strict"] is True
```

- [ ] **Step 2: Run tests and confirm missing-module failure**

```bash
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/envs/spat_edit/bin/python -m pytest tests/foa_vae/test_infer_wavvae_v4.py -q
```

Expected: collection fails because `inference.foa_vae.infer_wavvae_v4` does not exist.

- [ ] **Step 3: Implement direct v4 inference**

The script uses the v4 builder directly:

```python
DEFAULT_CONFIG = "egs/inference/inference_foa_vae_v4.yaml"

def build_model(config, model_hparams, device):
    model = build_foa_wavvae_v4(hparams=model_hparams, init_pretrained=False)
    resolved_ckpt = resolve_checkpoint_path(config["ckpt_path"], config.get("ckpt_steps"))
    load_ckpt(
        model,
        resolved_ckpt,
        config.get("model_name", "model_gen"),
        strict=True,
        force=True,
        map_location="cpu",
    )
    return model.eval().to(device), resolved_ckpt
```

Independently implement config loading, input collection, checkpoint resolution, torchaudio plus ffmpeg fallback loading, four-channel validation, resampling, 2048 padding, reconstruction trimming, unique output paths, and float WAV saving. Do not import the old inference module and do not infer model versions.

Create a default YAML with explicit `model_config: egs/foa_vae/wavvae_v4.yaml`, strict loading, four expected channels, `pad_to_multiple: 2048`, bf16 CUDA inference, and FLOAT WAV output. Leave `ckpt_path` empty so the checked-in config does not point to a nonexistent experiment.

- [ ] **Step 4: Run inference tests**

Expected: all standalone inference tests pass.

- [ ] **Step 5: Commit inference**

```bash
git add inference/foa_vae/infer_wavvae_v4.py egs/inference/inference_foa_vae_v4.yaml tests/foa_vae/test_infer_wavvae_v4.py
git commit -m "feat: add standalone foa wavvae v4 inference"
```

### Task 5: Verify the Complete v4 Without Training

**Files:**
- Verify only; do not create or modify product files unless a test exposes a defect.

- [ ] **Step 1: Compile all new Python modules**

```bash
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/envs/spat_edit/bin/python -m py_compile \
  modules/foa_vae/wavvae_v4.py \
  tasks/foa_vae/wavvae_v4_task.py \
  inference/foa_vae/infer_wavvae_v4.py
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 2: Run the complete targeted test suite**

```bash
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/envs/spat_edit/bin/python -m pytest \
  tests/foa_vae/test_wavvae_v4.py \
  tests/foa_vae/test_wavvae_v4_task.py \
  tests/foa_vae/test_infer_wavvae_v4.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run existing adjacent tests**

```bash
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/envs/spat_edit/bin/python -m pytest \
  tests/foa_vae/test_inference_utils.py \
  tests/foa_vae/test_high_frequency_excess_db_loss.py \
  tests/foa_vae/test_pyroom_dataset.py -q
```

Expected: all existing adjacent tests pass unchanged.

- [ ] **Step 4: Build v4 from the real Stable Audio checkpoint**

Run a bounded Python command that resolves `egs/foa_vae/wavvae_v4.yaml`, calls `build_foa_wavvae_v4(init_pretrained=True)`, prints the target config and parameter count, and exits. Assert four audio channels, 64 decoder input channels, and `[2,4,4,8,8]` ratios.

Expected: strict initialization succeeds and logs exactly the three special tensors.

- [ ] **Step 5: Run one bounded no-gradient forward smoke test when a CUDA device is available**

Use one synthetic `[1,4,65536]` tensor, `torch.no_grad()`, eval mode, and no optimizer or dataloader. Check reconstruction has four channels and latent length 32, then exit.

Expected: one forward pass succeeds. If the entry host has no suitable CUDA device, report the skipped smoke test rather than launching a training or cluster job.

- [ ] **Step 6: Confirm no formal training process was started**

Do not execute `python tasks/run.py`, `torchrun`, `nohup`, or any distributed launch command.

- [ ] **Step 7: Inspect final scope**

```bash
git status --short
git diff --stat HEAD~4..HEAD
```

Expected: only the approved v4 files and documentation commits are present in implementation commits; pre-existing user modifications remain unstaged and unchanged.
