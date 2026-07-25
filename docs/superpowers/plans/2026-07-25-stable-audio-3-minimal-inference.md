# Stable Audio 3 Minimal Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor the minimum Stable Audio 3 text-to-audio runtime into SwanSphere and provide a directly runnable WAV-generating quickstart.

**Architecture:** Keep upstream runtime modules unchanged in a nested `stable_audio_3` package so their absolute and relative imports remain valid. Add one small CLI entry point beside that package; model weights continue to resolve through the existing Hugging Face cache.

**Tech Stack:** Python 3.10, PyTorch 2.7.1 CUDA 12.6, Torchaudio 2.7.1 CUDA 12.6, Transformers, Hugging Face Hub, Safetensors, pytest.

---

### Task 1: Vendor the minimal runtime package

**Files:**
- Create: `data_gen/stable_audio_3/tests/test_runtime_import.py`
- Create: `data_gen/stable_audio_3/stable_audio_3/__init__.py`
- Create: `data_gen/stable_audio_3/stable_audio_3/{factory,loading_utils,model,model_configs,verbose}.py`
- Create: `data_gen/stable_audio_3/stable_audio_3/data/{__init__,utils}.py`
- Create: `data_gen/stable_audio_3/stable_audio_3/inference/*.py`
- Create: `data_gen/stable_audio_3/stable_audio_3/models/{__init__,autoencoders,blocks,bottleneck,conditioners,diffusion,dit,pretransforms,transformer,utils}.py`
- Create: `data_gen/stable_audio_3/stable_audio_3/models/lora/{__init__,loader,model,utils}.py`

- [ ] **Step 1: Write the failing isolated-import test**

```python
from pathlib import Path
import importlib
import sys


def test_runtime_imports_from_vendored_package():
    runtime_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(runtime_root))
    sys.modules.pop("stable_audio_3", None)
    module = importlib.import_module("stable_audio_3")
    assert Path(module.__file__).resolve().is_relative_to(runtime_root)
    assert module.StableAudioModel is not None
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
conda run -n sdaudio3 python -m pytest data_gen/stable_audio_3/tests/test_runtime_import.py -v
```

Expected: FAIL because the target package does not exist or resolves outside `data_gen/stable_audio_3`.

- [ ] **Step 3: Copy only the transitive runtime modules**

From `/mnt/bn/sa-ag-data/leike/code/stable-audio-3`, create the target directories and copy these unchanged files:

```bash
target=/mnt/bn/sa-ag-data/leike/code/SwanSphere/data_gen/stable_audio_3/stable_audio_3
mkdir -p "$target/data" "$target/inference" "$target/models/lora"
cp stable_audio_3/__init__.py stable_audio_3/factory.py stable_audio_3/loading_utils.py stable_audio_3/model.py stable_audio_3/model_configs.py stable_audio_3/verbose.py "$target/"
cp stable_audio_3/data/__init__.py stable_audio_3/data/utils.py "$target/data/"
cp stable_audio_3/inference/__init__.py stable_audio_3/inference/audio_utils.py stable_audio_3/inference/distribution_shift.py stable_audio_3/inference/sampling.py "$target/inference/"
cp stable_audio_3/models/__init__.py stable_audio_3/models/autoencoders.py stable_audio_3/models/blocks.py stable_audio_3/models/bottleneck.py stable_audio_3/models/conditioners.py stable_audio_3/models/diffusion.py stable_audio_3/models/dit.py stable_audio_3/models/pretransforms.py stable_audio_3/models/transformer.py stable_audio_3/models/utils.py "$target/models/"
cp stable_audio_3/models/lora/__init__.py stable_audio_3/models/lora/loader.py stable_audio_3/models/lora/model.py stable_audio_3/models/lora/utils.py "$target/models/lora/"
```

Do not copy `__pycache__`, training, dataset, interface, CLI, or optimized files.

- [ ] **Step 4: Run the isolated-import test and verify GREEN**

Run the Task 1 pytest command. Expected: one passing test and the imported file under SwanSphere.

- [ ] **Step 5: Commit the vendored runtime and test**

```bash
git add data_gen/stable_audio_3/stable_audio_3 data_gen/stable_audio_3/tests/test_runtime_import.py
git commit -m "feat: vendor stable audio inference runtime"
```

### Task 2: Add the directly runnable quickstart

**Files:**
- Create: `data_gen/stable_audio_3/tests/test_quickstart.py`
- Create: `data_gen/stable_audio_3/quickstart.py`

- [ ] **Step 1: Write failing tests for CLI defaults and validation**

```python
import quickstart


def test_default_arguments_are_runnable():
    args = quickstart.parse_args([])
    assert args.model == "medium"
    assert args.duration == 10.0
    assert args.steps == 8
    assert args.output == "output.wav"


def test_non_positive_duration_is_rejected():
    try:
        quickstart.parse_args(["--duration", "0"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("zero duration must be rejected")
```

- [ ] **Step 2: Run the quickstart tests and verify RED**

Run:

```bash
cd data_gen/stable_audio_3
conda run -n sdaudio3 python -m pytest tests/test_quickstart.py -v
```

Expected: FAIL because `quickstart.py` does not exist.

- [ ] **Step 3: Implement the minimal CLI**

Implement `parse_args(argv=None)` with model, prompt, duration, steps, seed, and output arguments. Implement `main(argv=None)` to require CUDA, call `StableAudioModel.from_pretrained`, generate one batch using `model.model_config["sample_size"]`, save `audio[0]` at `model.model_config["sample_rate"]`, and print the output path and shape. Guard execution with `if __name__ == "__main__": main()`.

- [ ] **Step 4: Run unit tests and CLI help**

```bash
conda run -n sdaudio3 python -m pytest tests -v
conda run -n sdaudio3 python quickstart.py --help
```

Expected: all tests pass and help exits zero without loading weights.

- [ ] **Step 5: Commit the quickstart and tests**

```bash
git add data_gen/stable_audio_3/quickstart.py data_gen/stable_audio_3/tests/test_quickstart.py
git commit -m "feat: add stable audio quickstart"
```

### Task 3: Verify CUDA dependencies and end-to-end inference

**Files:**
- Verify: `data_gen/stable_audio_3/quickstart.py`
- Generate temporarily: `/tmp/stable_audio_3_quickstart.wav`

- [ ] **Step 1: Verify the PyTorch family and Torchvision operator**

```bash
conda run -n sdaudio3 python -c 'import torch, torchvision, torchaudio; from torchvision.ops import nms; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.cuda.is_available())'
```

Expected: all three versions end in `+cu126`, CUDA is true, and imports succeed.

- [ ] **Step 2: Run a short end-to-end generation**

From `data_gen/stable_audio_3`, run `quickstart.py` with `--duration 1 --steps 2 --output /tmp/stable_audio_3_quickstart.wav`. Export the supplied HTTP(S) proxy only if Hugging Face reports a cache miss.

- [ ] **Step 3: Verify the WAV artifact**

Use Torchaudio to assert the WAV exists, has two channels, the expected sample rate, finite samples, and a nonzero frame count.

- [ ] **Step 4: Run final tests and source-path scan**

```bash
conda run -n sdaudio3 python -m pytest data_gen/stable_audio_3/tests -v
grep -R "/mnt/bn/sa-ag-data/leike/code/stable-audio-3" data_gen/stable_audio_3 --include='*.py'
```

Expected: tests pass and grep finds no source-repository dependency.
