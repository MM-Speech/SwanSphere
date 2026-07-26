# Stable Audio 3 Absolute Imports Design

## Goal

Run Stable Audio 3 from the SwanSphere repository root with:

```bash
python data_gen/stable_audio_3/quickstart.py
```

All imports between files under `data_gen/stable_audio_3` must use fully
qualified absolute package paths. The implementation must not modify
`sys.path`, change the process working directory, or manage `PYTHONPATH`.

## Package Root

The canonical runtime package is:

```text
data_gen.stable_audio_3.stable_audio_3
```

Examples:

```python
from data_gen.stable_audio_3.stable_audio_3 import StableAudioModel
from data_gen.stable_audio_3.stable_audio_3.models.transformer import TransformerBlock
```

The caller is responsible for making the SwanSphere root importable. The
existing environment provides `PYTHONPATH=.` when commands are run from the
repository root.

## Import Policy

Within `data_gen/stable_audio_3/**/*.py`:

- reject every `ImportFrom` AST node with `level > 0`;
- reject imports whose module is exactly `stable_audio_3` or starts with
  `stable_audio_3.`;
- require project-local imports to start with
  `data_gen.stable_audio_3.stable_audio_3`;
- do not add code that mutates `sys.path` or calls `os.chdir`.

Third-party and standard-library imports remain unchanged.

## Scope

Rewrite imports in `quickstart.py`, package entry points, factories, loading
helpers, inference helpers, model definitions, and LoRA helpers. Model logic,
checkpoint resolution, generation behavior, CLI arguments, and output handling
remain unchanged.

## Validation

1. Add an AST-based import-policy test covering every Python file under the
   target directory.
2. Add a subprocess test that runs `quickstart.py --help` with SwanSphere as
   the working directory and `PYTHONPATH` pointing to that directory.
3. Update existing import tests to import the fully qualified package.
4. Run all unit tests from the SwanSphere root.
5. Run a short real inference from the SwanSphere root when gated-model access
   is available; otherwise report the external authentication failure separately.
