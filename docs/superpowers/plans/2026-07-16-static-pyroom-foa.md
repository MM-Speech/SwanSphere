# Static Pyroom FOA Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a restartable remote dataset generator that attempts five deterministic pyroom FOA renders per FSDKaggle2019 noisy-training clip and publishes only valid 4-channel FLAC files and JSONL records.

**Architecture:** Keep the complete generator in `utils/data_sim/sim_static_pyroom.py`, split internally into pure position/schema helpers, acoustic/FLAC helpers, per-input failure isolation, and parent-owned dataset orchestration. Use standard-library `unittest` because `spat_edit` has no pytest installation, and use ordered multiprocessing so JSONL output remains deterministic.

**Tech Stack:** Python 3.11, NumPy, SciPy, SoundFile/libsndfile, pyroomacoustics 0.10.1, multiprocessing, unittest

---

## File Structure

- Modify `utils/data_sim/sim_static_pyroom.py`: constants, deterministic sampling, FOA rendering, FLAC validation, per-input processing, ordered multiprocessing, JSONL publication, and CLI.
- Create `tests/data_sim/test_sim_static_pyroom.py`: focused unit and integration tests executable with the Python standard library.

### Task 1: Deterministic Positions, Paths, and Metadata

**Files:**
- Modify: `utils/data_sim/sim_static_pyroom.py`
- Create: `tests/data_sim/test_sim_static_pyroom.py`

- [ ] **Step 1: Write failing tests for the room constants, deterministic position sampling, output path, label parsing, and exact metadata keys**

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from utils.data_sim.sim_static_pyroom import (
    DEFAULT_LISTENER_POSITION,
    DEFAULT_ROOM_DIM,
    build_metadata_record,
    output_path_for_item,
    parse_labels,
    sample_source_positions,
)


class StaticPyroomMetadataTest(unittest.TestCase):
    def test_positions_are_reproducible_inside_room_and_away_from_center(self):
        first = sample_source_positions(row_index=17, seed=1234)
        second = sample_source_positions(row_index=17, seed=1234)
        self.assertEqual(first.shape, (5, 3))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first > 0.0))
        self.assertTrue(np.all(first < DEFAULT_ROOM_DIM))
        distances = np.linalg.norm(first - DEFAULT_LISTENER_POSITION, axis=1)
        self.assertTrue(np.all(distances >= 0.5))

    def test_output_path_uses_source_stem_and_zero_based_position(self):
        root = Path("/dataset/static/fsdkaggle2019")
        self.assertEqual(
            output_path_for_item(root, "00097e21.wav", 3),
            root / "00097e21" / "3.flac",
        )

    def test_metadata_schema_is_exact(self):
        record = build_metadata_record(
            wav_path=Path("/out/00097e21/0.flac"),
            duration=1.25,
            position=np.array([1.0, 2.0, 3.0]),
            source_wav_path=Path("/in/00097e21.wav"),
            source_fname="00097e21.wav",
            position_index=0,
            labels=["Bathtub_(filling_or_washing)"],
        )
        self.assertEqual(
            set(record),
            {
                "wav_path",
                "duration",
                "position",
                "source_wav_path",
                "source_fname",
                "position_index",
                "labels",
            },
        )
        self.assertEqual(record["position"], {"x": 1.0, "y": 2.0, "z": 3.0})
        self.assertEqual(parse_labels("Dog,Marimba_and_xylophone"), ["Dog", "Marimba_and_xylophone"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/bin/conda run -n spat_edit \
  python tests/data_sim/test_sim_static_pyroom.py
```

Expected: import failure for the first missing helper from the empty production module.

- [ ] **Step 3: Implement the pure constants and helpers**

Add these concrete definitions to `sim_static_pyroom.py`:

```python
DEFAULT_ROOM_DIM = np.array([11.0, 11.0, 5.0], dtype=np.float64)
DEFAULT_LISTENER_POSITION = DEFAULT_ROOM_DIM / 2.0
DEFAULT_NUM_POSITIONS = 5
MIN_CENTER_DISTANCE = 0.5
WALL_INSET_METERS = 0.001


def sample_source_positions(row_index, seed, count=DEFAULT_NUM_POSITIONS):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(row_index)]))
    positions = []
    low = np.full(3, WALL_INSET_METERS, dtype=np.float64)
    high = DEFAULT_ROOM_DIM - WALL_INSET_METERS
    while len(positions) < count:
        candidate = rng.uniform(low, high)
        if np.linalg.norm(candidate - DEFAULT_LISTENER_POSITION) >= MIN_CENTER_DISTANCE:
            positions.append(candidate)
    return np.stack(positions)


def output_path_for_item(output_root, source_fname, position_index):
    return Path(output_root) / Path(source_fname).stem / f"{position_index}.flac"


def parse_labels(value):
    return [label.strip() for label in str(value or "").split(",") if label.strip()]


def build_metadata_record(
    wav_path,
    duration,
    position,
    source_wav_path,
    source_fname,
    position_index,
    labels,
):
    return {
        "wav_path": str(wav_path),
        "duration": float(duration),
        "position": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        },
        "source_wav_path": str(source_wav_path),
        "source_fname": str(source_fname),
        "position_index": int(position_index),
        "labels": list(labels),
    }
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run the Task 1 command and expect all Task 1 tests to pass with exit code 0.

- [ ] **Step 5: Commit Task 1**

```bash
git add utils/data_sim/sim_static_pyroom.py tests/data_sim/test_sim_static_pyroom.py
git commit -m "test: define static pyroom dataset schema"
```

### Task 2: FOA Rendering and Validated FLAC Output

**Files:**
- Modify: `utils/data_sim/sim_static_pyroom.py`
- Modify: `tests/data_sim/test_sim_static_pyroom.py`

- [ ] **Step 1: Add failing tests for `WYZX` gains, a real pyroom render, and FLAC round trip**

```python
from utils.data_sim.sim_static_pyroom import (
    DEFAULT_SAMPLE_RATE,
    foa_gains_from_room_position,
    render_foa,
    write_validated_flac,
)


class StaticPyroomAudioTest(unittest.TestCase):
    def test_right_source_maps_to_expected_wyzx_gains(self):
        source = DEFAULT_LISTENER_POSITION + np.array([1.0, 0.0, 0.0])
        gains = foa_gains_from_room_position(source)
        np.testing.assert_allclose(
            gains,
            np.array([1.0 / np.sqrt(2.0), -1.0, 0.0, 0.0]),
            atol=1.0e-7,
        )

    def test_real_pyroom_render_has_four_non_silent_channels(self):
        mono = np.zeros(256, dtype=np.float32)
        mono[0] = 1.0
        source = DEFAULT_LISTENER_POSITION + np.array([1.0, 0.0, 0.0])
        rendered = render_foa(mono, source)
        self.assertEqual(rendered.ndim, 2)
        self.assertEqual(rendered.shape[1], 4)
        self.assertGreater(float(np.max(np.abs(rendered))), 0.0)

    def test_flac_round_trip_is_four_channel_44100_hz(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.flac"
            audio = np.full((441, 4), 0.25, dtype=np.float32)
            duration = write_validated_flac(path, audio)
            info = sf.info(str(path))
            self.assertEqual(info.channels, 4)
            self.assertEqual(info.samplerate, DEFAULT_SAMPLE_RATE)
            self.assertEqual(info.frames, 441)
            self.assertAlmostEqual(duration, 0.01, places=7)
```

- [ ] **Step 2: Run the test and verify RED**

Run the full test file. Expected: import failure for `foa_gains_from_room_position` or the next missing Task 2 function.

- [ ] **Step 3: Implement audio loading, RIR shaping, FOA encoding, convolution, normalization, and validated atomic FLAC writing**

Use the requested constants and these exact interfaces:

```python
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_ABSORPTION = 0.80
DEFAULT_MAX_ORDER = 2
DEFAULT_SOUND_SPEED = 343.0
DEFAULT_IR_SECONDS = 0.10
DEFAULT_IR_TRIM_MS = 50.0
DEFAULT_PEAK_DBFS = -1.0
DEFAULT_FOA_ORDER = "WYZX"


def foa_gains_from_room_position(source_position):
    room_offset = np.asarray(source_position) - DEFAULT_LISTENER_POSITION
    relative = np.array([room_offset[0], room_offset[2], room_offset[1]])
    unit = relative / np.linalg.norm(relative)
    return np.array([1.0 / math.sqrt(2.0), -unit[0], unit[1], -unit[2]], dtype=np.float32)


def render_foa(mono_audio, source_position):
    pra = load_pyroomacoustics()
    pra.constants.set("c", DEFAULT_SOUND_SPEED)
    room = pra.ShoeBox(
        DEFAULT_ROOM_DIM.tolist(),
        fs=DEFAULT_SAMPLE_RATE,
        materials=pra.Material(energy_absorption=DEFAULT_ABSORPTION),
        max_order=DEFAULT_MAX_ORDER,
    )
    room.add_source(np.asarray(source_position, dtype=float).tolist())
    microphones = DEFAULT_LISTENER_POSITION.reshape(3, 1)
    room.add_microphone_array(pra.MicrophoneArray(microphones, fs=DEFAULT_SAMPLE_RATE))
    room.compute_rir()
    center_rir = shape_ir(np.asarray(room.rir[0][0], dtype=np.float32))
    rirs = foa_gains_from_room_position(source_position)[:, None] * center_rir[None, :]
    rendered = convolve_multichannel(np.asarray(mono_audio, dtype=np.float32), rirs)
    return normalize_peak(rendered, DEFAULT_PEAK_DBFS)


def write_validated_flac(path, audio):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.flac")
    try:
        sf.write(str(temporary), audio, DEFAULT_SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        duration = validate_existing_flac(temporary)
        os.replace(temporary, path)
        return duration
    finally:
        temporary.unlink(missing_ok=True)
```

Implement the helpers used above as normal production functions:

```python
_PRA = None


def load_pyroomacoustics():
    global _PRA
    if _PRA is None:
        import pyroomacoustics as pra
        _PRA = pra
    return _PRA


def load_and_resample_mono(path):
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[0] == 0 or audio.shape[1] != 1:
        raise ValueError(f"Expected non-empty mono input: {path}")
    mono = audio[:, 0]
    if float(np.max(np.abs(mono))) <= 0.0:
        raise ValueError(f"Silent input: {path}")
    if sample_rate != DEFAULT_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), DEFAULT_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            DEFAULT_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
    return mono.astype(np.float32, copy=False)


def find_ir_onset(ir):
    peak = float(np.max(np.abs(ir))) if ir.size else 0.0
    if peak <= 0.0:
        return 0
    hits = np.flatnonzero(np.abs(ir) >= max(peak * 1.0e-3, 1.0e-8))
    return int(hits[0]) if hits.size else 0


def shape_ir(ir):
    ir = np.asarray(ir, dtype=np.float32)
    onset = find_ir_onset(ir)
    trim = max(1, int(round(DEFAULT_SAMPLE_RATE * DEFAULT_IR_TRIM_MS / 1000.0)))
    limit = max(1, int(round(DEFAULT_SAMPLE_RATE * DEFAULT_IR_SECONDS)))
    end = min(ir.shape[0], onset + min(trim, limit))
    shaped = ir[:max(onset + 1, end)]
    if float(np.max(np.abs(shaped))) <= 0.0:
        raise ValueError("Silent pyroom RIR")
    return shaped


def fft_convolve_1d(signal, kernel):
    output_length = signal.shape[0] + kernel.shape[0] - 1
    fft_length = 1 << (output_length - 1).bit_length()
    output = np.fft.irfft(
        np.fft.rfft(signal, n=fft_length) * np.fft.rfft(kernel, n=fft_length),
        n=fft_length,
    )
    return output[:output_length].astype(np.float32)


def convolve_multichannel(mono_audio, rirs):
    return np.stack([fft_convolve_1d(mono_audio, rir) for rir in rirs], axis=1)


def normalize_peak(audio, peak_dbfs):
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        raise ValueError("Cannot normalize silent FOA audio")
    target = 10.0 ** (float(peak_dbfs) / 20.0)
    return np.clip(audio * (target / peak), -1.0, 1.0).astype(np.float32)


def validate_existing_flac(path):
    saved, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if sample_rate != DEFAULT_SAMPLE_RATE or saved.shape[1] != 4 or saved.shape[0] == 0:
        raise ValueError(f"Invalid 4-channel 44.1 kHz FLAC: {path}")
    if float(np.max(np.abs(saved))) <= 0.0:
        raise ValueError(f"Silent FLAC output: {path}")
    return saved.shape[0] / sample_rate
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run the full unittest file and expect all Task 1 and Task 2 tests to pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add utils/data_sim/sim_static_pyroom.py tests/data_sim/test_sim_static_pyroom.py
git commit -m "feat: render static pyroom FOA FLAC"
```

### Task 3: Per-Position Failure Isolation and Dataset Orchestration

**Files:**
- Modify: `utils/data_sim/sim_static_pyroom.py`
- Modify: `tests/data_sim/test_sim_static_pyroom.py`

- [ ] **Step 1: Add a failing test proving one failed position is deleted and siblings continue**

```python
class StaticPyroomFailureTest(unittest.TestCase):
    def test_failed_position_is_removed_without_dropping_siblings(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_path = root / "clip.wav"
            sf.write(str(source_path), np.ones(128, dtype=np.float32) * 0.1, DEFAULT_SAMPLE_RATE)
            stale = root / "output" / "clip" / "1.flac"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale")
            calls = 0

            def render_with_one_failure(mono, position):
                nonlocal calls
                position_index = calls
                calls += 1
                if position_index == 1:
                    raise RuntimeError("intentional render failure")
                return np.repeat(mono[:, None], 4, axis=1)

            item = InputItem(0, source_path, "clip.wav", ("Dog",))
            config = RenderConfig(root / "output", seed=9, overwrite=True)
            result = process_input_item(item, config, render_fn=render_with_one_failure)

            self.assertEqual([row["position_index"] for row in result.records], [0, 2, 3, 4])
            self.assertFalse(stale.exists())
            self.assertEqual(len(result.failures), 1)
```

- [ ] **Step 2: Run the test and verify RED**

Run the full test file. Expected: import failure for `InputItem`, `RenderConfig`, or `process_input_item`.

- [ ] **Step 3: Implement typed work items and per-position isolation**

```python
@dataclass(frozen=True)
class InputItem:
    row_index: int
    source_path: Path
    source_fname: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class RenderConfig:
    output_root: Path
    seed: int
    overwrite: bool


@dataclass
class ItemResult:
    records: list[dict]
    failures: list[str]


def process_input_item(item, config, render_fn=render_foa):
    positions = sample_source_positions(item.row_index, config.seed)
    records = []
    failures = []
    mono = None
    for position_index, position in enumerate(positions):
        output_path = output_path_for_item(config.output_root, item.source_fname, position_index)
        try:
            if output_path.exists() and not config.overwrite:
                duration = validate_existing_flac(output_path)
            else:
                if mono is None:
                    mono = load_and_resample_mono(item.source_path)
                duration = write_validated_flac(output_path, render_fn(mono, position))
            records.append(build_metadata_record(
                output_path,
                duration,
                position,
                item.source_path,
                item.source_fname,
                position_index,
                item.labels,
            ))
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            failures.append(f"{item.source_fname}[{position_index}]: {exc}")
    return ItemResult(records, failures)
```

The helpers from Task 2 provide identical validation for reused and newly written FLAC files and reject empty, silent, or non-mono source files.

- [ ] **Step 4: Implement CSV loading, ordered multiprocessing, atomic JSONL publication, CLI defaults, and summary output**

```python
def load_input_items(csv_path, audio_dir, limit=None):
    items = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as csv_file:
        for row_index, row in enumerate(csv.DictReader(csv_file)):
            source_fname = row["fname"].strip()
            items.append(InputItem(
                row_index=row_index,
                source_path=Path(audio_dir) / source_fname,
                source_fname=source_fname,
                labels=tuple(parse_labels(row.get("labels", ""))),
            ))
            if limit is not None and len(items) >= limit:
                break
    return items
```

Implement ordered processing, one-writer JSONL publication, and the exact CLI:

```python
def _process_task(task):
    return process_input_item(*task)


def render_dataset(items, config, metadata_jsonl, num_workers):
    metadata_jsonl = Path(metadata_jsonl)
    config.output_root.mkdir(parents=True, exist_ok=True)
    metadata_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_jsonl.with_name(f".{metadata_jsonl.name}.tmp")
    tasks = [(item, config) for item in items]
    pool = None
    if num_workers <= 1:
        results = map(_process_task, tasks)
    else:
        pool = mp.get_context("fork").Pool(processes=num_workers)
        results = pool.imap(_process_task, tasks, chunksize=1)
    record_count = 0
    failure_count = 0
    try:
        with temporary.open("w", encoding="utf-8") as jsonl_file:
            for result in results:
                for record in result.records:
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    record_count += 1
                for failure in result.failures:
                    print(f"[WARN] {failure}", file=sys.stderr, flush=True)
                    failure_count += 1
                jsonl_file.flush()
        if pool is not None:
            pool.close()
            pool.join()
        os.replace(temporary, metadata_jsonl)
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    return record_count, failure_count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Render static FSDKaggle2019 FOA FLAC data")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metadata-jsonl", type=Path, default=DEFAULT_METADATA_JSONL)
    parser.add_argument("--num-workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    items = load_input_items(args.metadata_csv, args.audio_dir, args.limit)
    config = RenderConfig(args.output_root, args.seed, args.overwrite)
    records, failures = render_dataset(items, config, args.metadata_jsonl, args.num_workers)
    print(f"Processed {len(items)} inputs: {records} outputs, {failures} failed positions")


if __name__ == "__main__":
    main()
```

Define the default paths exactly as follows:

```python
DEFAULT_AUDIO_DIR = Path(
    "/mnt/bn/sa-ag-data/leike/spatial_edit/dataset/fsdkaggle2019/"
    "FSDKaggle2019.audio_train_noisy"
)
DEFAULT_METADATA_CSV = Path(
    "/mnt/bn/sa-ag-data/leike/spatial_edit/dataset/fsdkaggle2019/"
    "FSDKaggle2019.meta/FSDKaggle2019.meta/train_noisy_post_competition.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/bn/sa-ag-data/leike/foa_dataset/random_pyroom/static/fsdkaggle2019"
)
DEFAULT_METADATA_JSONL = Path(
    "/mnt/bn/sa-ag-data/leike/foa_dataset/metadata/"
    "fsdkaggle2019_random_pyroom_static.jsonl"
)
```

- [ ] **Step 5: Run the full tests and verify GREEN**

Run:

```bash
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/bin/conda run -n spat_edit \
  python tests/data_sim/test_sim_static_pyroom.py
```

Expected: all tests pass, including the real pyroom render and failure-isolation test.

- [ ] **Step 6: Commit Task 3**

```bash
git add utils/data_sim/sim_static_pyroom.py tests/data_sim/test_sim_static_pyroom.py
git commit -m "feat: generate static FOA dataset metadata"
```

### Task 4: End-to-End Smoke Test and Requirement Audit

**Files:**
- Verify: `utils/data_sim/sim_static_pyroom.py`
- Verify: `tests/data_sim/test_sim_static_pyroom.py`

- [ ] **Step 1: Run all focused tests fresh**

```bash
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/bin/conda run -n spat_edit \
  python tests/data_sim/test_sim_static_pyroom.py
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/bin/conda run -n spat_edit \
  python tests/foa_vae/test_pyroom_dataset.py
```

Expected: both commands exit 0 with no failures.

- [ ] **Step 2: Render one real CSV input to an isolated smoke-test directory**

```bash
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/bin/conda run -n spat_edit \
  python utils/data_sim/sim_static_pyroom.py \
  --limit 1 \
  --num-workers 1 \
  --overwrite \
  --output-root /tmp/swansphere_static_pyroom_smoke/fsdkaggle2019 \
  --metadata-jsonl /tmp/swansphere_static_pyroom_smoke/metadata.jsonl
```

Expected: up to five numbered FLAC files and exactly one JSONL row per successful FLAC; partial failures are allowed but reported.

- [ ] **Step 3: Independently validate the smoke outputs**

Run this independent verifier in `spat_edit`:

```python
import json
from pathlib import Path

import numpy as np
import soundfile as sf

metadata = Path("/tmp/swansphere_static_pyroom_smoke/metadata.jsonl")
rows = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
expected = {
    "wav_path", "duration", "position", "source_wav_path",
    "source_fname", "position_index", "labels",
}
assert rows
for row in rows:
    assert set(row) == expected
    path = Path(row["wav_path"])
    assert path.exists()
    info = sf.info(str(path))
    assert info.channels == 4 and info.samplerate == 44100 and info.frames > 0
    assert abs(row["duration"] - info.frames / info.samplerate) < 1.0e-9
    position = np.array([row["position"][axis] for axis in "xyz"])
    assert np.all(position > 0.0) and np.all(position < np.array([11.0, 11.0, 5.0]))
    assert np.linalg.norm(position - np.array([5.5, 5.5, 2.5])) >= 0.5
```

- [ ] **Step 4: Compile and inspect CLI help**

```bash
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/bin/conda run -n spat_edit \
  python -m py_compile utils/data_sim/sim_static_pyroom.py tests/data_sim/test_sim_static_pyroom.py
/mnt/bn/sa-ag-data/zhangyu.34/anaconda/bin/conda run -n spat_edit \
  python utils/data_sim/sim_static_pyroom.py --help
```

Expected: compilation exits 0 and help lists every required path/control option.

- [ ] **Step 5: Audit the final diff and requirement coverage**

Confirm that no unrelated files are staged, output defaults exactly match the approved paths, room dimensions are `11 x 11 x 5`, the listener is centred, five positions are attempted, output indices are `0-4`, JSONL omits relative position/distance/sample-rate/renderer parameters, and failures delete their numbered file and omit its record.
