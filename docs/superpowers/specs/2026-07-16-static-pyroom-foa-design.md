# Static Pyroom FOA Dataset Design

## Goal

Create `utils/data_sim/sim_static_pyroom.py` to render up to five static first-order Ambisonics (FOA) variants for every mono clip listed by the FSDKaggle2019 noisy-training metadata.

## Inputs and Outputs

The default input audio directory is:

```text
/mnt/bn/sa-ag-data/leike/spatial_edit/dataset/fsdkaggle2019/FSDKaggle2019.audio_train_noisy
```

The default input CSV is:

```text
/mnt/bn/sa-ag-data/leike/spatial_edit/dataset/fsdkaggle2019/FSDKaggle2019.meta/FSDKaggle2019.meta/train_noisy_post_competition.csv
```

Each input filename stem gets its own output directory. Position indices are zero-based:

```text
/mnt/bn/sa-ag-data/leike/foa_dataset/random_pyroom/static/fsdkaggle2019/00097e21/0.flac
/mnt/bn/sa-ag-data/leike/foa_dataset/random_pyroom/static/fsdkaggle2019/00097e21/1.flac
...
/mnt/bn/sa-ag-data/leike/foa_dataset/random_pyroom/static/fsdkaggle2019/00097e21/4.flac
```

Successful outputs are indexed in:

```text
/mnt/bn/sa-ag-data/leike/foa_dataset/metadata/fsdkaggle2019_random_pyroom_static.jsonl
```

## Room and Position Sampling

The fixed pyroom room dimensions are `[11.0, 11.0, 5.0]` metres. The listener is fixed at the exact room centre, `[5.5, 5.5, 2.5]`.

For each CSV row, the script deterministically attempts five independently sampled source positions. Coordinates are sampled uniformly with a `0.001` metre inset from every wall, and the Euclidean distance between the absolute source position and the listener is at least `0.5` metres. A global CLI seed plus the stable CSV row index makes the positions reproducible regardless of worker count.

The metadata `position` uses absolute pyroom coordinates in metres: `x` is room length/right, `y` is room width/back, and `z` is height/up.

## FOA Rendering

The renderer follows the repository's established pyroom convention:

- Construct a `pyroomacoustics.ShoeBox` with a colocated omnidirectional microphone at the room centre.
- Compute and trim the centre RIR.
- Encode that RIR analytically as first-order ACN/SN3D channels in `W,Y,Z,X` order using the source direction.
- Convolve the resampled mono input with the four RIR channels.
- Peak-normalize all four channels together to `-1.0 dBFS`.
- Save the complete rendered signal, including its retained RIR tail, as 4-channel, 44.1 kHz, PCM16 FLAC.

For FOA encoding, the absolute pyroom offset `[dx, dy, dz]` is mapped to the established listener-relative convention `[right, up, back] = [dx, dz, dy]` before applying the `WYZX` gains. This keeps the new dataset consistent with the existing FOA VAE channel convention.

Rendering defaults are fixed to the requested values:

```text
sample_rate = 44100
absorption = 0.80
max_order = 2
sound_speed = 343.0
ir_seconds = 0.10
ir_trim_ms = 50.0
peak_dbfs = -1.0
foa_order = WYZX
```

## Metadata Schema

One JSON object is written for every successfully validated FLAC file:

```json
{
  "wav_path": "/mnt/bn/sa-ag-data/leike/foa_dataset/random_pyroom/static/fsdkaggle2019/00097e21/0.flac",
  "duration": 1.234,
  "position": {"x": 6.321, "y": 8.027, "z": 3.911},
  "source_wav_path": "/mnt/bn/sa-ag-data/leike/spatial_edit/dataset/fsdkaggle2019/FSDKaggle2019.audio_train_noisy/00097e21.wav",
  "source_fname": "00097e21.wav",
  "position_index": 0,
  "labels": ["Bathtub_(filling_or_washing)"]
}
```

`duration` is derived from the saved FLAC frame count rather than estimated from the input. The CSV `labels` field is split into a list. Relative position, source distance, sample rate, and renderer parameters are deliberately omitted from each row.

## Failure and Resume Semantics

Each output is rendered to a temporary file in its destination directory. The script validates sample rate, channel count, frame count, and non-silence before atomically replacing the final `0.flac` through `4.flac` path and returning a metadata row.

If any position attempt fails, its temporary and final output files are removed and no JSONL row is emitted. Other attempts for the same mono input continue, so an input may produce between zero and five FOA files.

The parent process is the only JSONL writer. It writes successful records in CSV order and then position-index order to a temporary metadata file, and atomically publishes the final JSONL after processing completes. Existing valid outputs can be reused with deterministic positions; invalid existing outputs are regenerated. Failures are reported to stderr and summarized at the end without adding extra records to the dataset metadata.

## Command-Line Interface

All input and output paths have the defaults above. The CLI also exposes `--num-workers`, `--seed`, `--limit`, and `--overwrite` for parallel generation, reproducibility, smoke tests, and explicit regeneration. Acoustic constants and room dimensions remain named module constants so tests and future controlled runs can inspect them without expanding the metadata schema.

## Tests and Verification

Focused tests cover:

- Exactly five deterministic positions per input, all strictly inside the room and at least `0.5` metres from the centre.
- Canonical source directions map to the expected `WYZX` gains.
- A FLAC round trip preserves four channels and 44.1 kHz and yields duration from saved frames.
- Successful records match the exact metadata schema and output path layout.
- A failed position removes its output and produces no metadata row while sibling attempts continue.

An end-to-end smoke test in the `spat_edit` environment renders one source with a single worker, then checks the generated FLAC files and JSONL rows independently.
