import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inference.foa_vae.infer_wavvae_v4 import (
    DEFAULT_CONFIG,
    collect_input_paths,
    pad_to_multiple,
    reconstruct,
    resolve_checkpoint_path,
    resolve_output_wav_path,
)


class ToyV4Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = SimpleNamespace(
            config=SimpleNamespace(downsampling_ratios=[2, 4, 4, 8, 8])
        )
        self.seen_length = None

    def forward(self, audio):
        self.seen_length = audio.shape[-1]
        return {"recon": audio.clone()}


class InferWavVAEV4Test(unittest.TestCase):
    def test_script_is_independent_and_uses_direct_v4_builder(self):
        source = Path(REPO_ROOT, "inference/foa_vae/infer_wavvae_v4.py").read_text(encoding="utf-8")

        self.assertNotIn("infer_wavvae import", source)
        self.assertNotIn("resolve_model_module", source)
        self.assertIn("from modules.foa_vae.wavvae_v4 import build_foa_wavvae_v4", source)
        self.assertIn("strict=True", source)

    def test_default_config_is_explicitly_v4_and_strict(self):
        self.assertEqual(DEFAULT_CONFIG, "egs/inference/inference_foa_vae_v4.yaml")
        config_path = Path(REPO_ROOT) / DEFAULT_CONFIG
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["model_config"], "egs/foa_vae/wavvae_v4.yaml")
        self.assertEqual(config["expected_channels"], 4)
        self.assertEqual(config["pad_to_multiple"], 2048)
        self.assertIs(config["load_ckpt_strict"], True)
        self.assertEqual(config["ckpt_path"], "")

    def test_pad_to_multiple_and_reconstruct_trim_to_original_length(self):
        wav = torch.randn(4, 5000)
        padded, original_length = pad_to_multiple(wav, 2048)

        self.assertEqual(original_length, 5000)
        self.assertEqual(padded.shape[-1], 6144)

        model = ToyV4Model()
        reconstruction = reconstruct(
            model=model,
            wav=wav,
            device="cpu",
            precision="fp32",
            pad_multiple=2048,
        )

        self.assertEqual(model.seen_length, 6144)
        self.assertEqual(tuple(reconstruction.shape), (4, 5000))
        self.assertTrue(torch.equal(reconstruction, wav))

    def test_collect_input_paths_supports_paths_lists_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_list = tmpdir / "inputs.txt"
            input_list.write_text("/list/a.wav\n/list/b.wav\n", encoding="utf-8")
            metadata = tmpdir / "items.jsonl"
            metadata.write_text(
                json.dumps({"wav_path": "/ignored.wav", "target_wav_path": "/target.wav"}) + "\n",
                encoding="utf-8",
            )

            paths = collect_input_paths(
                {
                    "input_paths": ["/direct.wav"],
                    "input_list": str(input_list),
                    "metadata_path": str(metadata),
                    "path_fields": ["target_wav_path", "wav_path"],
                }
            )

        self.assertEqual(paths, ["/direct.wav", "/list/a.wav", "/list/b.wav", "/target.wav"])

    def test_checkpoint_and_output_resolution_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "model_ckpt_steps_10.ckpt").write_text("x", encoding="utf-8")
            (tmpdir / "model_ckpt_steps_200.ckpt").write_text("x", encoding="utf-8")

            self.assertEqual(
                resolve_checkpoint_path(str(tmpdir)),
                str(tmpdir / "model_ckpt_steps_200.ckpt"),
            )
            self.assertEqual(
                resolve_output_wav_path("/audio/foa.flac", str(tmpdir), 0, 2, "_v4"),
                str(tmpdir / "foa_v4.wav"),
            )


if __name__ == "__main__":
    unittest.main()
