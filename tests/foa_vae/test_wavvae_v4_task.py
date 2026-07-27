import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.foa_vae.dataset_utils.pyroom_dataset import compute_target_num_samples
from tasks.foa_vae.wavvae_v4_task import (
    FOAWavVAEV4Task,
    HighFrequencyExcessDBLoss,
    TensorStateEMAModel,
    trim_to_shortest,
)
from utils.commons.hparams import hparams


class WavVAEV4ConfigTest(unittest.TestCase):
    def test_training_config_matches_approved_contract(self):
        config_path = Path(REPO_ROOT) / "egs/foa_vae/wavvae_v4.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["task_cls"], "tasks.foa_vae.wavvae_v4_task.FOAWavVAEV4Task")
        self.assertEqual(
            config["foa_vae"],
            {
                "pretrained_model_dir": "checkpoints/vae",
                "input_channels": 4,
                "output_channels": 4,
                "latent_channels": 64,
                "downsampling_ratios": [2, 4, 4, 8, 8],
            },
        )
        self.assertEqual(
            config["losses"],
            {
                "lambda_mrstft": 1.0,
                "lambda_high_frequency_excess_db": 0.01,
                "lambda_kl": 1.0e-5,
                "lambda_mel_multiband": 0.0,
                "lambda_adv": 0.05,
                "lambda_feature_matching": 5.0,
                "lambda_dis": 1.0,
            },
        )
        self.assertNotIn("foa_spatial_loss", config)
        self.assertFalse(any("direction" in key or "cov" in key or "phase" in key for key in config["losses"]))

    def test_training_crop_is_exactly_65536_samples_and_32_latent_frames(self):
        config_path = Path(REPO_ROOT) / "egs/foa_vae/wavvae_v4.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        target_samples = compute_target_num_samples(
            sample_rate=config["sample_rate"],
            target_seconds=config["foa_target_seconds"],
            frames_multiple=config["frames_multiple"],
            hop_size=config["hop_size"],
            align_to_frames_multiple=True,
        )

        self.assertEqual(config["foa_target_seconds"], 1.5)
        self.assertEqual(config["sample_size"], 65536)
        self.assertEqual(target_samples, 65536)
        self.assertEqual(target_samples // 2048, 32)


class WavVAEV4TaskBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.original_hparams = dict(hparams)

    def tearDown(self):
        hparams.clear()
        hparams.update(self.original_hparams)

    def test_task_is_independent_and_builds_only_v4(self):
        source = Path(REPO_ROOT, "tasks/foa_vae/wavvae_v4_task.py").read_text(encoding="utf-8")

        self.assertNotIn("wavvae_v1", source)
        self.assertNotIn("wavvae_v2", source)
        self.assertNotIn("wavvae_v3", source)
        self.assertIn("from modules.foa_vae.wavvae_v4 import", source)
        self.assertIn("build_foa_wavvae_v4", source)
        self.assertNotIn("FOAWavVAEV3Task", source)

    def test_gan_ramp_has_10k_pretrain_and_20k_ramp(self):
        hparams.clear()
        hparams.update({"gan_ramp_steps": 20000})
        task = object.__new__(FOAWavVAEV4Task)

        for step, expected in ((9999, 0.0), (10000, 0.0), (20000, 0.5), (30000, 1.0), (40000, 1.0)):
            with self.subTest(step=step):
                task.global_step = step
                self.assertAlmostEqual(task._gan_ramp(10000), expected)

    def test_tensor_state_ema_round_trips_without_python_values(self):
        parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
        source = TensorStateEMAModel([parameter], decay=0.9, update_after_step=3)
        state = source.state_dict()

        self.assertTrue(all(torch.is_tensor(value) for value in state.values()))

        target = TensorStateEMAModel([nn.Parameter(torch.zeros(2))], decay=0.1)
        target.load_state_dict(state, strict=True)

        self.assertAlmostEqual(target.decay, 0.9, places=6)
        self.assertEqual(target.update_after_step, 3)
        self.assertTrue(torch.equal(target.shadow_params[0], source.shadow_params[0]))

    def test_trim_to_shortest_never_pads_audio(self):
        longer = torch.randn(1, 4, 20)
        shorter = torch.randn(1, 4, 16)

        first, second = trim_to_shortest(longer, shorter)
        self.assertEqual(first.shape[-1], 16)
        self.assertEqual(second.shape[-1], 16)

        first, second = trim_to_shortest(shorter, longer)
        self.assertEqual(first.shape[-1], 16)
        self.assertEqual(second.shape[-1], 16)

    def test_high_frequency_loss_is_zero_for_identical_audio(self):
        loss_fn = HighFrequencyExcessDBLoss(
            sample_rate=16000,
            min_frequency_hz=6000.0,
            margin_db=1.0,
            fft_size=512,
            hop_size=128,
            win_length=512,
            min_db=-80.0,
        )
        audio = torch.randn(1, 4, 4096) * 0.01

        self.assertEqual(float(loss_fn(audio, audio)), 0.0)


if __name__ == "__main__":
    unittest.main()
