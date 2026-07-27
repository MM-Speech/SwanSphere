import os
import sys
import unittest
from pathlib import Path

import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.foa_vae.dataset_utils.pyroom_dataset import compute_target_num_samples


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


if __name__ == "__main__":
    unittest.main()
