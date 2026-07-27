import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.foa_vae.wavvae_v4 import (
    DEFAULT_V4_DOWNSAMPLING_RATIOS,
    V4_SHAPE_MISMATCH_KEYS,
    FOAWavVAEV4,
    _validate_source_config,
    _validate_target_settings,
    adapt_oobleck_v4_state_dict,
)


class StateModel(nn.Module):
    def __init__(self, state):
        super().__init__()
        self._test_state = state

    def state_dict(self, *args, **kwargs):
        return self._test_state


def make_states():
    source = {
        "encoder.conv1.weight_v": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
        "encoder.conv1.weight_g": torch.tensor([1.0, 2.0]).reshape(2, 1, 1),
        "encoder.conv1.bias": torch.tensor([0.25, -0.25]),
        "encoder.block.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "decoder.conv2.weight_v": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
        "decoder.conv2.weight_g": torch.tensor([3.0, 4.0]).reshape(2, 1, 1),
    }
    target = {
        "encoder.conv1.weight_v": torch.randn(2, 4, 3),
        "encoder.conv1.weight_g": torch.randn(2, 1, 1),
        "encoder.conv1.bias": torch.randn(2),
        "encoder.block.weight": torch.randn(2, 2),
        "decoder.conv2.weight_v": torch.randn(4, 3, 3),
        "decoder.conv2.weight_g": torch.randn(4, 1, 1),
    }
    return target, source


class ToyLatentDistribution:
    def __init__(self, latents):
        self.mean = latents
        self.logvar = torch.zeros_like(latents)

    def sample(self):
        return self.mean

    def kl(self):
        return torch.ones(self.mean.shape[0], self.mean.shape[-1], device=self.mean.device)


class ToyAutoencoder(nn.Module):
    def encode(self, audio):
        return SimpleNamespace(latent_dist=ToyLatentDistribution(audio[:, :2]))

    def decode(self, latents):
        return SimpleNamespace(sample=torch.cat([latents, latents], dim=1))


class WavVAEV4StateAdaptationTest(unittest.TestCase):
    def test_expected_mismatch_allowlist_is_exact(self):
        self.assertEqual(
            V4_SHAPE_MISMATCH_KEYS,
            {
                "encoder.conv1.weight_v",
                "decoder.conv2.weight_v",
                "decoder.conv2.weight_g",
            },
        )

    def test_equal_shapes_are_copied_and_encoder_direction_stays_random(self):
        target, source = make_states()
        encoder_direction = target["encoder.conv1.weight_v"].clone()

        patched = adapt_oobleck_v4_state_dict(StateModel(target), source, verbose=False)

        self.assertTrue(torch.equal(patched["encoder.block.weight"], source["encoder.block.weight"]))
        self.assertTrue(torch.equal(patched["encoder.conv1.weight_g"], source["encoder.conv1.weight_g"]))
        self.assertTrue(torch.equal(patched["encoder.conv1.bias"], source["encoder.conv1.bias"]))
        self.assertTrue(torch.equal(patched["encoder.conv1.weight_v"], encoder_direction))
        self.assertGreater(float(patched["encoder.conv1.weight_v"][:, 2:].abs().sum()), 0.0)

    def test_decoder_directions_are_orthogonal_and_not_repeated(self):
        target, source = make_states()

        patched = adapt_oobleck_v4_state_dict(StateModel(target), source, verbose=False)
        directions = patched["decoder.conv2.weight_v"].flatten(1)
        normalized = torch.nn.functional.normalize(directions, dim=1)
        gram = normalized @ normalized.T

        self.assertTrue(torch.allclose(gram, torch.eye(4), atol=1.0e-5, rtol=1.0e-5))
        self.assertFalse(torch.equal(directions[0], directions[2]))
        self.assertFalse(torch.equal(directions[1], directions[3]))

    def test_decoder_gain_is_symmetric_nonzero_and_energy_preserving(self):
        target, source = make_states()

        patched = adapt_oobleck_v4_state_dict(StateModel(target), source, verbose=False)
        gain = patched["decoder.conv2.weight_g"]

        self.assertTrue(torch.all(gain > 0))
        self.assertTrue(torch.allclose(gain, gain[:1].expand_as(gain)))
        self.assertTrue(torch.allclose(gain.square().sum(), source["decoder.conv2.weight_g"].square().sum()))
        self.assertAlmostEqual(float(gain[0]), 2.5, places=6)

    def test_unexpected_shape_mismatch_raises(self):
        target, source = make_states()
        target["encoder.block.weight"] = torch.randn(3, 2)

        with self.assertRaisesRegex(RuntimeError, "Unexpected v4 shape mismatch"):
            adapt_oobleck_v4_state_dict(StateModel(target), source, verbose=False)

    def test_missing_source_parameter_raises(self):
        target, source = make_states()
        del source["encoder.block.weight"]

        with self.assertRaisesRegex(RuntimeError, "Missing Stable Audio parameter"):
            adapt_oobleck_v4_state_dict(StateModel(target), source, verbose=False)


class WavVAEV4ArchitectureTest(unittest.TestCase):
    def test_target_settings_are_fixed_to_native_4ch_ds2048_z64(self):
        _validate_target_settings(4, 4, 64, [2, 4, 4, 8, 8])
        self.assertEqual(DEFAULT_V4_DOWNSAMPLING_RATIOS, [2, 4, 4, 8, 8])

        invalid = (
            (2, 4, 64, [2, 4, 4, 8, 8]),
            (4, 4, 128, [2, 4, 4, 8, 8]),
            (4, 4, 64, [2, 2, 4, 8, 8]),
        )
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                _validate_target_settings(*args)

    def test_source_config_must_be_original_stable_audio_layout(self):
        valid = {
            "audio_channels": 2,
            "decoder_input_channels": 64,
            "downsampling_ratios": [2, 4, 4, 8, 8],
        }
        _validate_source_config(valid)

        for key, value in (
            ("audio_channels", 4),
            ("decoder_input_channels", 128),
            ("downsampling_ratios", [2, 2, 4, 8, 8]),
        ):
            invalid = dict(valid)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                _validate_source_config(invalid)

    def test_wrapper_has_no_learnable_projectors_and_returns_task_contract(self):
        model = FOAWavVAEV4(ToyAutoencoder())
        audio = torch.randn(2, 4, 16)

        outputs = model(audio)

        self.assertEqual(tuple(outputs["recon"].shape), (2, 4, 16))
        self.assertEqual(tuple(outputs["mu"].shape), (2, 2, 16))
        self.assertEqual(float(outputs["kl"]), 1.0)
        self.assertEqual(set(dict(model.named_children())), {"autoencoder"})

    def test_module_does_not_import_old_versions_or_define_projectors(self):
        source = Path(REPO_ROOT, "modules/foa_vae/wavvae_v4.py").read_text(encoding="utf-8")
        self.assertNotIn("wavvae_v1", source)
        self.assertNotIn("wavvae_v2", source)
        self.assertNotIn("wavvae_v3", source)
        self.assertNotIn("Projector", source)


if __name__ == "__main__":
    unittest.main()
