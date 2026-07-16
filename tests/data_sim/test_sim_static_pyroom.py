import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data_sim.sim_static_pyroom import (  # noqa: E402
    DEFAULT_LISTENER_POSITION,
    DEFAULT_ROOM_DIM,
    DEFAULT_SAMPLE_RATE,
    build_metadata_record,
    foa_gains_from_room_position,
    output_path_for_item,
    parse_labels,
    render_foa,
    sample_source_positions,
    write_validated_flac,
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

    def test_room_and_listener_constants_match_the_approved_design(self):
        np.testing.assert_array_equal(DEFAULT_ROOM_DIM, np.array([11.0, 11.0, 5.0]))
        np.testing.assert_array_equal(
            DEFAULT_LISTENER_POSITION,
            np.array([5.5, 5.5, 2.5]),
        )

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

    def test_labels_are_split_and_empty_values_are_dropped(self):
        self.assertEqual(
            parse_labels("Dog, Marimba_and_xylophone,"),
            ["Dog", "Marimba_and_xylophone"],
        )
        self.assertEqual(parse_labels(None), [])


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


if __name__ == "__main__":
    unittest.main()
