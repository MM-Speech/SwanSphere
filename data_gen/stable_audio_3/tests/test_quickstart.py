from pathlib import Path
from contextlib import redirect_stderr
import importlib.util
from io import StringIO
import sys
import unittest


class QuickstartTest(unittest.TestCase):
    @classmethod
    def quickstart_path(cls):
        return Path(__file__).resolve().parents[1] / "quickstart.py"

    @classmethod
    def load_quickstart(cls):
        path = cls.quickstart_path()
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("quickstart", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_quickstart_module_exists(self):
        self.assertTrue(
            self.quickstart_path().is_file(),
            f"missing {self.quickstart_path()}",
        )

    def test_default_arguments_are_runnable(self):
        args = self.load_quickstart().parse_args([])
        self.assertEqual(args.model, "medium")
        self.assertEqual(args.duration, 10.0)
        self.assertEqual(args.steps, 8)
        self.assertEqual(args.output, "output.wav")

    def test_non_positive_duration_is_rejected(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.load_quickstart().parse_args(["--duration", "0"])
        self.assertEqual(error.exception.code, 2)

    def test_non_finite_duration_is_rejected(self):
        for duration in ("nan", "inf", "-inf"):
            with self.subTest(duration=duration):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        self.load_quickstart().parse_args(
                            ["--duration", duration]
                        )
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
