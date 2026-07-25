from pathlib import Path
import importlib
import sys
import unittest


class RuntimeImportTest(unittest.TestCase):
    def test_runtime_imports_from_vendored_package(self):
        runtime_root = Path(__file__).resolve().parents[1]
        package_init = runtime_root / "stable_audio_3" / "__init__.py"
        self.assertTrue(package_init.is_file(), f"missing {package_init}")

        sys.path.insert(0, str(runtime_root))
        sys.modules.pop("stable_audio_3", None)
        module = importlib.import_module("stable_audio_3")

        self.assertTrue(Path(module.__file__).resolve().is_relative_to(runtime_root))
        self.assertIsNotNone(module.StableAudioModel)


if __name__ == "__main__":
    unittest.main()
