import ast
from pathlib import Path
import subprocess
import sys
import unittest


class ImportPolicyTest(unittest.TestCase):
    @staticmethod
    def is_module_path_mutation(node):
        function = node.func
        return (
            isinstance(function, ast.Attribute)
            and function.attr in {"append", "extend", "insert"}
            and isinstance(function.value, ast.Attribute)
            and function.value.attr == "path"
            and isinstance(function.value.value, ast.Name)
            and function.value.value.id == "sys"
        )

    @staticmethod
    def is_working_directory_change(node):
        function = node.func
        return (
            isinstance(function, ast.Attribute)
            and function.attr == "chdir"
            and isinstance(function.value, ast.Name)
            and function.value.id == "os"
        )

    def test_all_project_imports_are_absolute(self):
        target = Path(__file__).resolve().parents[1]
        violations = []

        for path in target.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if (
                        node.level
                        or module == "stable_audio_3"
                        or module.startswith("stable_audio_3.")
                    ):
                        violations.append(f"{path}:{node.lineno}: import")
                if isinstance(node, ast.Call):
                    if self.is_module_path_mutation(
                        node
                    ) or self.is_working_directory_change(node):
                        violations.append(f"{path}:{node.lineno}: path mutation")

        self.assertEqual(violations, [])

    def test_quickstart_help_runs_from_repository_root(self):
        project_root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            [
                sys.executable,
                "data_gen/stable_audio_3/quickstart.py",
                "--help",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
