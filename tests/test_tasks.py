"""Run with: uv run --no-project --python 3.13 --with python-dotenv --with packaging \
python -m unittest discover -s tests -p test_tasks.py -v
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import tasks

ROOT = Path(__file__).resolve().parents[1]


class TaskTests(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(tmp) / "repo"
        self.root.mkdir()
        self.enterContext(patch.object(tasks, "ROOT", self.root))
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_missing_env_and_parent_env_are_ignored(self):
        (self.root.parent / ".env").write_text("DEVICE=cuda\n", encoding="utf-8")
        env = tasks.task_environment()
        self.assertNotIn("DEVICE", env)
        self.assertEqual(env["PYTHONUTF8"], "1")

    def test_repo_env_is_loaded(self):
        (self.root / ".env").write_text("DEVICE=mps\n", encoding="utf-8")
        self.assertEqual(tasks.task_environment()["DEVICE"], "mps")

    def test_host_environment_wins(self):
        (self.root / ".env").write_text("DEVICE=mps\n", encoding="utf-8")
        os.environ["DEVICE"] = "cuda"
        self.assertEqual(tasks.task_environment()["DEVICE"], "cuda")

    def test_backend_policy(self):
        cases = [
            ({}, "cpu"),
            ({"DEVICE": "cpu"}, "cpu"),
            ({"DEVICE": "cuda"}, ""),
            ({"DEVICE": "mps"}, ""),
            ({"PYTORCH_BACKEND": ""}, ""),
            ({"DEVICE": "cuda", "PYTORCH_BACKEND": "cu126"}, "cu126"),
        ]
        for env, expected in cases:
            with self.subTest(env=env):
                self.assertEqual(tasks.torch_backend(env), expected)


@unittest.skipUnless(
    shutil.which("uv") and sys.version_info >= (3, 11),
    "integration requires uv and Python >=3.11; see module command",
)
class Pep723Tests(unittest.TestCase):
    def test_project_install_and_python_constraints_with_external_environment(self):
        import tomllib
        from packaging.specifiers import SpecifierSet

        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        requires_python = project["requires-python"]
        allowed_python = SpecifierSet(requires_python)
        self.assertNotIn(
            "3.13", allowed_python, "fixture needs a runner outside project constraints"
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith("UV_")}
        env.pop("VIRTUAL_ENV", None)
        env.pop("CONDA_PREFIX", None)
        env.update(UV_NO_CONFIG="1", UV_PYTHON_DOWNLOADS="never")
        with tempfile.TemporaryDirectory(prefix="resonance-bootstrap-") as tmp:
            base = Path(tmp)
            external = base / "external"
            self.run_command(
                ["uv", "venv", "--python", "3.13", str(external)], base, env
            )
            for activation in ("VIRTUAL_ENV", "CONDA_PREFIX"):
                with self.subTest(activation=activation):
                    root = base / activation
                    (root / "scripts").mkdir(parents=True)
                    shutil.copy2(
                        ROOT / "scripts" / "tasks.py", root / "scripts" / "tasks.py"
                    )
                    (root / "pyproject.toml").write_text(
                        '[project]\nname="bootstrap-test"\nversion="0.0.0"\n'
                        f'requires-python="{requires_python}"\n'
                        'dependencies=["python-dotenv>=1.0.0"]\n'
                        "[dependency-groups]\ndev=[]\n",
                        encoding="utf-8",
                    )
                    child_env = dict(env, **{activation: str(external)})
                    command = ["uv", "run", "--python", "3.13", "--script"]
                    self.run_command(
                        command + ["scripts/tasks.py", "dev-deps"], root, child_env
                    )
                    version = self.run_python(
                        root / ".venv",
                        "import dotenv, platform; print(platform.python_version())",
                        env,
                    )
                    self.assertIn(version.strip(), allowed_python)
                    self.run_python(
                        external,
                        "import importlib.util; assert importlib.util.find_spec('dotenv') is None",
                        env,
                    )

    def run_command(self, cmd, cwd, env):
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def run_python(self, venv, code, env):
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        return self.run_command([str(python), "-c", code], venv.parent, env)


if __name__ == "__main__":
    unittest.main()
