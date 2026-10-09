"""Model cache and Docker run contracts; no daemon required."""

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import tasks


def docker_mounts(cmd):
    mounts = {}
    for i, arg in enumerate(cmd):
        if arg == "--mount":
            record = next(csv.reader([cmd[i + 1]]))
            fields = dict(field.split("=", 1) for field in record)
            mounts[fields["target"]] = Path(fields["source"])
    return mounts


class RunTests(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(tmp).resolve()
        self.home = self.root / "home"
        self.enterContext(patch.object(tasks, "ROOT", self.root))
        self.enterContext(patch.object(Path, "home", return_value=self.home))
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_cache_defaults_and_xdg_precedence(self):
        for env, base in [
            ({}, self.home / ".cache"),
            ({"XDG_CACHE_HOME": str(self.root / "xdg")}, self.root / "xdg"),
        ]:
            with self.subTest(env=env):
                self.assertEqual(
                    tasks.cache_paths(env),
                    {
                        "hf_hub": base / "huggingface/hub",
                        "torch": base / "torch",
                        "whisper": base / "whisper",
                        "gigaam": self.home / ".cache/gigaam",
                        "resonance": base / "resonance",
                        "sherpa": base / "resonance/sherpa",
                    },
                )

    def test_linux_root_cannot_start_container(self):
        with (
            patch.object(tasks.sys, "platform", "linux"),
            patch.object(tasks.os, "geteuid", return_value=0, create=True),
            patch.object(tasks.subprocess, "run") as docker,
            self.assertRaises(PermissionError),
        ):
            tasks.run_container({}, [])
        docker.assert_not_called()
        self.assertFalse((self.home / ".cache").exists())

    def test_gigaam_host_source_is_created_and_mounted(self):
        os.environ.update(HOME=str(self.home), USERPROFILE=str(self.home))
        for value, expected in [
            ("", self.home / ".cache/gigaam"),
            ("~/models/gigaam", self.home / "models/gigaam"),
            ("models/gigaam", self.root / "models/gigaam"),
        ]:
            with self.subTest(value=value):
                with patch.object(tasks.subprocess, "run") as docker:
                    tasks.run_container({"GIGAAM_CACHE_DIR": value}, [])
                cmd = docker.call_args.args[0]
                self.assertIn(expected, docker_mounts(cmd).values())
                self.assertTrue(expected.is_dir())

    def test_hf_hub_and_sherpa_precedence(self):
        cases = [
            ({"HF_HOME": str(self.root / "custom-hf")}, "hf_hub", "custom-hf/hub"),
            ({"HUGGINGFACE_HUB_CACHE": str(self.root / "legacy")}, "hf_hub", "legacy"),
            (
                {
                    "HF_HUB_CACHE": str(self.root / "new"),
                    "HUGGINGFACE_HUB_CACHE": str(self.root / "legacy"),
                },
                "hf_hub",
                "new",
            ),
            ({"RESONANCE_CACHE_DIR": str(self.root / "shared")}, "sherpa", "shared"),
            (
                {
                    "SHERPA_HOME": str(self.root / "sherpa"),
                    "RESONANCE_CACHE_DIR": str(self.root / "shared"),
                },
                "sherpa",
                "sherpa",
            ),
            (
                {"SHERPA_HOME": "", "RESONANCE_CACHE_DIR": str(self.root / "shared")},
                "sherpa",
                "shared",
            ),
        ]
        for env, name, expected in cases:
            with self.subTest(env=env):
                self.assertEqual(tasks.cache_paths(env)[name], self.root / expected)

    def test_mounts_cover_runtime_caches_before_docker_starts(self):
        for env in (
            {},
            {"HF_HUB_CACHE": str(self.root / "external-hub")},
            {"SHERPA_HOME": str(self.root / "external-sherpa")},
        ):
            with self.subTest(env=env):

                def boundary(cmd, **kwargs):
                    for source in docker_mounts(cmd).values():
                        self.assertTrue(source.is_dir())

                with patch.object(tasks.subprocess, "run", side_effect=boundary):
                    tasks.run_container(env, [])

    def test_cli_options_precede_image(self):
        options = ["--label", "note=hello world"]
        with (
            patch.object(tasks.sys, "argv", ["tasks.py", "run", *options]),
            patch.object(tasks.subprocess, "run") as docker,
        ):
            self.assertEqual(tasks.main(), 0)
        cmd = docker.call_args.args[0]
        self.assertEqual(cmd[-len(options) - 1 : -1], options)

    def test_windows_mount_with_spaces_and_comma_round_trips(self):
        source = r"C:\Users\rosy\Model Cache, shared\huggingface"
        target = "/model-cache"
        spec = tasks.bind_mount(Path(source), target)
        self.assertEqual(
            docker_mounts(["--mount", spec]),
            {target: Path(source)},
        )


if __name__ == "__main__":
    unittest.main()
