"""Docker build backend and passthrough contracts."""

import unittest
from unittest.mock import patch

from scripts import tasks


class BuildTests(unittest.TestCase):
    def test_backend_policy(self):
        for env, expected in [
            ({}, "cpu"),
            ({"DEVICE": "mps", "PYTORCH_BACKEND": "cu130"}, "cpu"),
            ({"DEVICE": "cpu", "PYTORCH_BACKEND": ""}, "cpu"),
            ({"DEVICE": "cuda"}, ""),
            ({"DEVICE": "cuda", "PYTORCH_BACKEND": ""}, ""),
            ({"DEVICE": "cuda", "PYTORCH_BACKEND": "cu130"}, "cu130"),
        ]:
            with self.subTest(env=env):
                self.assertEqual(tasks.build_backend(env), expected)

    def test_build_passes_options_before_context(self):
        env = {"DEVICE": "cuda", "IMAGE_TAG": "probe"}
        options = ["--label", "note=hello world", "--no-cache"]
        with (
            patch.object(tasks, "task_environment", return_value=env),
            patch.object(tasks.sys, "argv", ["tasks.py", "build", *options]),
            patch.object(tasks.subprocess, "run") as docker,
        ):
            self.assertEqual(tasks.main(), 0)
        cmd = docker.call_args.args[0]
        self.assertEqual(cmd[-len(options) - 1 : -1], options)
