"""Streaming gzip output and preservation of archives on failed exports."""

import contextlib
import gzip
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import tasks

POPEN = subprocess.Popen


class SaveTests(unittest.TestCase):
    def producer(self, code):
        def start(cmd, **kwargs):
            return POPEN([sys.executable, "-c", code], **kwargs)

        return patch.object(tasks.subprocess, "Popen", side_effect=start)

    def test_success_replaces_archive_with_gzip_stream(self):
        for config, filename in [
            ({}, "resonance_latest.tar.gz"),
            ({"DEVICE": "mps"}, "resonance_latest.tar.gz"),
            (
                {"DEVICE": "cuda", "IMAGE_TAG": "probe"},
                "resonance_gpu_probe.tar.gz",
            ),
        ]:
            with (
                self.subTest(config=config),
                tempfile.TemporaryDirectory() as directory,
                patch.object(tasks, "ROOT", Path(directory)),
                self.producer(
                    "import sys; sys.stdout.buffer.write(b'exported tar')"
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                archive = Path(directory) / filename
                archive.write_bytes(b"previous archive")
                tasks.save_image(
                    dict(os.environ, DEVICE="cpu", IMAGE_TAG="latest") | config
                )
                with gzip.open(archive, "rb") as saved:
                    self.assertEqual(saved.read(), b"exported tar")
                self.assertEqual(list(Path(directory).iterdir()), [archive])

    def test_failure_or_interrupt_preserves_previous_archive(self):
        for interrupt in (False, True):
            with (
                self.subTest(interrupt=interrupt),
                tempfile.TemporaryDirectory() as directory,
                patch.object(tasks, "ROOT", Path(directory)),
                self.producer(
                    "raise SystemExit(7)"
                    if interrupt
                    else "import sys; sys.stdout.buffer.write(b'partial tar'); sys.exit(7)",
                ),
                contextlib.ExitStack() as stack,
            ):
                if interrupt:
                    stack.enter_context(
                        patch.object(
                            tasks.shutil, "copyfileobj", side_effect=KeyboardInterrupt
                        )
                    )
                archive = Path(directory) / "resonance_latest.tar.gz"
                archive.write_bytes(b"previous archive")
                with self.assertRaises(
                    KeyboardInterrupt if interrupt else subprocess.CalledProcessError
                ):
                    tasks.save_image(dict(os.environ, DEVICE="cpu", IMAGE_TAG="latest"))
                self.assertEqual(archive.read_bytes(), b"previous archive")
                self.assertEqual(list(Path(directory).iterdir()), [archive])
