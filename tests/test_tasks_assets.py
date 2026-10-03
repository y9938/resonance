"""Generated web icon formats and dimensions."""

import contextlib
import io
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import tasks


class AssetTests(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("resvg") and shutil.which("magick"), "Requires resvg and magick"
    )
    def test_real_web_icon_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "public").mkdir()
            shutil.copy(tasks.ROOT / "public/icon.svg", root / "public/icon.svg")
            with (
                patch.object(tasks, "ROOT", root),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                tasks.icons(os.environ.copy())
            for size in (16, 32, 48, 180):
                path = root / (
                    f"build/favicons/{size}.png"
                    if size != 180
                    else "public/apple-touch-icon.png"
                )
                self.assertEqual(
                    struct.unpack(">II", path.read_bytes()[16:24]), (size, size)
                )
            favicon = (root / "public/favicon.ico").read_bytes()
            self.assertEqual(struct.unpack("<HHH", favicon[:6]), (0, 1, 3))
            self.assertEqual(
                [(favicon[6 + i * 16], favicon[7 + i * 16]) for i in range(3)],
                [(16, 16), (32, 32), (48, 48)],
            )
