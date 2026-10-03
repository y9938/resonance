"""Opt-in boundary test: RESONANCE_DOCKER_TESTS=1; requires a local Linux daemon."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import tasks


@unittest.skipUnless(
    sys.platform == "linux" and os.getenv("RESONANCE_DOCKER_TESTS") == "1",
    "requires opt-in and a native Linux Docker daemon without UID remapping",
)
class NativeDockerTests(unittest.TestCase):
    def test_container_writes_caches_without_sharing_host_hf_token(self):
        device = os.getenv("RESONANCE_DOCKER_DEVICE", "cpu")
        with tempfile.TemporaryDirectory(prefix="resonance-docker-test-") as tmp:
            root = Path(tmp)
            hf_home = root / "huggingface"
            hf_home.mkdir()
            (hf_home / "token").write_text("host-only", encoding="utf-8")
            env = dict(
                os.environ,
                DEVICE=device,
                IMAGE_TAG="latest",
                HF_HOME=str(hf_home),
                HF_HUB_CACHE=str(hf_home / "hub"),
                XDG_CACHE_HOME=str(root),
                GIGAAM_CACHE_DIR=str(root / "gigaam"),
                SHERPA_HOME=str(root / "sherpa"),
                RESONANCE_CACHE_DIR="",
            )
            with (
                patch.object(tasks, "ROOT", root),
                patch.object(tasks.subprocess, "run") as docker,
            ):
                tasks.run_container(env, [])
            cmd = docker.call_args.args[0]
            # Exercise the runner's mounts/env/user without a terminal or server.
            cmd.remove("-it")
            for option in ("--name", "-p"):
                index = cmd.index(option)
                del cmd[index : index + 2]
            probe = """
import os
from pathlib import Path
hf = Path(os.environ['HF_HOME'])
assert not (hf / 'token').exists()
hf.mkdir(parents=True, exist_ok=True)
for name in ('huggingface/hub', 'torch', 'whisper', 'gigaam', 'resonance', 'sherpa'):
    (Path(os.environ['XDG_CACHE_HOME']) / name / 'from-container').write_text('ok')
for name in ('xet', 'assets'):
    (hf / name).mkdir()
(hf / 'token').write_text('container-only')
if os.environ['DEVICE'] == 'cuda':
    import torch
    assert torch.cuda.is_available()
    assert torch.ones(1, device='cuda').sum().item() == 1
"""
            subprocess.run(cmd + ["python", "-c", probe], check=True)
            for name in (
                "huggingface/hub",
                "torch",
                "whisper",
                "gigaam",
                "resonance",
                "sherpa",
            ):
                marker = root / name / "from-container"
                self.assertEqual(marker.read_text(), "ok")
                self.assertEqual(marker.stat().st_uid, os.getuid())
                self.assertEqual(marker.stat().st_gid, os.getgid())
            self.assertEqual((hf_home / "token").read_text(), "host-only")
            self.assertFalse((hf_home / "xet").exists())
            self.assertFalse((hf_home / "assets").exists())
