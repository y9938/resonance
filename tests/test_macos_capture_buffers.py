"""Run the app's Swift audio assembler on macOS, without devices or permissions."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.tasks import macos_sdk_env

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Native macOS capture test")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def native_capture(tmp_path_factory):
    executable = tmp_path_factory.mktemp("native-capture") / "capture-buffers"
    env = macos_sdk_env(os.environ.copy())
    subprocess.run(
        ["swiftc", "-sdk", env["SDKROOT"], str(ROOT / "src/macos/CaptureBuffer.swift"),
         str(ROOT / "tests/macos/main.swift"), "-o", str(executable)],
        env=env, check=True, capture_output=True, text=True, timeout=60,
    )
    return executable


@pytest.mark.parametrize("mode", ["system", "dual", "quiet_system"])
def test_native_capture_preserves_sources_and_microphone_clock(native_capture, mode):
    frames = 4096
    phase = np.arange(frames, dtype=np.float32)
    system = (0.25 * np.sin(phase / 16)).astype(np.float32)
    microphone = (0.75 * np.cos(phase / 32)).astype(np.float32)
    channels = 1 if mode == "system" else 2
    batches = [{"sys": system.tolist(), "mic": [] if channels == 1 else microphone.tolist()}]
    expected = [system[:, None] if channels == 1 else np.column_stack((system, microphone))]
    if mode == "quiet_system":
        # MIC must flow before playback starts and after it stops; the system sends no callbacks.
        batches.insert(0, {"sys": [], "mic": microphone.tolist()})
        batches.append({"sys": [], "mic": microphone.tolist()})
        quiet = np.column_stack((np.zeros(frames, dtype=np.float32), microphone))
        expected = [quiet, *expected, quiet]

    result = subprocess.run(
        [str(native_capture)], input=json.dumps({"channels": channels, "frames_per_slot": frames,
                                                "batches": batches}),
        check=True, capture_output=True, text=True, timeout=10,
    )
    published = json.loads(result.stdout)
    assert len(published) == len(expected)
    for slots, samples in zip(published, expected):
        assert len(slots) == 1, "Capture must publish a slot for every full source buffer"
        actual = np.asarray(slots[0], dtype=np.float32).reshape(frames, channels)
        np.testing.assert_allclose(actual, samples, atol=1e-7)
