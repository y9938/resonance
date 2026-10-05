"""Development reload behavior and cleanup of required processes."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import tasks


def test_dev_reloads_backend_python_but_not_frontend_edits(tmp_path):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    for name in ("core", "stt", "tts", "src/web", "node_modules/package", "dist/assets"):
        (root / name).mkdir(parents=True)
    backend = root / "server.py"
    backend.write_text('''
import os
from pathlib import Path
from fastapi import FastAPI
def create_local_app():
    with Path("starts").open("a") as output:
        output.write(str(os.getpid()) + "\\n")
    return FastAPI()
''')
    starts = root / "starts"
    command = tasks.local_server_command({"RESONANCE_PORT": "0"}, reload=True)
    # Use the test environment's Python, retaining the actual server arguments.
    command = [sys.executable, *command[4:]]
    with (root / "server.log").open("w") as log, subprocess.Popen(
        command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    ) as runner:
        try:
            deadline = time.monotonic() + 10
            while not starts.exists():
                assert runner.poll() is None, (root / "server.log").read_text()
                assert time.monotonic() < deadline, "backend did not start"
                time.sleep(0.05)
            time.sleep(0.5)  # Allow the real filesystem watcher to start.
            initial = starts.read_text()
            for name in ("src/web/App.svelte", "src/web/main.ts", "node_modules/package/index.js", "dist/assets/index.js"):
                (root / name).write_text("// frontend edit\n")
            time.sleep(1.5)
            assert runner.poll() is None
            assert starts.read_text() == initial

            backend.write_text(backend.read_text() + "\n# backend edit\n")
            deadline = time.monotonic() + 10
            while starts.read_text() == initial:
                assert runner.poll() is None, (root / "server.log").read_text()
                assert time.monotonic() < deadline, "backend edit did not trigger reload"
                time.sleep(0.05)
        finally:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(runner.pid), "/T", "/F"], check=False, capture_output=True)
            else:
                os.killpg(runner.pid, signal.SIGTERM)
            try:
                runner.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    runner.kill()
                else:
                    os.killpg(runner.pid, signal.SIGKILL)
                runner.wait()

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group integration; Windows uses taskkill")
@pytest.mark.parametrize("interrupt", [True, False])
def test_dev_cleans_up_required_processes(tmp_path, interrupt):
    worker = tmp_path / "worker.py"
    worker.write_text('''
import os, signal, sys, time
from pathlib import Path
name = sys.argv[1]
Path(name + ".pid").write_text(str(os.getpid()))
def stop(*args):
    Path(name + ".stopped").touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
while not Path("exit").exists():
    time.sleep(0.02)
if name == "backend":
    raise SystemExit(7)
while True:
    time.sleep(0.02)
''')
    harness = tmp_path / "harness.py"
    harness.write_text(f'''
import os, subprocess, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
from scripts import tasks
tasks.ROOT = Path({str(tmp_path)!r})
tasks.npm_command = lambda env: "frontend"
tasks.local_server_command = lambda env, reload=False: ["backend"]
original = subprocess.Popen
def child(cmd, **kwargs):
    return original([sys.executable, {str(worker)!r}, cmd[0]], **kwargs)
tasks.subprocess.Popen = child
try:
    tasks.dev(dict(os.environ))
except KeyboardInterrupt:
    raise SystemExit(130)
except subprocess.CalledProcessError as error:
    raise SystemExit(error.returncode)
''')
    with subprocess.Popen([sys.executable, str(harness)], cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as runner:
        try:
            deadline = time.monotonic() + 10
            while not all((tmp_path / (name + ".pid")).exists() for name in ("frontend", "backend")):
                assert time.monotonic() < deadline, "dev children did not start"
                time.sleep(0.02)
            if interrupt:
                runner.send_signal(signal.SIGINT)
            else:
                (tmp_path / "exit").touch()
            _, error = runner.communicate(timeout=10)
            assert runner.returncode == (130 if interrupt else 7)
            assert (tmp_path / "frontend.stopped").exists()
            if interrupt:
                assert (tmp_path / "backend.stopped").exists()
            else:
                assert b"required dev process exited (7)" in error
            for name in ("frontend", "backend"):
                with pytest.raises(ProcessLookupError):
                    os.kill(int((tmp_path / (name + ".pid")).read_text()), 0)
        finally:
            if runner.poll() is None:
                runner.terminate()
                runner.wait(timeout=10)
