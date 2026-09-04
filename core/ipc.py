from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import suppress
from typing import Any

from core.context import session_context_manager

log = logging.getLogger("resonance.core.ipc")

# Domain Invariant: Socket files must reside in runtime user directory with strict DAC (0600)
def get_default_ipc_path() -> str:
    if custom := os.environ.get("RESONANCE_IPC_PATH"):
        return custom
    if sys.platform.startswith("win"):
        return r"\\.\pipe\resonance-ipc"
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime and os.path.isdir(xdg_runtime):
        return os.path.join(xdg_runtime, "resonance.sock")
    return os.path.expanduser("~/.cache/resonance/ipc.sock")


def _dispatch_ipc_command(raw_cmd: str) -> dict[str, Any]:
    cmd = raw_cmd.strip()
    if not cmd:
        return {"error": "empty_command"}

    parts = cmd.split()
    action = parts[0].lower()

    if action == "tail":
        lines = 5
        if len(parts) > 1 and parts[1].isdigit():
            lines = max(1, min(50, int(parts[1])))

        recent = session_context_manager.get_latest_tail(lines=lines)
        return {
            "lines": recent,
            "combined": " ".join(recent),
            "count": len(recent),
        }
    elif action == "ping":
        return {"pong": True}
    else:
        return {"error": f"unknown_command: {action}"}


class UnixSocketIPCServer:
    """POSIX local IPC server backed by AF_UNIX domain socket with chmod 0600."""

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        self._server: asyncio.Server | None = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            response_payload = _dispatch_ipc_command(line.decode(errors="replace"))
            data = json.dumps(response_payload) + "\n"
            writer.write(data.encode())
            await writer.drain()
        except Exception as exc:
            log.debug(f"IPC client error: {exc}")
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def start(self) -> None:
        # Assumes: Path must be cleanly unlinked before bind to prevent EADDRINUSE
        sock_dir = os.path.dirname(self.socket_path)
        if sock_dir:
            os.makedirs(sock_dir, mode=0o700, exist_ok=True)
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        # Invariant: Socket permissions must be restricted to owner only (0600)
        old_umask = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path)
        finally:
            os.umask(old_umask)

        try:
            os.chmod(self.socket_path, 0o600)
        except OSError:
            pass

        log.info(f"UNIX domain socket IPC listening on {self.socket_path}")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
        log.info("UNIX domain socket IPC server stopped")


class WindowsNamedPipeIPCServer:
    """Windows local IPC server backed by Kernel Named Pipes (\\.\\pipe\\...)."""

    def __init__(self, pipe_name: str) -> None:
        self.pipe_name = pipe_name
        self._servers: list[Any] = []

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            response_payload = _dispatch_ipc_command(line.decode(errors="replace"))
            data = json.dumps(response_payload) + "\n"
            writer.write(data.encode())
            await writer.drain()
        except Exception as exc:
            log.debug(f"Windows IPC client error: {exc}")
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def start(self) -> None:
        loop = asyncio.get_running_loop()

        def client_connected_cb(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            asyncio.create_task(self._handle_client(reader, writer))

        def protocol_factory() -> asyncio.StreamReaderProtocol:
            reader = asyncio.StreamReader()
            return asyncio.StreamReaderProtocol(reader, client_connected_cb=client_connected_cb)

        # Assumes: Running on Windows ProactorEventLoop with IOCP named pipe support
        if hasattr(loop, "start_serving_pipe"):
            self._servers = await loop.start_serving_pipe(protocol_factory, self.pipe_name)
            log.info(f"Windows Named Pipe IPC listening on {self.pipe_name}")
        else:
            log.warning("Current event loop does not support Windows named pipes (requires ProactorEventLoop)")

    async def stop(self) -> None:
        for s in self._servers:
            with suppress(Exception):
                s.close()
        self._servers.clear()
        log.info("Windows Named Pipe IPC server stopped")


def create_local_ipc_server(custom_path: str | None = None) -> UnixSocketIPCServer | WindowsNamedPipeIPCServer:
    path = custom_path or get_default_ipc_path()
    if sys.platform.startswith("win"):
        return WindowsNamedPipeIPCServer(pipe_name=path)
    return UnixSocketIPCServer(socket_path=path)
