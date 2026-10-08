from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import suppress
from typing import Any

from core.context import SessionContextManager

log = logging.getLogger("resonance.core.ipc")


# POSIX sockets use owner-only permissions (0600); RESONANCE_IPC_PATH may override the default location.
def get_default_ipc_path() -> str:
    if custom := os.environ.get("RESONANCE_IPC_PATH"):
        return custom
    if sys.platform.startswith("win"):
        return r"\\.\pipe\resonance-ipc"
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime and os.path.isdir(xdg_runtime):
        return os.path.join(xdg_runtime, "resonance.sock")
    return os.path.expanduser("~/.cache/resonance/ipc.sock")


def _dispatch_ipc_command(
    raw_cmd: str,
    context_mgr: SessionContextManager,
) -> dict[str, Any]:
    cmd = raw_cmd.strip()
    if not cmd:
        return {"error": "empty_command"}

    parts = cmd.split()
    action = parts[0].lower()

    if action == "ping":
        if len(parts) != 1:
            return {
                "error": "invalid_command_syntax",
                "detail": f"ping does not accept arguments, got: {' '.join(parts[1:])}",
            }
        return {"pong": True}

    if action == "tail":
        if len(parts) == 1:
            lines = 5
        elif len(parts) == 2:
            if not parts[1].isascii() or not parts[1].removeprefix("-").isdecimal():
                return {
                    "error": "invalid_lines_argument",
                    "detail": f"Expected integer line count, got: {parts[1]}",
                }
            try:
                lines = int(parts[1])
            except ValueError:
                return {
                    "error": "invalid_lines_argument",
                    "detail": f"Expected integer line count, got: {parts[1]}",
                }
            if lines < 1:
                return {
                    "error": "lines_must_be_positive",
                    "detail": "Requested line count must be >= 1",
                }
        else:
            return {
                "error": "invalid_command_syntax",
                "detail": f"tail accepts at most 1 argument, got: {' '.join(parts[1:])}",
            }

        recent = context_mgr.get_ipc_tail(lines=lines)
        return {
            "lines": recent,
            "combined": " ".join(recent),
            "count": len(recent),
        }

    return {"error": f"unknown_command: {action}"}


class UnixSocketIPCServer:
    """POSIX local IPC server backed by AF_UNIX domain socket with chmod 0600."""

    def __init__(
        self,
        socket_path: str,
        context_mgr: SessionContextManager,
    ) -> None:
        self.socket_path = socket_path
        self._context_mgr = context_mgr
        self._server: asyncio.Server | None = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            response_payload = _dispatch_ipc_command(
                line.decode(errors="replace"),
                context_mgr=self._context_mgr,
            )
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

        log.debug(f"UNIX domain socket IPC listening on {self.socket_path}")

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
        log.debug("UNIX domain socket IPC server stopped")


class WindowsNamedPipeIPCServer:
    """Windows local IPC server backed by Kernel Named Pipes (\\.\\pipe\\...)."""

    def __init__(self, pipe_name: str, context_mgr: SessionContextManager) -> None:
        self.pipe_name = pipe_name
        self._context_mgr = context_mgr
        self._servers: list[Any] = []

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            response_payload = _dispatch_ipc_command(
                line.decode(errors="replace"),
                context_mgr=self._context_mgr,
            )
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
            log.debug(f"Windows Named Pipe IPC listening on {self.pipe_name}")
        else:
            log.warning("Current event loop does not support Windows named pipes (requires ProactorEventLoop)")

    async def stop(self) -> None:
        for s in self._servers:
            with suppress(Exception):
                s.close()
        self._servers.clear()
        log.debug("Windows Named Pipe IPC server stopped")


def create_local_ipc_server(
    context_mgr: SessionContextManager,
    custom_path: str | None = None,
) -> UnixSocketIPCServer | WindowsNamedPipeIPCServer:
    path = custom_path or get_default_ipc_path()
    if sys.platform.startswith("win"):
        return WindowsNamedPipeIPCServer(pipe_name=path, context_mgr=context_mgr)
    return UnixSocketIPCServer(socket_path=path, context_mgr=context_mgr)
