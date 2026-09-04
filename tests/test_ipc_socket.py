import asyncio
import json
import os
import sys

import pytest

from core.context import session_context_manager
from core.ipc import (
    UnixSocketIPCServer,
    _dispatch_ipc_command,
)


def test_ipc_command_dispatch():
    session_context_manager.clear("test_session")
    session_context_manager.append("test_session", "Context line 1")
    session_context_manager.append("test_session", "Context line 2")

    res_ping = _dispatch_ipc_command("ping")
    assert res_ping == {"pong": True}

    res_tail = _dispatch_ipc_command("tail 2")
    assert res_tail["count"] == 2
    assert res_tail["lines"] == ["Context line 1", "Context line 2"]
    assert "Context line 1 Context line 2" in res_tail["combined"]

    res_unknown = _dispatch_ipc_command("unknown_action")
    assert "error" in res_unknown


@pytest.mark.asyncio
async def test_unix_socket_ipc_server(tmp_path):
    if sys.platform.startswith("win"):
        pytest.skip("Unix domain socket tests run on POSIX")

    sock_path = str(tmp_path / "test_res.sock")
    server = UnixSocketIPCServer(socket_path=sock_path)

    session_context_manager.append("sock_session", "Kernel socket check")

    await server.start()
    try:
        assert os.path.exists(sock_path)
        # Check permissions: strictly owner-only (0600)
        mode = os.stat(sock_path).st_mode & 0o777
        assert mode == 0o600

        reader, writer = await asyncio.open_unix_connection(sock_path)
        writer.write(b"tail 1\n")
        await writer.drain()

        raw = await reader.readline()
        data = json.loads(raw.decode())
        assert "lines" in data
        assert data["lines"] == ["Kernel socket check"]

        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()
        assert not os.path.exists(sock_path)


@pytest.mark.asyncio
async def test_windows_named_pipe_ipc_server():
    if not sys.platform.startswith("win"):
        pytest.skip("Windows named pipe tests run on Windows")

    from core.ipc import WindowsNamedPipeIPCServer

    pipe_name = r"\\.\pipe\test-resonance-ipc-unit"
    server = WindowsNamedPipeIPCServer(pipe_name=pipe_name)

    session_context_manager.append("win_pipe_session", "Pipe payload check")

    await server.start()
    try:
        loop = asyncio.get_running_loop()
        client_reader = asyncio.StreamReader()
        client_proto = asyncio.StreamReaderProtocol(client_reader)
        transport, _ = await loop.create_pipe_connection(lambda: client_proto, pipe_name)
        writer = asyncio.StreamWriter(transport, client_proto, client_reader, loop)

        writer.write(b"tail 1\n")
        await writer.drain()

        raw = await client_reader.readline()
        data = json.loads(raw.decode())
        assert "lines" in data
        assert data["lines"] == ["Pipe payload check"]

        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()
