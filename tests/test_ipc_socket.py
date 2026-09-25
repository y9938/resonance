import asyncio
import json
import os
import sys

import pytest

from core.context import SessionContextManager
from core.ipc import UnixSocketIPCServer, _dispatch_ipc_command


def test_ipc_tail_contract() -> None:
    manager = SessionContextManager()
    for index in range(7):
        manager.append_live("session", f"line {index}")

    assert _dispatch_ipc_command("tail", manager)["lines"] == [
        f"line {index}" for index in range(2, 7)
    ]
    response = _dispatch_ipc_command("tail 3", manager)
    assert response == {
        "lines": ["line 4", "line 5", "line 6"],
        "combined": "line 4 line 5 line 6",
        "count": 3,
    }
    assert _dispatch_ipc_command("tail 10", manager)["lines"] == [
        f"line {index}" for index in range(7)
    ]
    assert _dispatch_ipc_command("tail 0", manager)["error"] == "lines_must_be_positive"
    assert _dispatch_ipc_command("tail -1", manager)["error"] == "lines_must_be_positive"
    for argument in ("invalid", "1_0", "+1", "1.0", "0x10", "１２"):
        assert _dispatch_ipc_command(f"tail {argument}", manager)["error"] == "invalid_lines_argument"
    assert _dispatch_ipc_command("tail 2 extra", manager)["error"] == "invalid_command_syntax"


def test_ipc_ping_contract() -> None:
    manager = SessionContextManager()
    assert _dispatch_ipc_command("ping", manager) == {"pong": True}
    assert _dispatch_ipc_command("ping extra", manager)["error"] == "invalid_command_syntax"


@pytest.mark.asyncio
async def test_unix_socket_permissions_and_tail(tmp_path):
    if sys.platform.startswith("win"):
        pytest.skip("Unix domain socket tests run on POSIX")

    socket_path = str(tmp_path / "test_res.sock")
    manager = SessionContextManager()
    manager.append_live("session", "Kernel socket check")
    ipc_server = UnixSocketIPCServer(socket_path=socket_path, context_mgr=manager)

    await ipc_server.start()
    try:
        assert os.stat(socket_path).st_mode & 0o777 == 0o600
        reader, writer = await asyncio.open_unix_connection(socket_path)
        writer.write(b"tail 1\n")
        await writer.drain()

        response = json.loads((await reader.readline()).decode())
        assert response["lines"] == ["Kernel socket check"]

        writer.close()
        await writer.wait_closed()
    finally:
        await ipc_server.stop()
    assert not os.path.exists(socket_path)


@pytest.mark.asyncio
async def test_windows_named_pipe_tail():
    if not sys.platform.startswith("win"):
        pytest.skip("Windows named pipe tests run on Windows")

    from core.ipc import WindowsNamedPipeIPCServer

    pipe_name = r"\\.\pipe\test-resonance-ipc-unit"
    manager = SessionContextManager()
    manager.append_live("session", "Pipe payload check")
    ipc_server = WindowsNamedPipeIPCServer(pipe_name=pipe_name, context_mgr=manager)

    await ipc_server.start()
    try:
        loop = asyncio.get_running_loop()
        client_reader = asyncio.StreamReader()
        client_proto = asyncio.StreamReaderProtocol(client_reader)
        transport, _ = await loop.create_pipe_connection(lambda: client_proto, pipe_name)
        writer = asyncio.StreamWriter(transport, client_proto, client_reader, loop)

        writer.write(b"tail 1\n")
        await writer.drain()
        response = json.loads((await client_reader.readline()).decode())
        assert response["lines"] == ["Pipe payload check"]

        writer.close()
        await writer.wait_closed()
    finally:
        await ipc_server.stop()
