"""
Guard tests for the Swift ↔ Python IPC binary protocol.

IPCHeader (Swift) and _HEADER_FMT (Python) share a raw memory region.
A single-sided edit to either silently breaks the protocol at runtime.
These tests make that breakage loud and immediate.
"""
import struct

# Mirror of MacOSSharedMemoryStrategy class constants.
_HEADER_FMT    = '<4sIIIIIQIIi20x'
_HEADER_SIZE   = 64
_CMD_OFFSET    = 32
_STATUS_OFFSET = 36
_ERROR_OFFSET  = 40


def test_header_fmt_is_64_bytes():
    """Invariant: struct size must equal IPCHeader stride in Swift (64 bytes / one ARM64 cache line)."""
    assert struct.calcsize(_HEADER_FMT) == _HEADER_SIZE


def test_header_fmt_field_count():
    """Invariant: 10 unpacked fields — magic, version, sampleRate, channels,
    framesPerSlot, slotCount, writeIndex, command, status, errorCode."""
    dummy = struct.unpack(_HEADER_FMT, bytes(_HEADER_SIZE))
    assert len(dummy) == 10


def test_magic_field_is_first():
    buf = bytearray(_HEADER_SIZE)
    buf[0:4] = b'RESO'
    magic, *_ = struct.unpack(_HEADER_FMT, bytes(buf))
    assert magic == b'RESO'


def test_command_offset():
    """Invariant: command field at byte 32 — Swift writes header.command, Python reads at _CMD_OFFSET."""
    buf = bytearray(_HEADER_SIZE)
    struct.pack_into('<I', buf, _CMD_OFFSET, 0xDEAD)
    fields = struct.unpack(_HEADER_FMT, bytes(buf))
    assert fields[7] == 0xDEAD, f"command field not at offset {_CMD_OFFSET}"


def test_status_offset():
    """Invariant: status field at byte 36 — Python polls this after sending START command."""
    buf = bytearray(_HEADER_SIZE)
    struct.pack_into('<I', buf, _STATUS_OFFSET, 2)  # CAPTURING
    fields = struct.unpack(_HEADER_FMT, bytes(buf))
    assert fields[8] == 2, f"status field not at offset {_STATUS_OFFSET}"


def test_error_code_offset():
    """Invariant: errorCode field at byte 40, signed — -3801 means TCC denial (SCK error)."""
    buf = bytearray(_HEADER_SIZE)
    struct.pack_into('<i', buf, _ERROR_OFFSET, -3801)
    fields = struct.unpack(_HEADER_FMT, bytes(buf))
    assert fields[9] == -3801, f"errorCode field not at offset {_ERROR_OFFSET}"


def test_audio_slot_base_offset():
    """Invariant: audio ring buffer starts immediately after the 64-byte header.
    Python reads: offset = _HEADER_SIZE + slot_idx * bytes_per_slot.
    Swift writes: audioSlots = shmPointer.advanced(by: headerSize)."""
    frames_per_slot = 4096
    bytes_per_slot  = frames_per_slot * 4  # Float32
    slot_count      = 16

    # First slot starts at exactly HEADER_SIZE, last slot ends at HEADER_SIZE + total_data.
    first_slot_offset = _HEADER_SIZE
    last_slot_offset  = _HEADER_SIZE + (slot_count - 1) * bytes_per_slot
    total_shm_size    = _HEADER_SIZE + slot_count * bytes_per_slot

    assert first_slot_offset == 64
    assert last_slot_offset  == 64 + 15 * 4096 * 4
    assert total_shm_size    == 64 + 16 * 4096 * 4  # 262208 bytes


def test_dual_channel_interleaved_slot_layout():
    """Invariant: When channels == 2, bytes_per_slot = frames_per_slot * 2 * 4.
    Interleaving format: [sys_0, mic_0, sys_1, mic_1, ...]"""
    import numpy as np

    frames_per_slot = 4096
    channels = 2
    bytes_per_slot = frames_per_slot * channels * 4  # 32768 bytes
    slot_count = 16
    total_shm_size = _HEADER_SIZE + slot_count * bytes_per_slot  # 524352 bytes

    shm_buf = bytearray(total_shm_size)
    # Pack header with channels = 2
    struct.pack_into(_HEADER_FMT, shm_buf, 0, b'RESO', 1, 16000, 2, frames_per_slot, slot_count, 0, 0, 2, 0)

    # Fill slot 0 with synthetic interleaved data: sys = 0.5, mic = -0.5
    slot0_offset = _HEADER_SIZE
    interleaved_data = np.zeros(frames_per_slot * 2, dtype=np.float32)
    interleaved_data[0::2] = 0.5   # sys
    interleaved_data[1::2] = -0.5  # mic
    shm_buf[slot0_offset:slot0_offset + bytes_per_slot] = interleaved_data.tobytes()

    # Emulate Python unpacking
    raw = np.ndarray((frames_per_slot * 2,), dtype=np.float32, buffer=shm_buf, offset=slot0_offset)
    sys_chunk = raw[0::2]
    mic_chunk = raw[1::2]

    assert sys_chunk.shape == (4096,)
    assert mic_chunk.shape == (4096,)
    assert np.allclose(sys_chunk, 0.5)
    assert np.allclose(mic_chunk, -0.5)
