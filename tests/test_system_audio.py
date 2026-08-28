import numpy as np

from stt.system_audio import (
    LinuxPulseParecStrategy,
    NativeLinuxWindowsStrategy,
    SoundcardSystemAudioStrategy,
    WindowsWasapiStrategy,
    get_system_audio_capture,
)


def test_system_audio_factory():
    strategy = get_system_audio_capture(include_microphone=True)
    assert strategy is not None
    assert NativeLinuxWindowsStrategy is LinuxPulseParecStrategy
    assert WindowsWasapiStrategy is SoundcardSystemAudioStrategy


def test_dual_stream_mixer_logic():
    strategy = LinuxPulseParecStrategy(include_microphone=True)
    strategy.is_active = True

    # Feed system audio chunk (amplitude 0.5) and mic chunk (amplitude 0.5)
    sys_data = np.full(4096, 0.5, dtype=np.float32)
    mic_data = np.full(4096, 0.5, dtype=np.float32)

    strategy.queue.put(sys_data)
    strategy.mic_queue.put(mic_data)

    gen = strategy.get_audio_stream()

    source1, chunk1 = next(gen)
    assert source1 == "sys"
    assert np.allclose(chunk1, 0.5, atol=1e-4)
    assert chunk1.shape == (4096,)

    source2, chunk2 = next(gen)
    assert source2 == "mic"
    assert np.allclose(chunk2, 0.5, atol=1e-4)
    assert chunk2.shape == (4096,)

    strategy.stop_capture()
    assert strategy.is_active is False


def test_windows_wasapi_strategy_lifecycle():
    strat = WindowsWasapiStrategy(include_microphone=True)
    assert strat.queue is not None
    assert strat.mic_queue is not None

    # Test draining
    strat.is_active = True
    strat.queue.put(np.full(1024, 0.3, dtype=np.float32))
    strat.mic_queue.put(np.full(1024, 0.7, dtype=np.float32))

    gen = strat.get_audio_stream()
    src1, c1 = next(gen)
    assert src1 == "sys"
    assert np.allclose(c1, 0.3, atol=1e-4)

    src2, c2 = next(gen)
    assert src2 == "mic"
    assert np.allclose(c2, 0.7, atol=1e-4)

    strat.stop_capture()
    assert strat.is_active is False
