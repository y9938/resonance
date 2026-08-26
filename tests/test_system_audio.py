import numpy as np

from stt.system_audio import NativeLinuxWindowsStrategy, get_system_audio_capture


def test_system_audio_factory():
    strategy = get_system_audio_capture(include_microphone=True)
    assert strategy is not None


def test_dual_stream_mixer_logic(monkeypatch):
    class FakeInputStream:
        def __init__(self, *args, **kwargs):
            self.callback = kwargs.get("callback")

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    class FakeSoundDevice:
        def query_devices(self):
            return [{"name": "pulse", "max_input_channels": 2}]

        def InputStream(self, *args, **kwargs):
            return FakeInputStream(*args, **kwargs)

    monkeypatch.setattr("sounddevice.query_devices", lambda: [{"name": "pulse", "max_input_channels": 2}])

    strategy = NativeLinuxWindowsStrategy(include_microphone=True)
    strategy.sd = FakeSoundDevice()
    strategy.start_capture()

    # Feed system audio chunk (amplitude 0.5) and mic chunk (amplitude 0.5)
    sys_data = np.full(4096, 0.5, dtype=np.float32)
    mic_data = np.full(4096, 0.5, dtype=np.float32)

    strategy.queue.put(sys_data)
    strategy.mic_queue.put(mic_data)

    gen = strategy.get_audio_stream()
    mixed = next(gen)

    # 0.5 * 0.7 + 0.5 * 0.7 = 0.70
    assert np.allclose(mixed, 0.7, atol=1e-4)
    assert mixed.shape == (4096,)

    strategy.stop_capture()


def test_windows_wasapi_strategy_lifecycle():
    from stt.system_audio import WindowsWasapiStrategy

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
