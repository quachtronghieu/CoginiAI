from dataclasses import replace

import pytest

from app.config import get_settings
from app.vad import SileroVadSession


class FakeDecoder:
    def decode(self, packet: bytes) -> bytes:
        assert packet
        return b"\x00\x00" * 512

    def close(self) -> None:
        pass


class FakeCodec:
    def create_stream_decoder(self) -> FakeDecoder:
        return FakeDecoder()


class FakeIterator:
    def __init__(self, model, **kwargs) -> None:
        del model, kwargs
        self.calls = 0

    def __call__(self, tensor, return_seconds=False):
        del tensor, return_seconds
        self.calls += 1
        if self.calls == 1:
            return {"start": 0}
        return {"end": 512}


def make_vad() -> SileroVadSession:
    settings = replace(
        get_settings(),
        vad_enabled=True,
        auto_stop_timeout_seconds=30,
    )
    return SileroVadSession(object(), FakeIterator, FakeCodec(), settings)


def test_vad_detects_speech_start_and_end() -> None:
    vad = make_vad()

    assert vad.should_finalize(b"frame-1", 0.1) is False
    assert vad.speech_detected is True
    assert vad.should_finalize(b"frame-2", 0.2) is True


def test_vad_state_isolated_per_session() -> None:
    first = make_vad()
    second = make_vad()

    assert first.should_finalize(b"frame", 0.1) is False
    assert second.should_finalize(b"frame", 0.1) is False


def test_idle_timeout_does_not_cut_active_speech():
    class SpeakingIterator(FakeIterator):
        def __call__(self, tensor, return_seconds=False):
            self.calls += 1
            return {"start": 0} if self.calls == 1 else None
    settings = replace(get_settings(), rnnoise_enabled=False, auto_stop_timeout_seconds=6)
    vad = SileroVadSession(object(), SpeakingIterator, FakeCodec(), settings)
    assert vad.should_finalize(b"first", .1) is False
    assert vad.should_finalize(b"speech", 7.0) is False
    assert vad.should_finalize(b"speech", 12.0) is False


def test_idle_timeout_still_finishes_when_no_speech_detected():
    class SilentIterator(FakeIterator):
        def __call__(self, tensor, return_seconds=False):
            return None
    settings = replace(get_settings(), rnnoise_enabled=False, auto_stop_timeout_seconds=6)
    vad = SileroVadSession(object(), SilentIterator, FakeCodec(), settings)
    assert vad.should_finalize(b"silence", 5.9) is False
    assert vad.should_finalize(b"silence", 6.1) is True
    assert vad.speech_detected is False


def test_vad_evidence_records_each_model_probability_without_double_inference():
    import torch
    from silero_vad import VADIterator

    class ProbabilityModel:
        def __init__(self):
            self.values = iter((.1, .8, .9, .1))
            self.calls = 0

        def reset_states(self):
            pass

        def __call__(self, samples, sample_rate):
            del samples, sample_rate
            self.calls += 1
            return torch.tensor(next(self.values))

    model = ProbabilityModel()
    settings = replace(get_settings(), rnnoise_enabled=False, vad_threshold=.6)
    vad = SileroVadSession(model, VADIterator, FakeCodec(), settings)
    for _ in range(4):
        vad.should_finalize(b"frame", .1)

    evidence = vad.evidence
    assert model.calls == 4
    assert evidence["frame_probabilities"] == [.1, .8, .9, .1]
    assert evidence["max_probability"] == pytest.approx(.9)
    assert evidence["max_contiguous_speech_ms"] == 64.0
