import asyncio
from dataclasses import replace

import numpy as np
import pytest

from app.ai import PcmAudio
from app.config import get_settings
from app.noise import RnnoiseNoiseSuppressor, RnnoiseStreamFilter


def test_rnnoise_filters_16khz_pcm_and_preserves_format() -> None:
    settings = replace(get_settings(), rnnoise_enabled=True)
    service = RnnoiseNoiseSuppressor(settings)
    rate = 16_000
    time = np.arange(rate // 2, dtype=np.float64) / rate
    rng = np.random.default_rng(7)
    samples = np.clip(
        5_000 * np.sin(2 * np.pi * 220 * time) + rng.normal(0, 1_500, time.size),
        -32_768,
        32_767,
    ).astype("<i2")
    source = PcmAudio(samples.tobytes(), rate, 1)

    result = asyncio.run(service.process(source))

    assert service.available() is True
    assert result.sample_rate == rate
    assert result.channels == 1
    assert len(result.data) == len(source.data)
    assert result.data != source.data


def test_installed_rnnoise_runtime_matches_native_contract() -> None:
    from pyrnnoise.rnnoise import FRAME_SIZE, SAMPLE_RATE

    assert SAMPLE_RATE == 48_000
    assert FRAME_SIZE == 480


def test_disabled_rnnoise_is_passthrough() -> None:
    service = RnnoiseNoiseSuppressor(
        replace(get_settings(), rnnoise_enabled=False)
    )
    source = PcmAudio(b"\x00\x00" * 160, 16_000, 1)

    result = asyncio.run(service.process(source))

    assert result is source
    assert service.available() is False


def test_rnnoise_stream_filter_produces_pcm_for_vad() -> None:
    stream_filter = RnnoiseStreamFilter(16_000)
    source = (np.ones(960, dtype="<i2") * 1_000).tobytes()

    result = stream_filter.process(source)
    stream_filter.close()

    assert result
    assert len(result) % 2 == 0


def test_rnnoise_silence_stays_silent_and_length_stable():
    source = PcmAudio(b"\x00\x00" * 16000, 16000)
    result = asyncio.run(RnnoiseNoiseSuppressor(replace(get_settings(), rnnoise_enabled=True)).process(source))
    assert len(result.data) == len(source.data)
    assert np.max(np.abs(np.frombuffer(result.data, dtype='<i2').astype(np.int32))) <= 1


@pytest.mark.parametrize("source", [PcmAudio(b"\x00", 16000), PcmAudio(b"\x00\x00", 16000, 2), PcmAudio(b"", 16000)])
def test_invalid_noise_input_fails_explicitly(source):
    from app.ai import AiPipelineError
    service = RnnoiseNoiseSuppressor(replace(get_settings(), rnnoise_enabled=True))
    with pytest.raises(AiPipelineError):
        asyncio.run(service.process(source))
