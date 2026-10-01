import asyncio
import math
from fractions import Fraction

import av
import numpy as np
import pytest

from app.ai import AudioCodecError, PcmAudio
from app.codec import PyAvOpusCodec


def encode_esp32_packet(samples: np.ndarray) -> bytes:
    encoder = av.CodecContext.create("libopus", "w")
    encoder.sample_rate = 16_000
    encoder.layout = "mono"
    encoder.format = "s16"
    encoder.time_base = Fraction(1, 16_000)
    encoder.options = {"frame_duration": "60", "application": "voip"}
    encoder.open()
    frame = av.AudioFrame.from_ndarray(
        samples.reshape(1, -1), format="s16", layout="mono"
    )
    frame.sample_rate = 16_000
    frame.time_base = Fraction(1, 16_000)
    return bytes(encoder.encode(frame)[0])


def test_opus_encode_and_decode_real_frames() -> None:
    codec = PyAvOpusCodec()
    sample_count = 2_400
    time = np.arange(sample_count) / 24_000
    samples = (np.sin(2 * math.pi * 440 * time) * 8_000).astype("<i2")

    packets = asyncio.run(
        codec.encode_opus(PcmAudio(samples.tobytes(), sample_rate=24_000))
    )
    decoded = asyncio.run(codec.decode_opus(packets))

    assert len(packets) == 2
    assert all(packets)
    assert decoded.sample_rate == 16_000
    assert decoded.channels == 1
    assert len(decoded.data) >= 3_100


def test_invalid_opus_frame_raises_codec_error() -> None:
    codec = PyAvOpusCodec()

    with pytest.raises(AudioCodecError, match="Malformed Opus"):
        asyncio.run(codec.decode_opus([b"not-an-opus-packet"]))


def test_decode_esp32_16khz_60ms_packet() -> None:
    codec = PyAvOpusCodec()
    packet = encode_esp32_packet(np.zeros(960, dtype=np.int16))

    decoded = asyncio.run(codec.decode_opus([packet]))

    assert decoded.sample_rate == 16_000
    assert len(decoded.data) == 960 * 2


def test_pcm_encode_rejects_invalid_sample_width() -> None:
    codec = PyAvOpusCodec()

    with pytest.raises(AudioCodecError, match="signed 16-bit mono"):
        asyncio.run(codec.encode_opus(PcmAudio(b"\x00", sample_rate=24_000)))
