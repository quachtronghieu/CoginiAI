"""Local Opus codec adapter using PyAV/FFmpeg libopus."""

from __future__ import annotations

import asyncio
from fractions import Fraction

from .ai import AudioCodecError, PcmAudio
from .config import Settings


class PyAvOpusCodec:
    """Decode ESP32 Opus packets and encode 24 kHz mono response packets.

    Decoder/encoder contexts are created per utterance. This keeps codec state
    isolated between devices while avoiding a process per packet.
    """

    input_sample_rate = 16_000
    output_sample_rate = 24_000
    channels = 1
    frame_duration_ms = 60

    def __init__(self, settings: Settings | None = None) -> None:
        self.input_sample_rate = settings.xiaozhi_input_sample_rate if settings else 16_000
        self.output_sample_rate = settings.server_sample_rate if settings else 24_000
        self.frame_duration_ms = settings.frame_duration_ms if settings else 60
        self._av = None
        self._np = None
        try:
            import av
            import numpy as np

            self._av = av
            self._np = np
        except ImportError:
            pass

    def available(self) -> bool:
        return self._av is not None and self._np is not None

    async def decode_opus(self, frames: list[bytes]) -> PcmAudio:
        return await asyncio.to_thread(self._decode_sync, frames)

    async def encode_opus(self, audio: PcmAudio) -> list[bytes]:
        return await asyncio.to_thread(self._encode_sync, audio)

    def decode_packet(self, packet: bytes) -> bytes:
        """Decode one packet for streaming VAD use."""
        if not self.available():
            raise AudioCodecError("PyAV and NumPy are required for Opus decoding")
        try:
            return self._decode_packets([packet], flush=False)
        except Exception as exc:
            raise AudioCodecError("Malformed Opus audio frame") from exc

    def create_stream_decoder(self) -> "PyAvOpusStreamDecoder":
        return PyAvOpusStreamDecoder(self)

    def _decode_sync(self, frames: list[bytes]) -> PcmAudio:
        if not frames:
            raise AudioCodecError("Cannot decode an empty Opus utterance")
        try:
            pcm = self._decode_packets(frames, flush=True)
        except AudioCodecError:
            raise
        except Exception as exc:
            raise AudioCodecError("Malformed Opus audio frame") from exc
        if not pcm:
            raise AudioCodecError("Opus decoder produced no PCM audio")
        return PcmAudio(data=pcm, sample_rate=self.input_sample_rate, channels=1)

    def _decode_packets(self, frames: list[bytes], flush: bool) -> bytes:
        av = self._require_av()
        np = self._require_numpy()
        decoder = av.CodecContext.create("opus", "r")
        decoder.sample_rate = self.input_sample_rate
        decoder.open()
        resampler = av.AudioResampler(
            format="s16", layout="mono", rate=self.input_sample_rate
        )
        output: list[bytes] = []
        for raw in frames:
            if not raw:
                raise AudioCodecError("Empty Opus audio frame")
            decoded = decoder.decode(av.Packet(raw))
            for audio_frame in decoded:
                output.extend(self._resample_bytes(resampler, audio_frame, np))
        if flush:
            for audio_frame in decoder.decode(None):
                output.extend(self._resample_bytes(resampler, audio_frame, np))
            output.extend(self._flush_resampler(resampler, np))
        return b"".join(output)

    @staticmethod
    def _resample_bytes(resampler, frame, np) -> list[bytes]:
        result = resampler.resample(frame)
        if result is None:
            return []
        frames = result if isinstance(result, list) else [result]
        return [np.asarray(item.to_ndarray(), dtype=np.int16).tobytes() for item in frames]

    @staticmethod
    def _flush_resampler(resampler, np) -> list[bytes]:
        result = resampler.resample(None)
        if result is None:
            return []
        frames = result if isinstance(result, list) else [result]
        return [np.asarray(item.to_ndarray(), dtype=np.int16).tobytes() for item in frames]

    def _encode_sync(self, audio: PcmAudio) -> list[bytes]:
        if not audio.data:
            raise AudioCodecError("Cannot encode empty PCM audio")
        if audio.channels != 1 or len(audio.data) % 2:
            raise AudioCodecError("PCM audio must be signed 16-bit mono")
        av = self._require_av()
        np = self._require_numpy()
        try:
            samples = np.frombuffer(audio.data, dtype="<i2")
            input_frame = av.AudioFrame.from_ndarray(
                samples.reshape(1, -1), format="s16", layout="mono"
            )
            input_frame.sample_rate = audio.sample_rate
            input_frame.time_base = Fraction(1, audio.sample_rate)
            resampler = av.AudioResampler(
                format="s16", layout="mono", rate=self.output_sample_rate
            )
            pcm_frames = self._resample_bytes(resampler, input_frame, np)
            pcm_frames.extend(self._flush_resampler(resampler, np))
            pcm = b"".join(pcm_frames)
            samples_per_packet = self.output_sample_rate * self.frame_duration_ms // 1000
            bytes_per_packet = samples_per_packet * 2
            encoder = av.CodecContext.create("libopus", "w")
            encoder.sample_rate = self.output_sample_rate
            encoder.layout = "mono"
            encoder.format = "s16"
            encoder.bit_rate = 24_000
            encoder.time_base = Fraction(1, self.output_sample_rate)
            encoder.options = {
                "frame_duration": str(self.frame_duration_ms),
                "application": "voip",
            }
            encoder.open()
            packets: list[bytes] = []
            pts = 0
            for offset in range(0, len(pcm), bytes_per_packet):
                chunk = pcm[offset : offset + bytes_per_packet]
                if len(chunk) < bytes_per_packet:
                    chunk += b"\x00" * (bytes_per_packet - len(chunk))
                frame = av.AudioFrame.from_ndarray(
                    np.frombuffer(chunk, dtype="<i2").reshape(1, samples_per_packet),
                    format="s16",
                    layout="mono",
                )
                frame.sample_rate = self.output_sample_rate
                frame.time_base = Fraction(1, self.output_sample_rate)
                frame.pts = pts
                pts += samples_per_packet
                packets.extend(bytes(packet) for packet in encoder.encode(frame))
            if not packets:
                raise AudioCodecError("Opus encoder produced no packets")
            return packets
        except AudioCodecError:
            raise
        except Exception as exc:
            raise AudioCodecError("PCM to Opus encoding failed") from exc

    def _require_av(self):
        if self._av is None:
            raise AudioCodecError("PyAV is required for Opus codec")
        return self._av

    def _require_numpy(self):
        if self._np is None:
            raise AudioCodecError("NumPy is required for Opus codec")
        return self._np


class PyAvOpusStreamDecoder:
    """Stateful decoder isolated to one WebSocket/VAD session."""

    def __init__(self, codec: PyAvOpusCodec) -> None:
        self.codec = codec
        av = codec._require_av()
        self.np = codec._require_numpy()
        self.decoder = av.CodecContext.create("opus", "r")
        self.decoder.sample_rate = codec.input_sample_rate
        self.decoder.open()
        self.resampler = av.AudioResampler(
            format="s16", layout="mono", rate=codec.input_sample_rate
        )
        self.closed = False

    def decode(self, packet: bytes) -> bytes:
        if self.closed:
            raise AudioCodecError("Opus stream decoder is closed")
        if not packet:
            raise AudioCodecError("Empty Opus audio frame")
        try:
            output: list[bytes] = []
            for frame in self.decoder.decode(self.codec._av.Packet(packet)):
                output.extend(self.codec._resample_bytes(self.resampler, frame, self.np))
            return b"".join(output)
        except Exception as exc:
            raise AudioCodecError("Malformed Opus audio frame") from exc

    def close(self) -> None:
        self.closed = True
        self.decoder = None
        self.resampler = None
