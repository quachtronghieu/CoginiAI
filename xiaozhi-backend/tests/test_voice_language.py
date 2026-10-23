import asyncio

import pytest

from app.ai import AsrResult, EmptyUtteranceError, LocalAiPipeline, NonSpeechUtteranceError, PcmAudio
from app.allergy import CustomerProfile
from app.diagnostics import VoiceDiagnostics


class Codec:
    async def decode_opus(self, frames):
        return PcmAudio(b"\x00\x00" * 16000, 16000)

    async def encode_opus(self, audio):
        return [b"audio"]


class Asr:
    def __init__(self, text):
        self.text = text

    async def transcribe(self, audio):
        return AsrResult(self.text, "en", .84, -.5)


class Llm:
    async def generate(self, transcript, language, probability, confidence, **kwargs):
        assert language == "vi"
        return "Thông tin món ăn tại Cà Mau."


class Tts:
    async def synthesize(self, answer, language, probability):
        assert language == "vi"
        return PcmAudio(b"\x00\x00" * 160, 16000)


def test_short_vietnamese_location_keeps_vietnamese_tts():
    profile = CustomerProfile(allergy_status="none", response_language="vi")
    result = asyncio.run(LocalAiPipeline(Codec(), Asr("Cà Mau"), Llm(), Tts()).run([b"audio"], customer_profile=profile))
    assert result.language == "vi" and result.audio_frames


def test_silence_is_saved_with_rejection_reason(tmp_path):
    diagnostics = VoiceDiagnostics(tmp_path)
    pipeline = LocalAiPipeline(Codec(), Asr(""), Llm(), Tts(), diagnostics=diagnostics)
    with pytest.raises(EmptyUtteranceError):
        asyncio.run(pipeline.run([b"audio"]))
    assert (tmp_path / "sample-00-raw.wav").exists()
    assert not (tmp_path / "sample-00-rnnoise.wav").exists()
    import json
    metadata = json.loads((tmp_path / "sample-00.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "asr_rejected"
    assert metadata["rejection_reason"] == "empty_transcript"
    assert metadata["audio"]["rnnoise"] is None


def test_raw_mode_skips_noise_and_calls_asr_once(tmp_path):
    class ForbiddenNoise:
        async def process(self, audio):
            pytest.fail("Raw conversation entered RNNoise")

    class TrackingAsr:
        def __init__(self):
            self.inputs = []

        async def transcribe(self, audio, **kwargs):
            self.inputs.append((audio.data, kwargs.get("raw_audio")))
            return AsrResult("Hải sản", "vi", .99, -.2)

    asr = TrackingAsr()
    pipeline = LocalAiPipeline(
        Codec(), asr, Llm(), Tts(),
        noise_suppressor=ForbiddenNoise(), diagnostics=VoiceDiagnostics(tmp_path),
    )
    result = asyncio.run(pipeline.run(
        [b"audio"], customer_profile=CustomerProfile(response_language="vi"),
    ))

    assert result.transcript == "Hải sản"
    assert asr.inputs == [(b"\x00\x00" * 16000, None)]
    assert not (tmp_path / "sample-00-rnnoise.wav").exists()
    import json
    metadata = json.loads((tmp_path / "sample-00.json").read_text(encoding="utf-8"))
    assert metadata["rnnoise_asr_mode"] == "raw"
    assert metadata["asr"]["primary_source"] == "raw"
    assert metadata["asr"]["raw"]["text"] == "Hải sản"
    assert metadata["asr"]["rnnoise"] is None
    assert set(metadata["timings_ms"]) >= {"asr", "dialogue_rag_llm", "tts", "encode", "total"}
    assert metadata["timings_ms"]["total"] >= metadata["timings_ms"]["asr"]


@pytest.mark.parametrize("mode", ["shadow", "filtered"])
def test_missing_noise_suppressor_uses_single_raw_asr(mode):
    class TrackingAsr:
        def __init__(self):
            self.calls = 0

        async def transcribe(self, audio, **kwargs):
            self.calls += 1
            return AsrResult("Hải sản", "vi", .99, -.2)

    asr = TrackingAsr()
    result = asyncio.run(LocalAiPipeline(
        Codec(), asr, Llm(), Tts(), rnnoise_asr_mode=mode,
    ).run([b"audio"]))
    assert result.transcript == "Hải sản"
    assert asr.calls == 1


def test_shadow_mode_uses_raw_for_dialogue_and_records_both_asr_results(tmp_path):
    class Noise:
        async def process(self, audio):
            return PcmAudio(b"\x01\x00" * 16000, 16000)

    class SourceAwareAsr:
        def __init__(self):
            self.calls = 0

        async def transcribe(self, audio, **kwargs):
            self.calls += 1
            text = "Xin chào" if audio.data.startswith(b"\x00\x00") else "Yeah"
            return AsrResult(text, "vi" if text == "Xin chào" else "en", .9, -.2)

    class GreetingLlm:
        async def generate(self, transcript, language, probability, confidence, **kwargs):
            assert transcript == "Xin chào"
            assert language == "vi"
            return "Xin chào."

    diagnostics = VoiceDiagnostics(tmp_path)
    asr = SourceAwareAsr()
    result = asyncio.run(LocalAiPipeline(
        Codec(), asr, GreetingLlm(), Tts(),
        noise_suppressor=Noise(), diagnostics=diagnostics,
        rnnoise_asr_mode="shadow",
    ).run([b"audio"], customer_profile=CustomerProfile()))

    assert result.transcript == "Xin chào"
    assert asr.calls == 2
    assert (tmp_path / "sample-00-rnnoise.wav").exists()
    import json
    metadata = json.loads((tmp_path / "sample-00.json").read_text(encoding="utf-8"))
    assert metadata["asr"]["primary_source"] == "raw"
    assert metadata["asr"]["raw"]["text"] == "Xin chào"
    assert metadata["asr"]["rnnoise"]["text"] == "Yeah"


def test_explicit_filtered_mode_preserves_raw_safety_comparison(tmp_path):
    filtered = PcmAudio(b"\x01\x00" * 16000, 16000)

    class Noise:
        async def process(self, audio):
            return filtered

    class FilteredAsr:
        async def transcribe(self, audio, raw_audio=None, **kwargs):
            assert audio is filtered
            assert raw_audio.data == b"\x00\x00" * 16000
            return AsrResult("Hải sản", "vi", .99, -.2)

    result = asyncio.run(LocalAiPipeline(
        Codec(), FilteredAsr(), Llm(), Tts(), noise_suppressor=Noise(),
        diagnostics=VoiceDiagnostics(tmp_path), rnnoise_asr_mode="filtered",
    ).run([b"audio"]))

    assert result.transcript == "Hải sản"
    import json
    metadata = json.loads((tmp_path / "sample-00.json").read_text(encoding="utf-8"))
    assert metadata["asr"]["primary_source"] == "rnnoise"
    assert metadata["asr"]["rnnoise"]["text"] == "Hải sản"
    assert (tmp_path / "sample-00-rnnoise.wav").exists()


def test_vad_rejected_audio_is_saved_without_calling_asr_or_llm(tmp_path):
    class ForbiddenNoise:
        async def process(self, audio):
            pytest.fail("Raw silence diagnostic entered RNNoise")

    class ForbiddenAsr:
        async def transcribe(self, *args, **kwargs):
            pytest.fail("VAD-rejected audio reached Whisper")

    class ForbiddenLlm:
        async def generate(self, *args, **kwargs):
            pytest.fail("VAD-rejected audio reached the LLM")

    diagnostics = VoiceDiagnostics(tmp_path)
    pipeline = LocalAiPipeline(
        Codec(), ForbiddenAsr(), ForbiddenLlm(), Tts(),
        noise_suppressor=ForbiddenNoise(), diagnostics=diagnostics,
    )

    asyncio.run(pipeline.record_rejected_utterance([b"noise"], "vad_no_speech"))

    import json
    metadata = json.loads((tmp_path / "sample-00.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "vad_rejected"
    assert metadata["rejection_reason"] == "vad_no_speech"
    assert metadata["asr"]["raw"] is None
    assert (tmp_path / "sample-00-raw.wav").exists()
    assert metadata["audio"]["rnnoise"] is None


def test_cancelled_shadow_write_keeps_vad_rejection_and_raw(tmp_path):
    class BlockingNoise:
        def __init__(self):
            self.started = asyncio.Event()

        async def process(self, audio):
            del audio
            self.started.set()
            await asyncio.Future()

    async def exercise():
        noise = BlockingNoise()
        diagnostics = VoiceDiagnostics(tmp_path)
        pipeline = LocalAiPipeline(
            Codec(), Asr("unused"), Llm(), Tts(),
            noise_suppressor=noise, diagnostics=diagnostics,
            rnnoise_asr_mode="shadow",
        )
        task = asyncio.create_task(
            pipeline.record_rejected_utterance([b"noise"], "vad_no_speech")
        )
        await noise.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    import json
    metadata = json.loads((tmp_path / "sample-00.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "vad_rejected"
    assert metadata["rejection_reason"] == "vad_no_speech"
    assert (tmp_path / "sample-00-raw.wav").exists()


def test_combined_gate_rejects_weak_vad_and_high_no_speech_asr(tmp_path):
    class UncertainAsr:
        async def transcribe(self, audio):
            del audio
            return AsrResult("You", "en", .99, -.92, no_speech_probability=.726)

    class ForbiddenLlm:
        async def generate(self, *args, **kwargs):
            pytest.fail("Uncertain environmental audio reached the LLM")

    diagnostics = VoiceDiagnostics(tmp_path)
    pipeline = LocalAiPipeline(Codec(), UncertainAsr(), ForbiddenLlm(), Tts(), diagnostics=diagnostics)
    evidence = {"available": True, "max_contiguous_speech_ms": 160.0}

    with pytest.raises(NonSpeechUtteranceError):
        asyncio.run(pipeline.run([b"audio"], customer_profile=CustomerProfile(), vad_evidence=evidence))

    import json
    metadata = json.loads((tmp_path / "sample-00.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "asr_vad_rejected"
    assert metadata["asr"]["raw"]["no_speech_probability"] == .726
    assert metadata["asr"]["rnnoise"] is None


def test_clear_short_speech_survives_when_only_vad_is_weak():
    class ClearAsr:
        async def transcribe(self, audio):
            del audio
            return AsrResult("Hải sản", "vi", .99, -.2, no_speech_probability=.01)

    result = asyncio.run(LocalAiPipeline(Codec(), ClearAsr(), Llm(), Tts()).run(
        [b"audio"],
        customer_profile=CustomerProfile(response_language="vi"),
        vad_evidence={"available": True, "max_contiguous_speech_ms": 96.0},
    ))

    assert result.transcript == "Hải sản"


def test_asr_disagreement_does_not_commit_or_call_llm():
    profile = CustomerProfile(pending_allergy_status="declared", pending_allergens=["sữa"], response_language="vi")
    class DisagreeingAsr:
        async def transcribe(self, audio, **kwargs):
            assert kwargs["customer_profile"] is profile
            return AsrResult("Đúng rồi", "vi", .99, -.2, True, ("Đúng rồi", "Không đúng"), "confirmation_disagreement")
    class ForbiddenLlm:
        async def generate(self, *args, **kwargs):
            pytest.fail("Ambiguous confirmation reached LLM/profile mutation")
    result = asyncio.run(LocalAiPipeline(Codec(), DisagreeingAsr(), ForbiddenLlm(), Tts()).run([b"audio"], customer_profile=profile, on_transcript=lambda text: pytest.fail("Ambiguous speech changed persona/profile")))
    assert "chưa nghe rõ" in result.answer and "chính xác" in result.answer
    assert profile.allergy_status == "unknown" and profile.pending_allergens == ["sữa"]
    assert profile.last_route == "asr_clarification"
    assert profile.dialogue_action == "asr_clarification"
