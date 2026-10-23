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
    assert (tmp_path / "sample-00-rnnoise.wav").exists()
    import json
    metadata = json.loads((tmp_path / "sample-00.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "asr_rejected"
    assert metadata["rejection_reason"] == "empty_transcript"


def test_shadow_mode_uses_raw_for_dialogue_and_records_both_asr_results(tmp_path):
    class Noise:
        async def process(self, audio):
            return PcmAudio(b"\x01\x00" * 16000, 16000)

    class SourceAwareAsr:
        async def transcribe(self, audio, **kwargs):
            text = "Xin chào" if audio.data.startswith(b"\x00\x00") else "Yeah"
            return AsrResult(text, "vi" if text == "Xin chào" else "en", .9, -.2)

    class GreetingLlm:
        async def generate(self, transcript, language, probability, confidence, **kwargs):
            assert transcript == "Xin chào"
            assert language == "vi"
            return "Xin chào."

    diagnostics = VoiceDiagnostics(tmp_path)
    result = asyncio.run(LocalAiPipeline(
        Codec(), SourceAwareAsr(), GreetingLlm(), Tts(),
        noise_suppressor=Noise(), diagnostics=diagnostics,
        rnnoise_asr_mode="shadow",
    ).run([b"audio"], customer_profile=CustomerProfile()))

    assert result.transcript == "Xin chào"
    import json
    metadata = json.loads((tmp_path / "sample-00.json").read_text(encoding="utf-8"))
    assert metadata["asr"]["primary_source"] == "raw"
    assert metadata["asr"]["raw"]["text"] == "Xin chào"
    assert metadata["asr"]["rnnoise"]["text"] == "Yeah"


def test_vad_rejected_audio_is_saved_without_calling_asr_or_llm(tmp_path):
    class ForbiddenAsr:
        async def transcribe(self, *args, **kwargs):
            pytest.fail("VAD-rejected audio reached Whisper")

    class ForbiddenLlm:
        async def generate(self, *args, **kwargs):
            pytest.fail("VAD-rejected audio reached the LLM")

    diagnostics = VoiceDiagnostics(tmp_path)
    pipeline = LocalAiPipeline(Codec(), ForbiddenAsr(), ForbiddenLlm(), Tts(), diagnostics=diagnostics)

    asyncio.run(pipeline.record_rejected_utterance([b"noise"], "vad_no_speech"))

    import json
    metadata = json.loads((tmp_path / "sample-00.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "vad_rejected"
    assert metadata["rejection_reason"] == "vad_no_speech"
    assert metadata["asr"]["raw"] is None
    assert (tmp_path / "sample-00-raw.wav").exists()


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
    assert metadata["asr"]["raw"] is None
    assert metadata["asr"]["rnnoise"]["no_speech_probability"] == .726


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

def test_fixed_vietnamese_asr_passes_language_and_hotwords():
    from dataclasses import replace
    from app.config import get_settings
    from app.asr import FasterWhisperAsrService

    settings = replace(get_settings(), asr_model="dummy-model", asr_language="vi")
    service = FasterWhisperAsrService(settings)

    class MockModel:
        def __init__(self):
            self.last_kwargs = {}

        def transcribe(self, samples, **kwargs):
            self.last_kwargs = kwargs
            segment = type("Segment", (), {"text": "bún chả Hà Nội", "avg_logprob": -0.1, "no_speech_prob": 0.01})()
            info = type("Info", (), {"language": "vi", "language_probability": 0.99})()
            return iter([segment]), info

    mock_model = MockModel()
    service._model = mock_model
    result = asyncio.run(service.transcribe(PcmAudio(b"\x00\x08" * 3200, 16000)))

    assert result.text == "bún chả Hà Nội"
    assert result.language == "vi"
    assert mock_model.last_kwargs["language"] == "vi"
    assert "initial_prompt" not in mock_model.last_kwargs
    assert "phở" in mock_model.last_kwargs["hotwords"]
    assert "Cà Mau" in mock_model.last_kwargs["hotwords"]


def test_combined_context_and_expanded_hotwords():
    from dataclasses import replace
    from app.config import get_settings
    from app.asr import FasterWhisperAsrService

    settings = replace(get_settings(), asr_model="dummy-model", asr_language="vi")
    service = FasterWhisperAsrService(settings)

    class MockModel:
        def __init__(self):
            self.last_kwargs = {}

        def transcribe(self, samples, **kwargs):
            self.last_kwargs = kwargs
            segment = type("Segment", (), {"text": "chính xác", "avg_logprob": -0.1, "no_speech_prob": 0.01})()
            info = type("Info", (), {"language": "vi", "language_probability": 0.99})()
            return iter([segment]), info

    mock_model = MockModel()
    service._model = mock_model
    profile = CustomerProfile(pending_allergy_status="declared")
    result = asyncio.run(service.transcribe(PcmAudio(b"\x00\x08" * 3200, 16000), customer_profile=profile))

    assert result.text == "chính xác"
    assert "chính xác" in mock_model.last_kwargs["hotwords"]
    assert "bún bò Huế" in mock_model.last_kwargs["hotwords"]
    assert "dị ứng" in mock_model.last_kwargs["hotwords"]


def test_settings_default_to_vietnamese_with_rich_hotwords():
    from app.config import get_settings
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.asr_language == "vi"
    assert settings.tts_default_language == "vi"
    assert "phở" in settings.asr_hotwords
    assert "bún bò Huế" in settings.asr_hotwords
    assert "cơm tấm" in settings.asr_hotwords
    assert "dị ứng" in settings.asr_hotwords
    assert "Cần Thơ" in settings.asr_hotwords
    assert "Hà Nội" in settings.asr_hotwords
