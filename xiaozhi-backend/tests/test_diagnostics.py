from pathlib import Path
import json

from app.ai import PcmAudio
from app.diagnostics import VoiceDiagnostics, supported_kwargs


def test_voice_audio_retention_is_bounded(tmp_path):
    diagnostics = VoiceDiagnostics(tmp_path, slots=2)
    pcm = PcmAudio(b"\x00\x00" * 160, 16000)
    for _ in range(7):
        diagnostics.save_pair(pcm, pcm)
    assert len(list(tmp_path.glob("*.wav"))) == 4
    assert len(list(tmp_path.glob("sample-*.json"))) == 2


def test_raw_and_metadata_exist_before_filter_or_asr(tmp_path):
    diagnostics = VoiceDiagnostics(tmp_path, slots=2)
    pcm = PcmAudio(b"\x01\x00" * 160, 16000)

    key = diagnostics.start_sample(pcm)

    assert (tmp_path / f"{key}-raw.wav").exists()
    assert not (tmp_path / f"{key}-rnnoise.wav").exists()
    metadata = json.loads((tmp_path / f"{key}.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "raw_saved"
    assert metadata["ground_truth"] is None


def test_active_benchmark_assigns_ground_truth(tmp_path):
    state = {
        "active": True,
        "next_index": 0,
        "cases": [{
            "ground_truth": "Tôi không dị ứng với tôm",
            "condition": "motor",
            "repeat": 1,
            "critical_terms": ["không", "tôm"],
        }],
    }
    (tmp_path / "benchmark-state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
    )
    diagnostics = VoiceDiagnostics(tmp_path)
    pcm = PcmAudio(b"\x00\x00" * 160, 16000)

    key = diagnostics.start_sample(pcm)

    metadata = json.loads((tmp_path / f"{key}.json").read_text(encoding="utf-8"))
    assert metadata["ground_truth"] == "Tôi không dị ứng với tôm"
    assert metadata["critical_terms"] == ["không", "tôm"]


def test_signature_adaptation_does_not_execute_callable():
    def legacy(text):
        raise TypeError("internal error")
    assert supported_kwargs(legacy, {"history": [], "customer_profile": None}) == {}
    def modern(text, **kwargs):
        return text
    assert supported_kwargs(modern, {"customer_profile": "profile"}) == {"customer_profile": "profile"}
