import json
import wave

from scripts import replay_voice_models as replay


def test_replay_archives_inputs_without_promoting_unverified_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "ROOT", tmp_path / "no-archives")
    source = tmp_path / "voice"
    source.mkdir()
    wav = source / "sample-00-raw.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 1600)
    metadata = {"audio": {"raw": wav.name}, "ground_truth": "scheduled, not confirmed"}
    path = source / "sample-00.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    original = (wav.read_bytes(), path.read_bytes())
    output = tmp_path / "report"

    replay.prepare(source, output)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1
    assert manifest[0]["reference_verified"] is False
    assert manifest[0]["ground_truth"] is None
    assert (output / manifest[0]["wav"]).read_bytes() == original[0]
    assert (wav.read_bytes(), path.read_bytes()) == original
