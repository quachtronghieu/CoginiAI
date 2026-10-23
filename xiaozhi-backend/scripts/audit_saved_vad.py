"""Compare endpoint waits using saved probabilities; never modify source recordings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def endpoint(probabilities, silence_ms, threshold=.6, frame_ms=32):
    import torch
    from silero_vad import VADIterator

    class Model:
        probability = 0.0
        def reset_states(self):
            pass
        def __call__(self, value, rate):
            return torch.tensor([self.probability])

    model = Model()
    iterator = VADIterator(model, threshold=threshold, sampling_rate=16000,
        min_silence_duration_ms=silence_ms, speech_pad_ms=200)
    speech = False
    end_index = None
    reason = "end_of_recording"
    for index, probability in enumerate(probabilities):
        model.probability = probability
        event = iterator(torch.zeros(512))
        if event and "start" in event:
            speech = True
        if event and "end" in event:
            end_index, reason = index, "silence_endpoint"
            break
        if not speech and (index + 1) * frame_ms >= 6000:
            end_index, reason = index, "idle_timeout"
            break
    later_frames = [index for index in range(end_index + 1, len(probabilities))
                    if probabilities[index] >= threshold] if end_index is not None else []
    return {"silence_ms": silence_ms, "speech_detected": speech, "reason": reason,
        "decision_audio_ms": (end_index + 1) * frame_ms if end_index is not None else None,
        "later_speech_frames": len(later_frames),
        "later_speech_audio_ms": len(later_frames) * frame_ms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "logs" / ("vad-endpoints-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for directory in args.voice_dir:
        for path in sorted(directory.glob("sample-*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            evidence = record.get("vad", {})
            probabilities = evidence.get("frame_probabilities", [])
            if not probabilities:
                continue
            raw_name = record.get("audio", {}).get("raw")
            raw_path = directory / raw_name if raw_name else None
            rows.append({"json": str(path.resolve()),
                "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path and raw_path.is_file() else None,
                "reference_verified": record.get("label_source") == "user_confirmed",
                "ground_truth": record.get("ground_truth") if record.get("label_source") == "user_confirmed" else None,
                "threshold": evidence.get("threshold"),
                "results": [endpoint(probabilities, wait, evidence.get("threshold", .6), evidence.get("frame_ms", 32)) for wait in (1000, 800, 700)]})
    (output / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Saved VAD endpoint audit", "",
        "Uses recorded probabilities, with unchanged speech threshold. Original packet timing is unavailable.",
        "Later speech after a decision requires review; it is not automatically an error or a separate turn.",
        "Absence of later speech does not validate pauses or quiet speech missing from these recordings.", "",
        "| JSON | Reference confirmed | Wait ms | Decision ms | Later speech ms |", "|---|---|---:|---:|---:|"]
    for row in rows:
        for result in row["results"]:
            lines.append(f"| {Path(row['json']).parent.name}/{Path(row['json']).name} | {row['reference_verified']} | {result['silence_ms']} | {result['decision_audio_ms']} | {result['later_speech_audio_ms']} |")
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"REPORT_DIR={output.resolve()} samples={len(rows)}")


if __name__ == "__main__":
    main()
