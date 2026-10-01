"""Manage the 24-sample raw/RNNoise/Whisper comparison run."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = ROOT / "logs" / "voice"
STATE_PATH = VOICE_DIR / "benchmark-state.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PHRASES = [
    ("Xin chào, tôi cần tư vấn món ăn.", ["xin chào"]),
    ("Gợi ý món ăn sáng ở Nha Trang.", ["Nha Trang"]),
    ("Tôi dị ứng với đậu phộng và tôm.", ["dị ứng", "đậu phộng", "tôm"]),
    ("Tôi không dị ứng với sữa và trứng.", ["không", "dị ứng", "sữa", "trứng"]),
]
CONDITIONS = ["quiet", "normal", "motor"]


def cases() -> list[dict]:
    return [
        {
            "ground_truth": phrase,
            "critical_terms": critical,
            "condition": condition,
            "repeat": repeat,
        }
        for condition in CONDITIONS
        for phrase, critical in PHRASES
        for repeat in (1, 2)
    ]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"[^\w\s]", "", text).strip()


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def begin() -> None:
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "active": True,
        "next_index": 0,
        "cases": cases(),
        "kind": "benchmark_24",
        "session_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã bật benchmark 24 mẫu. Hãy nói theo đúng thứ tự bên dưới:")
    for index, case in enumerate(state["cases"], 1):
        print(f'{index:02d}. [{case["condition"]}, lần {case["repeat"]}] {case["ground_truth"]}')


def mic_check() -> None:
    """Arm one controlled speech capture followed by one silent capture."""
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    planned = [
        {
            "ground_truth": "Gợi ý món ăn sáng ở Nha Trang.",
            "critical_terms": ["Nha Trang"],
            "condition": "mic_path_speech_15_20cm_motor_off_speaker_silent",
            "repeat": 1,
        },
        {
            "ground_truth": "",
            "critical_terms": [],
            "condition": "mic_path_silence_motor_off_speaker_silent",
            "repeat": 1,
        },
    ]
    state = {
        "active": True,
        "next_index": 0,
        "cases": planned,
        "kind": "mic_check",
        "session_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã sẵn sàng thu 2 lượt kiểm tra đường mic:")
    print("1. Nói bình thường ở khoảng cách 15–20 cm: Gợi ý món ăn sáng ở Nha Trang.")
    print("2. Giữ im lặng trong cùng điều kiện cho đến khi backend kết thúc lượt.")


def allergy_mic_check() -> None:
    """Arm the focused six-capture seafood/silence experiment."""
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    planned = []
    for phrase, critical, condition in (
        ("hải sản", ["hải sản"], "short_seafood_phrase"),
        (
            "Tôi bị dị ứng với hải sản.",
            ["dị ứng", "hải sản"],
            "full_seafood_allergy_sentence",
        ),
        ("", [], "user_silence"),
    ):
        for repeat in (1, 2):
            planned.append({
                "ground_truth": phrase,
                "critical_terms": critical,
                "condition": condition,
                "repeat": repeat,
            })
    state = {
        "active": True,
        "next_index": 0,
        "cases": planned,
        "kind": "allergy_mic_check_6",
        "session_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã sẵn sàng thử 6 lượt, giữ nguyên điều kiện thu:")
    for index, case in enumerate(planned, 1):
        spoken = case["ground_truth"] or "<IM LẶNG>"
        print(f'{index}. [lần {case["repeat"]}] {spoken}')


def vad06_check() -> None:
    """Validate silence rejection and quiet speech at the tuned VAD threshold."""
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    planned = [
        {
            "ground_truth": "",
            "critical_terms": [],
            "condition": "vad06_silence",
            "repeat": 1,
        },
        {
            "ground_truth": "",
            "critical_terms": [],
            "condition": "vad06_silence",
            "repeat": 2,
        },
        {
            "ground_truth": "hải sản",
            "critical_terms": ["hải sản"],
            "condition": "vad06_quiet_speech",
            "repeat": 1,
        },
        {
            "ground_truth": "Tôi bị dị ứng với hải sản.",
            "critical_terms": ["dị ứng", "hải sản"],
            "condition": "vad06_quiet_speech",
            "repeat": 1,
        },
    ]
    state = {
        "active": True,
        "next_index": 0,
        "cases": planned,
        "kind": "vad06_check_4",
        "session_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã sẵn sàng kiểm tra VAD 0.6 với 4 lượt:")
    print("1. <IM LẶNG>")
    print("2. <IM LẶNG>")
    print("3. Nói nhỏ nhưng rõ: Hải sản")
    print("4. Nói nhỏ nhưng rõ: Tôi bị dị ứng với hải sản.")


def status() -> None:
    if not STATE_PATH.exists():
        print("Chưa có benchmark.")
        return
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    done = int(state.get("next_index", 0))
    total = len(state.get("cases", []))
    print(f"Đã ghi {done}/{total} mẫu; active={state.get('active', False)}")
    if done < total:
        case = state["cases"][done]
        print(f'Lượt tiếp theo: [{case["condition"]}, lần {case["repeat"]}] {case["ground_truth"]}')


def report() -> None:
    records = []
    for path in VOICE_DIR.glob("sample-*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("ground_truth") is not None and record.get("benchmark_index"):
            records.append(record)
    records.sort(key=lambda item: item["benchmark_index"])
    if not records:
        print("Chưa có mẫu benchmark đã gắn đáp án.")
        return
    totals = {"raw": {"chars": 0, "errors": 0, "critical": 0, "empty": 0},
              "rnnoise": {"chars": 0, "errors": 0, "critical": 0, "empty": 0}}
    print("idx condition  raw CER/critical | rnnoise CER/critical")
    for record in records:
        truth = normalize(record["ground_truth"])
        row = []
        for source in ("raw", "rnnoise"):
            result = (record.get("asr") or {}).get(source) or {}
            hypothesis = normalize(result.get("text", ""))
            distance = edit_distance(truth, hypothesis)
            terms_ok = all(normalize(term) in hypothesis for term in record["critical_terms"])
            totals[source]["chars"] += max(1, len(truth))
            totals[source]["errors"] += distance
            totals[source]["critical"] += int(terms_ok)
            totals[source]["empty"] += int(not hypothesis)
            row.append(f"{distance/max(1, len(truth)):.2f}/{'OK' if terms_ok else 'FAIL'}")
        print(f'{record["benchmark_index"]:02d}  {record["condition"]:<9} {row[0]:<16} | {row[1]}')
    for source in ("raw", "rnnoise"):
        value = totals[source]
        print(
            f'{source}: CER={value["errors"]/value["chars"]:.3f}, '
            f'critical={value["critical"]}/{len(records)}, empty={value["empty"]}/{len(records)}'
        )


def stop() -> None:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state["active"] = False
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã dừng benchmark.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["begin", "mic_check", "allergy_mic_check", "vad06_check", "status", "report", "stop"],
    )
    arguments = parser.parse_args()
    globals()[arguments.command]()


if __name__ == "__main__":
    main()
