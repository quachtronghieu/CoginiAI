"""Print short-window energy around the Whisper VAD boundary for sample 00."""

from pathlib import Path
import wave

import numpy as np


voice_dir = Path(__file__).resolve().parents[1] / "logs" / "voice"
for name in ("sample-00-raw.wav", "sample-00-rnnoise.wav"):
    with wave.open(str(voice_dir / name), "rb") as source:
        samples = (
            np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
            .astype(np.float32)
            / 32768.0
        )
    window_samples = 320  # 20 ms at 16 kHz
    rms = np.array(
        [
            np.sqrt(np.mean(samples[index : index + window_samples] ** 2) + 1e-12)
            for index in range(0, len(samples) - window_samples + 1, window_samples)
        ]
    )
    dbfs = 20 * np.log10(rms)
    print(name)
    for threshold in (-45, -40, -35, -30, -25):
        indices = np.where(dbfs > threshold)[0]
        print(
            "threshold",
            threshold,
            "first_s",
            round(indices[0] * 0.02, 3) if len(indices) else None,
            "last_s",
            round((indices[-1] + 1) * 0.02, 3) if len(indices) else None,
        )
    print(
        "windows_1.20_1.60",
        [
            (round(index * 0.02, 2), round(float(dbfs[index]), 1))
            for index in range(60, 81)
        ],
    )
