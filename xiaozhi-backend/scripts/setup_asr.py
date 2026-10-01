"""Explicitly download/cache a faster-whisper model once."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model id, e.g. small or medium")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    kwargs = {"device": "cpu", "compute_type": "int8", "local_files_only": False}
    if args.cache_dir:
        kwargs["download_root"] = str(args.cache_dir)
    WhisperModel(args.model, **kwargs)
    print(f"ASR model cached: {args.model}")


if __name__ == "__main__":
    main()
