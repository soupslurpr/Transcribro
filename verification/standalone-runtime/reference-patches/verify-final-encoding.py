#!/usr/bin/env python3
"""Exercise the built native library; requires moonshine-voice==0.1.5, no microphone."""

import argparse
import array
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = {
    "abrupt": ("count-to-ten.pcm", "eca0a7a12313b9517115b16be5de8ea164b8b9e86790cb57c9fc36d808209485"),
    "slower": ("count-to-ten-slower.pcm", "55c8d02476a30cec94e64c7d31c6d7610db2a6db547bcc9074b45877fb3c5562"),
    "faster": ("count-to-ten-faster.pcm", "45f1fb1188c94342c001fbdb389e88c2825d1f6b1693c6bbb59182de481b6c03"),
}
NUMBERS = "one two three four five six seven eight nine ten".split()


def run_case(args):
    from moonshine_voice import Transcriber
    from moonshine_voice.moonshine_api import ModelArch, _MoonshineLib

    # Only override the test process's loader; do not replace installed libraries.
    native = object.__new__(_MoonshineLib)
    native._lib = ctypes.CDLL(str(args.native_library))
    native._setup_function_signatures()
    _MoonshineLib._instance = native

    fixture, expected_hash = FIXTURES[args.fixture]
    raw = (ROOT / "app/src/androidTest/assets" / fixture).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_hash, "Counting fixture changed"
    pcm = array.array("h", raw)
    if sys.byteorder != "little":
        pcm.byteswap()
    samples = [0.0] * args.case + [sample / 32768 for sample in pcm]
    if not args.raw:
        samples += [0.0] * (-len(samples) % 512)
    options = {
        "vad_threshold": args.threshold,
        "max_tokens_per_second": args.tokens_per_second,
        "max_preview_tokens_per_second": args.preview_tokens_per_second,
        "decode_incomplete_lines": args.previews,
        "transcription_interval": 1.0 if args.previews == "true" else 0.5,
        "return_audio_data": False,
        "log_output_text": False,
    }
    completed = {}

    def collect(transcript):
        for line in transcript.lines:
            if line.is_complete:
                completed.setdefault(line.line_id, line.text)

    with Transcriber(args.model_directory, ModelArch(args.model_arch), options=options) as engine:
        stream = engine.create_stream(update_interval=100000.0)
        stream.start()
        pending = []
        started = time.monotonic()
        for offset in range(0, len(samples), 512):
            pending.extend(samples[offset : offset + 512])
            if len(pending) == 8192:
                stream.add_audio(pending)
                collect(stream.update_transcription())
                pending = []
            if args.pace:
                delivered = min(offset + 512, len(samples))
                time.sleep(max(0.0, started + delivered / 16000 - time.monotonic()))
        if pending:
            stream.add_audio(pending)
        if args.final_drain_samples:
            stream.add_audio([0.0] * args.final_drain_samples)
        collect(stream.stop())
        stream.close()

    text = " ".join(completed.values()).strip()
    numbers = [
        int(word) if word.isdigit() else NUMBERS.index(word) + 1 if word in NUMBERS else 0
        for word in re.findall(r"[a-z]+|[0-9]+", text.lower())
    ]
    result = {
        "prefix": args.case,
        "previews": args.previews,
        "threshold": args.threshold,
        "fixture": args.fixture,
        "max_tokens_per_second": args.tokens_per_second,
        "max_preview_tokens_per_second": args.preview_tokens_per_second,
        "final_drain_samples": args.final_drain_samples,
        "model_arch": args.model_arch,
        "frame_padding": 1 if args.raw else 512,
        "text": text,
        "passed": numbers == list(range(1, 11)),
    }
    print(json.dumps(result), flush=True)
    return result["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.set_defaults(model_arch=5)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--tokens-per-second", type=float, default=20.0)
    parser.add_argument("--preview-tokens-per-second", type=float, default=6.5)
    parser.add_argument("--final-drain-samples", type=int, default=8192)
    parser.add_argument("--fixture", choices=FIXTURES, default="slower")
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--raw", action="store_true", help="Do not pad the last partial frame")
    parser.add_argument("--case", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--previews", choices=["true", "false"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.final_drain_samples < 0:
        parser.error("--final-drain-samples must be nonnegative")
    args.native_library = args.native_library.resolve(strict=True)
    args.model_directory = args.model_directory.resolve(strict=True)
    os.environ["MOONSHINE_ORT_SINGLE_THREAD"] = "1"
    if args.case is not None:
        return 0 if run_case(args) else 1

    # A fresh process per case also isolates upstream's global Silero state.
    passed = True
    for previews in ["true", "false"]:
        for prefix in [0, 1, 257, 511, 777, 1600]:
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--native-library", str(args.native_library),
                "--model-directory", str(args.model_directory),
                "--model-arch", str(args.model_arch),
                "--threshold", str(args.threshold),
                "--tokens-per-second", str(args.tokens_per_second),
                "--preview-tokens-per-second", str(args.preview_tokens_per_second),
                "--final-drain-samples", str(args.final_drain_samples),
                "--fixture", args.fixture,
                "--case", str(prefix), "--previews", previews,
            ]
            if args.pace:
                command.append("--pace")
            if args.raw:
                command.append("--raw")
            result = subprocess.run(command, check=False)
            passed = result.returncode == 0 and passed
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
