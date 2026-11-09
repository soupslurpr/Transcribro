#!/usr/bin/env python3
"""Run the ORT-backed audio comparator; never install or launch an app."""
import argparse
import json
from pathlib import Path
import subprocess
import uuid


def main():
    p = argparse.ArgumentParser()
    for name in ["adb", "build", "archive", "models", "vad", "pcm", "ort", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--serial", required=True)
    args = p.parse_args()
    if args.serial.startswith("emulator-") or args.output.exists():
        raise ValueError("physical device and new output required")
    device = "/data/local/tmp/transcribro-audio-" + uuid.uuid4().hex
    adb = [str(args.adb), "-s", args.serial]

    def run(*a, **kw):
        return subprocess.run(adb + list(a), check=True, text=True, **kw)

    try:
        run("shell", "mkdir", device)
        files = [
            args.archive,
            args.build / "verify_audio",
            args.build / "libtranscribro_standalone.so",
            args.vad,
            args.pcm,
            args.models / "frontend.model.ort",
            args.models / "frontend.weights.ort",
            args.ort,
        ]
        for f in files:
            run(
                "push",
                str(f),
                device + "/",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        command = f"cd {device} && LD_LIBRARY_PATH=. ./verify_audio {args.archive.name} . {args.vad.name} {args.pcm.name}"
        result = subprocess.run(
            adb + ["shell", command], text=True, capture_output=True
        )
        args.output.write_text(result.stdout)
        print(result.stdout, end="")
        print(result.stderr, end="")
        if result.returncode:
            raise RuntimeError(f"audio comparison exited {result.returncode}")
    finally:
        run("shell", "rm", "-rf", device)
        run("shell", "test", "!", "-e", device)


if __name__ == "__main__":
    main()
