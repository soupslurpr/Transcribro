#!/usr/bin/env python3
"""Convert saved, digest-checked benchmark WAVs to native float32 fixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import wave

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        nargs=2,
        action="append",
        metavar=("LABEL", "DIRECTORY"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for label, directory in args.corpus:
        if not re.fullmatch(r"[a-z][a-z0-9-]*", label):
            raise ValueError("invalid corpus label")
        directory = Path(directory).resolve()
        supplied = json.loads((directory / "clips.json").read_text())
        if not supplied["clips"]:
            raise ValueError("empty supplied corpus")
        for clip in supplied["clips"]:
            index = str(clip["index"])
            if not re.fullmatch(r"[A-Za-z0-9_-]+", index):
                raise ValueError("invalid clip index")
            source = directory / clip["wav"]
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != clip["sha256"]:
                raise ValueError("supplied recording digest mismatch")
            with wave.open(str(source)) as audio:
                if (
                    audio.getnchannels(),
                    audio.getsampwidth(),
                    audio.getframerate(),
                    audio.getcomptype(),
                ) != (1, 2, 16000, "NONE"):
                    raise ValueError("expected mono 16 kHz PCM16 WAV")
                samples = np.frombuffer(audio.readframes(audio.getnframes()), "<i2")
            pcm = (samples.astype("<f4") / np.float32(32768)).tobytes()
            filename = label + "-" + index + ".f32"
            with (args.output / filename).open("xb") as destination:
                destination.write(pcm)
            records.append(
                dict(
                    dataset=label,
                    index=clip["index"],
                    filename=filename,
                    reference=clip["reference"],
                    source_sha256=digest,
                    pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                )
            )
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Prepared", len(records), "digest-checked recordings")


if __name__ == "__main__":
    main()
