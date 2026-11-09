#!/usr/bin/env python3
"""Concatenate fixed corpus clips with silence, keeping evaluation data outside the app."""
import argparse
import hashlib
import json
from pathlib import Path
import wave

parser = argparse.ArgumentParser()
parser.add_argument("--audio", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True, help="New directory outside the repository")
parser.add_argument("--clips", type=int, default=24)
parser.add_argument("--pause-seconds", type=float, default=4)
args = parser.parse_args()
manifest = json.loads((args.audio / "clips.json").read_text())
if not 0 < args.clips <= len(manifest["clips"]) or args.pause_seconds < 0:
    parser.error("Choose an available positive clip count and a nonnegative pause")
args.output.mkdir(parents=True, exist_ok=False)
pause = bytes(round(args.pause_seconds * 16000) * 2)
clips = []
samples = 0
pcm_path = args.output / "workload.pcm"
with pcm_path.open("wb") as output:
    for clip in manifest["clips"][:args.clips]:
        wav_path = args.audio / clip["wav"]
        if hashlib.sha256(wav_path.read_bytes()).hexdigest() != clip["sha256"]:
            raise ValueError(f"WAV hash mismatch: {wav_path}")
        with wave.open(str(wav_path)) as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                raise ValueError(f"Expected mono 16 kHz PCM16: {wav_path}")
            pcm = audio.readframes(audio.getnframes())
        clips.append(dict(index=clip["index"], begin_seconds=samples / 16000,
                          speech_end_seconds=(samples + len(pcm) // 2) / 16000,
                          reference=clip["reference"]))
        output.write(pcm)
        output.write(pause)
        samples += (len(pcm) + len(pause)) // 2
workload = dict(dataset=f"{manifest['dataset']}, first {args.clips} clips in existing selection order",
                revision=manifest["revision"], pause_seconds=args.pause_seconds,
                seconds=samples / 16000, sha256=hashlib.sha256(pcm_path.read_bytes()).hexdigest(),
                clips=clips)
(args.output / "workload.json").write_text(json.dumps(workload, indent=2) + "\n")
print(f"{pcm_path}: {workload['seconds']} seconds, SHA-256 {workload['sha256']}")
