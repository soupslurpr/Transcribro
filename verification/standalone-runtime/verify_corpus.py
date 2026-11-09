#!/usr/bin/env python3
"""Run supplied benchmark speech without importing or loading ONNX Runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import wave

import numpy as np
from pipeline import Pipeline


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower().replace("’", "'"))


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        result = [i]
        for j, y in enumerate(b, 1):
            result.append(min(result[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = result
    return row[-1]


def main():
    parser = argparse.ArgumentParser()
    for name in ["library", "archive", "tokenizer", "audio", "reference", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--native-stream", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace corpus report")
    manifest = json.loads((args.audio / "clips.json").read_text())
    reference = json.loads(args.reference.read_text())
    expected = {v["index"]: v for v in reference["results"]["moonshine-medium"]}
    records = []
    started = time.monotonic()
    pipeline = Pipeline
    if args.native_stream:
        from native_pipeline import Pipeline as NativePipeline

        pipeline = NativePipeline
    engine = pipeline(args.library, args.archive, args.tokenizer)
    load_seconds = time.monotonic() - started
    try:
        for clip in manifest["clips"][: args.limit]:
            path = args.audio / clip["wav"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]:
                raise ValueError("evaluation audio changed")
            with wave.open(str(path)) as f:
                if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (
                    1,
                    2,
                    16000,
                ):
                    raise ValueError("unsupported supplied audio")
                audio = (
                    np.frombuffer(f.readframes(f.getnframes()), "<i2").astype("f4")
                    / 32768
                )
            start = time.monotonic()
            hypothesis = engine.transcribe(audio)
            record = {
                "index": clip["index"],
                "seconds": len(audio) / 16000,
                "processing_seconds": time.monotonic() - start,
                "hypothesis": hypothesis,
                "words": len(words(clip["reference"])),
                "errors": distance(words(clip["reference"]), words(hypothesis)),
                "reference_hypothesis": expected[clip["index"]]["hypothesis"],
                "exact": hypothesis == expected[clip["index"]]["hypothesis"],
            }
            records.append(record)
            print(
                len(records),
                clip["index"],
                "exact" if record["exact"] else "DIFFERENT",
                hypothesis,
                flush=True,
            )
            args.output.with_suffix(".partial.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
    finally:
        engine.close()
    if (
        "onnxruntime" in sys.modules
        or "libonnxruntime.so" in Path("/proc/self/maps").read_text()
    ):
        raise RuntimeError("ORT unexpectedly loaded")
    identity = {
        name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
        for name in ["library", "archive", "tokenizer", "reference"]
    }
    summary = {
        "dataset": manifest["dataset"],
        "revision": manifest["revision"],
        "sha256": identity,
        "load_seconds": load_seconds,
        "ort_loaded": False,
        "native_stream": args.native_stream,
        "clips": len(records),
        "exact_transcripts": sum(v["exact"] for v in records),
        "words": sum(v["words"] for v in records),
        "errors": sum(v["errors"] for v in records),
        "records": records,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    args.output.with_suffix(".partial.json").unlink()
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    if summary["exact_transcripts"] != len(records):
        sys.exit(1)


if __name__ == "__main__":
    main()
