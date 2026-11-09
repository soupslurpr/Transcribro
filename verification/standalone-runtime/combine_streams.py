#!/usr/bin/env python3
"""Combine completed correctness batches, including runs resumed after interruption."""

import argparse
import hashlib
import json
from pathlib import Path


def combine(reports, expected):
    clips = []
    for report in reports:
        if report["mode"] != "native" or report["ort_loaded"] is not False:
            raise ValueError("expected a completed native run without ONNX Runtime")
        if report.get("independent_recordings") is not True:
            raise ValueError(
                "recordings must reset all state, including the speech detector"
            )
        for clip in report["clips"]:
            updates = clip["updates"]
            if (
                not updates
                or not updates[-1]["final"]
                or updates[-1]["text"] != clip["text"]
            ):
                raise ValueError("incomplete recording")
            clips.append(clip)
    names = [Path(clip["path"]).name for clip in clips]
    if len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError("duplicate, missing, or unexpected recordings")
    return dict(
        mode="native",
        ort_loaded=False,
        independent_recordings=True,
        clips=clips,
        batches=len(reports),
        timings_for_correctness_only=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite combined report")
    supplied = json.loads(args.manifest.read_text())
    expected = {row["filename"]: row["pcm_sha256"] for row in supplied}
    if len(expected) != len(supplied):
        raise ValueError("duplicate manifest recording")
    reports, evidence = [], {}
    identity = None
    for directory in args.runs:
        inputs = json.loads((directory / "inputs.json").read_text())
        current = {
            key: value for key, value in inputs.items() if not key.endswith(".f32")
        }
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError("runtime, model, or tokenizer changed between batches")
        paths = sorted(directory.glob("stream-batch-*.json"))
        if not paths:
            raise ValueError("no completed batches")
        for path in paths:
            report = json.loads(path.read_text())
            for clip in report["clips"]:
                name = Path(clip["path"]).name
                if name not in expected or inputs.get(name) != expected[name]:
                    raise ValueError("recording digest mismatch")
            reports.append(report)
            evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = combine(reports, expected)
    result["input_identity"] = identity
    result["batch_sha256"] = evidence
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Combined", len(reports), "completed batches:", len(result["clips"]), "clips")


if __name__ == "__main__":
    main()
