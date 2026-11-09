#!/usr/bin/env python3
"""Summarize the phase-only diagnostics from device_app.py --startup-only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("report already exists")
    results = json.loads((args.runs / "results.json").read_text())
    assert len(results) == 8 and all(r["correct_numbers"] == 10 for r in results)
    trials = []
    for saved in results:
        directory = args.runs / saved["name"]
        conditions = json.loads((directory / "conditions.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        assert result["complete"] and result["error"] == 0
        assert result["bytes_sent"] == result["input_bytes"]
        assert conditions["audio_starts_with_request"]
        log = directory / "load-timings.log"
        records = re.findall(
            r"create (\w+) wall_ms=([\d.]+) cpu_ms=([\d.]+)", log.read_text()
        )
        assert [r[0] for r in records] == ["archive", "assets", "standalone", "vad"]
        phases = {
            name: {"wall_ms": float(wall), "cpu_ms": float(cpu)}
            for name, wall, cpu in records
        }
        samples = [
            json.loads(line)
            for line in (directory / "resources.jsonl").read_text().splitlines()
        ]
        first_text = next(e["ms"] for e in result["events"] if e.get("text"))
        trials.append(
            dict(
                name=saved["name"],
                group=saved["name"].split("-startup-")[0],
                apk_sha256=conditions["apk_sha256"],
                input_sha256=conditions["input_sha256"],
                capture_ready_ms=result["ready_ms"],
                first_text_ms=first_text,
                native_load_wall_ms=sum(p["wall_ms"] for p in phases.values()),
                native_load_cpu_ms=sum(p["cpu_ms"] for p in phases.values()),
                phases_ms=phases,
                skin_start_c=samples[0]["temperatures"]["VIRTUAL-SKIN"]["celsius"],
                maximum_thermal_status=max(s["thermal_status"] for s in samples),
                result_sha256=digest(directory / "result.json"),
                timings_sha256=digest(log),
            )
        )
    assert len({t["input_sha256"] for t in trials}) == 1
    labels = list(dict.fromkeys(t["group"] for t in trials))
    assert len(labels) == 2
    assert [labels.index(t["group"]) for t in trials] == [0, 1, 1, 0, 1, 0, 0, 1]
    groups = {}
    for label in labels:
        selected = [t for t in trials if t["group"] == label]
        assert len(selected) == 4 and len({t["apk_sha256"] for t in selected}) == 1
        groups[label] = dict(trials=4, apk_sha256=selected[0]["apk_sha256"])
        for field in [
            "capture_ready_ms",
            "first_text_ms",
            "native_load_wall_ms",
            "native_load_cpu_ms",
        ]:
            values = [t[field] for t in selected]
            groups[label][field] = dict(
                median=statistics.median(values),
                minimum=min(values),
                maximum=max(values),
            )
        groups[label]["phases_ms"] = {
            phase: {
                clock: statistics.median(t["phases_ms"][phase][clock] for t in selected)
                for clock in ["wall_ms", "cpu_ms"]
            }
            for phase in ["archive", "assets", "standalone", "vad"]
        }
    summary = dict(
        trials=trials,
        groups=groups,
        limits=[
            "One Pixel 8a; four trials per diagnostic APK in A-B-B-A-B-A-A-B order.",
            "Supplied PCM starts with the recognition request; no microphone or speakers.",
            "A fresh isolated worker loads the model each time; installation and OS file caches are warm.",
            "Native load timing excludes process creation, Binder calls and capture setup.",
            "Capture readiness occurs before native model loading finishes; audio buffers during loading.",
            "Phase logging is enabled only in these diagnostic APKs, not the distributed candidate.",
        ],
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
