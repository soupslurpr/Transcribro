#!/usr/bin/env python3
"""Summarize completed count trials without copying device identifiers or speech."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def summarize(directory):
    def read(name):
        return json.loads((directory / name).read_text())

    result = read("result.json")
    conditions = read("conditions.json")
    samples = [
        json.loads(line)
        for line in (directory / "resources.jsonl").read_text().splitlines()
    ]
    after = [
        json.loads(line)
        for line in (directory / "resources-after.jsonl").read_text().splitlines()
    ]
    log = (directory / "instrumentation.log").read_text()
    assert "OK (1 test)" in log and "FAILURES" not in log
    assert result["complete"] and result["error"] == 0
    assert result["bytes_sent"] == result["input_bytes"]
    assert len(samples) >= 2 and len(after) == 3
    intervals = [
        b["host_monotonic"] - a["host_monotonic"] for a, b in zip(samples, samples[1:])
    ]
    names = "one two three four five six seven eight nine ten".split()
    numbers = [
        int(token) if token.isdigit() else names.index(token) + 1
        for token in re.findall(r"[a-z]+|[0-9]+", result["text"].lower())
    ]
    repeats = 12 if result["audio_seconds"] > 60 else 1
    assert numbers == list(range(1, 11)) * repeats
    cpu = {}
    for sample in samples:
        for process in sample["processes"]:
            pid = process["pid"]
            cpu[pid] = max(
                cpu.get(pid, 0), process["user_ticks"] + process["system_ticks"]
            )
    temperatures = {}
    for name in ["VIRTUAL-SKIN", "BIG", "MID", "LITTLE", "battery"]:
        values = [s["temperatures"][name]["celsius"] for s in samples]
        temperatures[name] = dict(start=values[0], peak=max(values), end=values[-1])
    power_before = {
        value["name"]: value
        for value in result.get("power_before", {}).get("monitors", [])
    }
    rails = {}
    for value in result.get("power_after", {}).get("monitors", []):
        name = value["name"]
        if value["type"] != 1 or name not in power_before:
            continue
        previous = power_before[name]
        seconds = (value["timestamp_ms"] - previous["timestamp_ms"]) / 1000
        if seconds <= 0 or min(value["energy_uws"], previous["energy_uws"]) < 0:
            continue
        joules = (value["energy_uws"] - previous["energy_uws"]) / 1e6
        if joules < 0:
            continue
        rails[name] = dict(seconds=seconds, joules=joules, watts=joules / seconds)
    energy = {}
    for group in ["CPU", "DDR"]:
        selected = [
            value
            for name, value in rails.items()
            if (f"]:{group}(" in name if group == "CPU" else name.endswith(":DDR"))
        ]
        energy[group] = (
            dict(
                joules=sum(v["joules"] for v in selected),
                watts=sum(v["watts"] for v in selected),
                rail_count=len(selected),
            )
            if selected
            else None
        )
    return dict(
        name=directory.name,
        apk_sha256=conditions["apk_sha256"],
        input_sha256=conditions["input_sha256"],
        audio_seconds=result["audio_seconds"],
        audio_starts_with_request=conditions["audio_starts_with_request"],
        correct_numbers=len(numbers),
        previews=conditions["previews"],
        ready_ms=result["ready_ms"],
        post_input_ms=result["finished_ms"] - result["input_end_ms"],
        first_text_ms=next(
            (v["ms"] for v in result["events"] if v["event"] in ["partial", "phrase"]),
            None,
        ),
        writer_lag_ms=result["max_writer_lag_ms"],
        peak_pss_mib=max(s["total_pss_kib"] for s in samples) / 1024,
        sampled_cpu_seconds=sum(cpu.values()) / int(conditions["clock_ticks"]),
        sampling=dict(
            samples=len(samples),
            requested_interval_seconds=conditions["sample_interval_seconds"],
            actual_interval_seconds=dict(
                minimum=min(intervals),
                median=statistics.median(intervals),
                maximum=max(intervals),
            ),
        ),
        post_cleanup_pss_mib=[s["total_pss_kib"] / 1024 for s in after],
        temperatures_c=temperatures,
        maximum_thermal_status=max(s["thermal_status"] for s in samples),
        device_rail_energy=energy,
        result_sha256=hashlib.sha256(
            (directory / "result.json").read_bytes()
        ).hexdigest(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", nargs="+", default=["standalone", "hybrid"])
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("report already exists")
    trials = [
        summarize(directory)
        for root in args.runs
        for directory in sorted(root.iterdir())
        if directory.is_dir() and (directory / "conditions.json").exists()
    ]
    groups = {}
    sustained = [v for v in trials if v["audio_seconds"] > 60]
    if sustained:
        assert len({v["input_sha256"] for v in sustained}) == 1
        assert len({v["previews"] for v in sustained}) == 1
        assert len({v["audio_starts_with_request"] for v in sustained}) == 1
    for name in args.groups:
        selected = [
            v for v in trials if v["name"].startswith(name) and v["audio_seconds"] > 60
        ]
        if not selected:
            continue
        assert len({v["apk_sha256"] for v in selected}) == 1
        groups[name] = dict(trials=len(selected))
        for field in [
            "peak_pss_mib",
            "sampled_cpu_seconds",
            "ready_ms",
            "post_input_ms",
        ]:
            values = [v[field] for v in selected]
            groups[name][field] = dict(
                median=statistics.median(values),
                minimum=min(values),
                maximum=max(values),
            )
    summary = dict(
        trials=trials,
        sustained=groups,
        limits=[
            "One Pixel 8a, supplied PCM, no physical microphone or speakers.",
            "CPU and peak PSS use a two-second target sampling interval; collection may take longer and can miss final work or short peaks.",
            "Power rails cover the whole device, including background work; not app battery-life measurements.",
            "Short trials with unchanged power timestamps have no valid energy reading.",
            "Models start in a fresh isolated worker; installations and OS file cache are warm.",
            "ready_ms measures the capture onReadyForSpeech callback, not completion of native model loading; audio buffers while the model loads.",
        ],
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
