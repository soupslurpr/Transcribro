#!/usr/bin/env python3
"""Check whether batching existing 80 ms blocks preserves frontend arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from verify_components import FrontendState, Model


def main():
    p = argparse.ArgumentParser()
    for name in ["library", "archive", "audio", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()
    if args.output.exists() or not 1 <= args.rounds <= 10:
        raise ValueError("new report and bounded rounds required")
    model = Model(args.library, args.archive)
    cases = []
    try:
        for path in sorted(args.audio.glob("*.f32")):
            audio = np.fromfile(path, dtype="<f4")
            audio = np.pad(audio, (0, -audio.size % 1280))
            reference, states = [], []
            state = FrontendState()
            for at in range(0, audio.size, 1280):
                reference.append(model.frontend(audio[at : at + 1280], state))
                states.append(bytes(state))
            expected = np.concatenate(reference, axis=1)
            for repeat in range(args.rounds):
                groups = [1, 2, 4, 8, 12, 24]
                if repeat % 2:
                    groups.reverse()
                for blocks in groups:
                    state = FrontendState()
                    cpu_ns = wall_ns = different = state_differences = 0
                    maximum = 0.0
                    for at in range(0, audio.size, blocks * 1280):
                        chunk = audio[at : at + blocks * 1280]
                        wall_start = time.monotonic_ns()
                        cpu_start = time.process_time_ns()
                        actual = model.frontend(chunk, state)
                        cpu_ns += time.process_time_ns() - cpu_start
                        wall_ns += time.monotonic_ns() - wall_start
                        target = expected[:, at // 320 : (at + chunk.size) // 320]
                        assert actual.shape == target.shape
                        different += int(
                            np.count_nonzero(actual.view("u4") != target.view("u4"))
                        )
                        maximum = max(maximum, float(np.max(np.abs(actual - target))))
                        end_block = (at + chunk.size) // 1280 - 1
                        state_differences += bytes(state) != states[end_block]
                    cases.append(
                        dict(
                            clip=path.name,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            round=repeat,
                            blocks=blocks,
                            samples=audio.size,
                            cpu_seconds=cpu_ns / 1e9,
                            wall_seconds=wall_ns / 1e9,
                            different_features=different,
                            different_final_states=state_differences,
                            maximum_error=maximum,
                        )
                    )
                    print(
                        path.name,
                        repeat,
                        blocks,
                        different,
                        state_differences,
                        flush=True,
                    )
    finally:
        model.close()
    summaries = {}
    for blocks in [1, 2, 4, 8, 12, 24]:
        selected = [c for c in cases if c["blocks"] == blocks]
        totals = [
            sum(c["cpu_seconds"] for c in selected if c["round"] == repeat)
            for repeat in range(args.rounds)
        ]
        summaries[str(blocks)] = dict(
            exact=all(
                not c["different_features"] and not c["different_final_states"]
                for c in selected
            ),
            median_cpu_seconds=statistics.median(totals),
            rounds_cpu_seconds=totals,
        )
    report = dict(
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        scope="Host component test; no extra audio is awaited to form a batch",
        summaries=summaries,
        cases=cases,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
