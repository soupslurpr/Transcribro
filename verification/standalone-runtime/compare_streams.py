#!/usr/bin/env python3
"""Compare complete native streams on one architecture, including previews."""
import argparse
import hashlib
import json
from pathlib import Path
import re


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
    for name in ["baseline", "candidate", "manifest", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite comparison")
    supplied = json.loads(args.manifest.read_text())
    references = {r["filename"]: r for r in supplied}
    if len(references) != len(supplied):
        raise ValueError("duplicate supplied recording")
    reports = [json.loads(path.read_text()) for path in [args.baseline, args.candidate]]
    clips = []
    for index, report in enumerate(reports):
        if index == 0 and report["mode"] == "reference-independent":
            pass
        elif report["mode"] != "native" or report["ort_loaded"] is not False:
            raise ValueError("expected native execution without ONNX Runtime")
        if report.get("independent_recordings") is not True:
            raise ValueError(
                "recordings must reset all state, including the speech detector"
            )
        lookup = {Path(c["path"]).name: c for c in report["clips"]}
        if len(lookup) != len(report["clips"]) or lookup.keys() != references.keys():
            raise ValueError("recording sets differ")
        clips.append(lookup)
    groups = {}
    records = []
    for name, reference in references.items():
        before, after = (lookup[name] for lookup in clips)
        if before["audio_seconds"] != after["audio_seconds"]:
            raise ValueError("audio duration differs")
        frames = [
            [(u["samples"], u["final"]) for u in clip["updates"]]
            for clip in [before, after]
        ]
        if frames[0] != frames[1] or not frames[0] or not frames[0][-1][1]:
            raise ValueError("incomplete or differently scheduled stream")
        same_updates = sum(
            a["text"] == b["text"] for a, b in zip(before["updates"], after["updates"])
        )
        ref_words = words(reference["reference"])
        errors = [distance(ref_words, words(c["text"])) for c in [before, after]]
        group = groups.setdefault(
            reference["dataset"],
            dict(
                clips=0,
                exact_finals=0,
                updates=0,
                exact_updates=0,
                reference_words=0,
                baseline_errors=0,
                candidate_errors=0,
            ),
        )
        group["clips"] += 1
        group["exact_finals"] += before["text"] == after["text"]
        group["updates"] += len(frames[0])
        group["exact_updates"] += same_updates
        group["reference_words"] += len(ref_words)
        group["baseline_errors"] += errors[0]
        group["candidate_errors"] += errors[1]
        records.append(
            dict(
                dataset=reference["dataset"],
                index=reference["index"],
                final_exact=before["text"] == after["text"],
                updates=len(frames[0]),
                exact_updates=same_updates,
                baseline_errors=errors[0],
                candidate_errors=errors[1],
            )
        )
    exact = all(
        r["final_exact"] and r["updates"] == r["exact_updates"] for r in records
    )
    for group in groups.values():
        for kind in ["baseline", "candidate"]:
            group[kind + "_wer"] = group[kind + "_errors"] / group["reference_words"]
    result = dict(
        scope="Same-architecture supplied-PCM correctness comparison; timings are not thermally balanced app measurements",
        all_exact=exact,
        groups=groups,
        records=records,
        report_sha256={
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ["baseline", "candidate", "manifest"]
        },
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(all_exact=exact, groups=groups), indent=2))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
