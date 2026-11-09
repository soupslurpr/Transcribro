#!/usr/bin/env python3
"""Profile native supplied-PCM work without changing apps or Android settings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid
from combine_streams import combine
from device_quiet import SpeechActivity


def main():
    p = argparse.ArgumentParser()
    for name in ["adb", "build", "archive", "tokenizer", "audio", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--serial", required=True)
    p.add_argument(
        "--steps",
        nargs="+",
        choices=[
            "profile",
            "stream",
            "frontend",
            "tiled",
            "head",
            "token_selection",
            "loading",
            "decompression",
            "attention",
            "local_attention",
            "projection",
            "runtime",
        ],
        default=["profile", "frontend"],
    )
    p.add_argument("--projections", type=Path)
    p.add_argument("--head", type=Path)
    p.add_argument("--head-width")
    p.add_argument("--compressed", type=Path)
    p.add_argument("--tiled-order", action="store_true")
    p.add_argument("--tiled-wide", action="store_true")
    p.add_argument("--cpus", default="40")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--stream-batch-size", type=int, default=0)
    args = p.parse_args()
    if args.serial.startswith("emulator-"):
        raise ValueError("physical device required")
    if args.timeout < 1 or args.timeout > 7200:
        raise ValueError("native timeout must be 1..7200 seconds")
    if not 0 <= args.stream_batch_size <= 128 or (
        args.stream_batch_size and args.steps != ["stream"]
    ):
        raise ValueError("stream batch size requires --steps stream and must be 1..128")
    if not re.fullmatch(r"[0-9a-fA-F]+", args.cpus) or not int(args.cpus, 16):
        raise ValueError("invalid CPU affinity mask")
    args.output.mkdir(parents=True, exist_ok=False)
    base = [str(args.adb), "-s", args.serial]
    directory = "/data/local/tmp/transcribro-opt-" + uuid.uuid4().hex
    active = None

    def shell(command, check=True):
        return subprocess.run(
            base + ["shell", command], check=check, capture_output=True, text=True
        ).stdout.strip()

    observer = SpeechActivity(base)
    activity = observer.snapshot

    def quiet(previous, current, seconds):
        ticks = observer.ticks
        for pid, value in current.items():
            if (
                pid not in previous
                or value["started"] != previous[pid]["started"]
                or value["ticks"] - previous[pid]["ticks"] > ticks * seconds * 0.02
            ):
                raise RuntimeError("Another speech process is active: " + value["name"])

    def stop_ours():
        for line in shell("ps -A -o PID,ARGS").splitlines()[1:]:
            fields = line.strip().split(None, 1)
            if len(fields) == 2 and fields[1].startswith(directory + "/"):
                shell("kill -TERM " + fields[0], check=False)

    try:
        previous = activity()
        started = time.monotonic()
        time.sleep(30)
        current = activity()
        quiet(previous, current, time.monotonic() - started)
        (args.output / "other-processes-idle.json").write_text(
            json.dumps(dict(before=previous, after=current), indent=2) + "\n"
        )
        shell("mkdir -p " + directory + "/audio")
        files = [
            args.archive,
            args.tokenizer,
            args.build / "libtranscribro_standalone.so",
        ]
        if "profile" in args.steps or "stream" in args.steps:
            files.append(args.build / "run_stream")
        if "frontend" in args.steps:
            files.append(args.build / "benchmark_frontend")
        if "loading" in args.steps:
            files.append(args.build / "benchmark_loading")
        if "decompression" in args.steps:
            if not args.compressed:
                raise ValueError("--compressed required")
            files += [args.build / "benchmark_decompression", args.compressed]
        if "tiled" in args.steps:
            if not args.projections:
                raise ValueError("--projections required")
            files += [args.build / "test_projection", args.build / "benchmark_tiled"]
        if "head" in args.steps or "token_selection" in args.steps:
            if not args.head:
                raise ValueError("--head required")
            files.append(args.head)
            if "head" in args.steps:
                files.append(args.build / "benchmark_head_layout")
            if "token_selection" in args.steps:
                files.append(args.build / "benchmark_token_selection")
        if "attention" in args.steps:
            files.append(args.build / "benchmark_attention")
        if "local_attention" in args.steps:
            files.append(args.build / "benchmark_local_attention")
        if "projection" in args.steps:
            files.append(args.build / "test_projection")
        if "runtime" in args.steps:
            files.append(args.build / "test_runtime")
        audio = sorted(args.audio.glob("*.f32"))
        if not audio:
            raise ValueError("no supplied PCM")
        manifest = {}
        for source in files + audio:
            target = directory + ("/audio/" if source in audio else "/") + source.name
            subprocess.run(
                base + ["push", str(source), target],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            manifest[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        if "tiled" in args.steps:
            for source in sorted(args.projections.glob("*/projection.bin")):
                target_dir = directory + "/projections/" + source.parent.name
                shell("mkdir -p " + shlex.quote(target_dir))
                subprocess.run(
                    base + ["push", str(source), target_dir + "/projection.bin"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                manifest["projections/" + source.parent.name] = hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
        (args.output / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
        listing = args.output / "pcm.txt"
        listing.write_text(
            "".join(directory + "/audio/" + f.name + "\n" for f in audio)
        )
        subprocess.run(
            base + ["push", str(listing), directory + "/pcm.txt"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        runs = [(step, step, audio) for step in args.steps]
        if args.stream_batch_size:
            size = args.stream_batch_size
            runs = [
                (
                    "stream",
                    f"stream-batch-{offset // size:03}",
                    audio[offset : offset + size],
                )
                for offset in range(0, len(audio), size)
            ]
        for run_index, (step, name, recordings) in enumerate(runs):
            if args.stream_batch_size:
                listing.write_text(
                    "".join(directory + "/audio/" + f.name + "\n" for f in recordings)
                )
                subprocess.run(
                    base + ["push", str(listing), directory + "/pcm.txt"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            deadline = time.monotonic() + 900
            while True:
                thermal = shell("dumpsys thermalservice")
                if args.stream_batch_size and run_index:
                    break  # These checkpoints are for correctness, not timing.
                live = thermal.split("Current temperatures from HAL:", 1)[1]
                match = re.search(
                    r"mValue=([0-9.]+), mType=3, mName=VIRTUAL-SKIN,", live
                )
                if not match:
                    raise RuntimeError("live skin temperature missing")
                if float(match[1]) <= 34:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("device did not cool")
                print(name, "cooling", match[1], flush=True)
                time.sleep(15)
            (args.output / (name + "-before.txt")).write_text(thermal)
            if step in ["profile", "stream"]:
                args_list = [
                    directory + "/run_stream",
                    "native",
                    directory + "/" + args.archive.name,
                    directory + "/" + args.tokenizer.name,
                    directory + "/pcm.txt",
                    directory + "/" + name + ".json",
                ]
            elif step == "decompression":
                args_list = [
                    directory + "/benchmark_decompression",
                    directory + "/" + args.compressed.name,
                    directory + "/" + args.archive.name,
                    directory + "/decompression.json",
                ]
            elif step == "loading":
                args_list = [
                    directory + "/benchmark_loading",
                    directory + "/" + args.archive.name,
                    directory + "/" + args.tokenizer.name,
                    directory + "/loading.json",
                ]
            elif step == "frontend":
                args_list = [
                    directory + "/benchmark_frontend",
                    directory + "/" + args.archive.name,
                    directory + "/pcm.txt",
                    str(args.rounds),
                    directory + "/" + step + ".json",
                ]
            elif step == "tiled":
                args_list = [
                    directory + "/benchmark_tiled",
                    directory + "/projections",
                    directory + "/tiled.json",
                ]
                shell("taskset 40 " + directory + "/test_projection")
                if args.tiled_order:
                    args_list.append("column-order")
                elif args.tiled_wide:
                    args_list.append("wide-tile")
            elif step in ["head", "token_selection"]:
                args_list = [
                    directory
                    + "/benchmark_"
                    + ("head_layout" if step == "head" else step),
                    directory + "/" + args.head.name,
                    directory + "/" + step + ".json",
                ]
                if step == "head" and args.head_width is not None:
                    args_list.append(str(args.head_width))
            elif step == "projection":
                args_list = [directory + "/test_projection"]
            elif step == "runtime":
                args_list = [
                    directory + "/test_runtime",
                    directory + "/" + args.archive.name,
                    directory + "/" + args.tokenizer.name,
                    directory + "/audio/" + audio[0].name,
                ]
            else:
                args_list = [
                    directory + "/benchmark_" + step,
                    directory + "/" + step + ".json",
                ]
            command = (
                "LD_LIBRARY_PATH="
                + directory
                + " TRANSCRIBRO_PROFILE="
                + ("1" if step in ["profile", "loading"] else "0")
            )
            command += " taskset " + args.cpus + " " + shlex.join(args_list)
            previous = activity()
            sampled = time.monotonic()
            with (args.output / (name + ".log")).open("w") as log:
                active = subprocess.Popen(
                    base + ["shell", command], stdout=log, stderr=subprocess.STDOUT
                )
                deadline = sampled + args.timeout
                while active.poll() is None:
                    time.sleep(5)
                    current = activity()
                    quiet(previous, current, time.monotonic() - sampled)
                    previous, sampled = current, time.monotonic()
                    if sampled > deadline:
                        raise RuntimeError("native test deadline")
                if active.returncode:
                    raise RuntimeError("native test failed: " + name)
            if step in ["projection", "runtime"]:
                (args.output / (step + ".json")).write_text(
                    json.dumps(
                        dict(
                            passed=True,
                            log_sha256=hashlib.sha256(
                                (args.output / (step + ".log")).read_bytes()
                            ).hexdigest(),
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            else:
                temporary = args.output / (name + ".json.partial")
                subprocess.run(
                    base
                    + [
                        "pull",
                        directory + "/" + name + ".json",
                        str(temporary),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                report = json.loads(temporary.read_text())
                if args.stream_batch_size:
                    combine([report], [f.name for f in recordings])
                temporary.rename(args.output / (name + ".json"))
            (args.output / (name + "-after.txt")).write_text(
                shell("dumpsys thermalservice")
            )
            print(name, "complete", flush=True)
        if args.stream_batch_size:
            reports = [
                json.loads((args.output / (name + ".json")).read_text())
                for _, name, _ in runs
            ]
            (args.output / "stream.json").write_text(
                json.dumps(combine(reports, [f.name for f in audio]), indent=2) + "\n"
            )
    finally:
        if active is not None and active.poll() is None:
            active.terminate()
        stop_ours()
        if active is not None and active.poll() is None:
            active.wait(timeout=30)
        shell("rm -rf " + directory)
        shell("test ! -e " + directory)
        (args.output / "cleanup.json").write_text(
            json.dumps(dict(fixtures_removed=True, apps_or_settings_changed=False))
            + "\n"
        )


if __name__ == "__main__":
    main()
