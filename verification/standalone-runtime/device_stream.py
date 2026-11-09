#!/usr/bin/env python3
"""Silent USB-device native comparison; no application or settings changes."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import re
import subprocess
import time
import uuid


def main():
    p = argparse.ArgumentParser()
    for name in [
        "adb",
        "build",
        "archive",
        "models",
        "audio",
        "reference_libraries",
        "output",
    ]:
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--serial", required=True)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--max-start-skin", type=float)
    p.add_argument("--native-first", action="store_true")
    args = p.parse_args()
    if args.serial.startswith("emulator-") or args.rounds < 1 or args.rounds > 4:
        raise ValueError("physical device and bounded rounds required")
    args.output.mkdir(parents=True, exist_ok=False)
    device = "/data/local/tmp/transcribro-stream-" + uuid.uuid4().hex
    adb = [str(args.adb), "-s", args.serial]

    def run(*a, **kw):
        return subprocess.run(adb + list(a), check=True, text=True, **kw)

    def shell(command):
        return run("shell", command, capture_output=True).stdout

    files = [
        args.archive,
        args.build / "run_stream",
        args.build / "libtranscribro_standalone.so",
    ]
    files += [
        args.reference_libraries / name
        for name in ["libmoonshine.so", "libonnxruntime.so"]
    ]
    identity = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    pcm = sorted(args.audio.glob("*.f32"))
    if not pcm:
        raise ValueError("no supplied PCM")
    identity.update({str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in pcm})
    (args.output / "inputs.json").write_text(json.dumps(identity, indent=2) + "\n")
    try:
        run("shell", "mkdir", "-p", device + "/models", device + "/audio")
        for f in files:
            run(
                "push",
                str(f),
                device + "/",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        for name in [
            "adapter.ort",
            "cross_kv.ort",
            "decoder_kv.ort",
            "encoder.ort",
            "frontend.model.ort",
            "frontend.weights.ort",
            "streaming_config.json",
            "tokenizer.bin",
        ]:
            run(
                "push",
                str(args.models / name),
                device + "/models/",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        for f in pcm:
            run(
                "push",
                str(f),
                device + "/audio/",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        listing = args.output / "pcm.txt"
        listing.write_text("".join("audio/" + f.name + "\n" for f in pcm))
        run(
            "push",
            str(listing),
            device + "/pcm.txt",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for repeat in range(args.rounds):
            order = (
                ["reference-independent", "native"] if repeat % 2 == 0 else ["native", "reference-independent"]
            )
            if args.native_first:
                order.reverse()
            for mode in order:
                name = f"{repeat}-{mode}"
                if args.max_start_skin is not None:
                    deadline = time.monotonic() + 600
                    while True:
                        current = (
                            shell("dumpsys thermalservice")
                            .split("Current temperatures from HAL:", 1)[1]
                            .split("Current cooling devices", 1)[0]
                        )
                        match = re.search(
                            r"mValue=([0-9.]+), mType=3, mName=VIRTUAL-SKIN,", current
                        )
                        if not match:
                            raise RuntimeError("current skin temperature unavailable")
                        value = float(match[1])
                        if value <= args.max_start_skin:
                            break
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "device did not reach the requested starting temperature"
                            )
                        print(name, "waiting for cooling:", value, flush=True)
                        time.sleep(15)
                (args.output / (name + "-before.txt")).write_text(
                    shell("dumpsys thermalservice")
                )
                extra = (
                    f"{shlex.quote(args.archive.name)} models/tokenizer.bin"
                    if mode == "native"
                    else "models ./libmoonshine.so"
                )
                command = f"cd {device} && LD_LIBRARY_PATH=. ./run_stream {mode} {extra} pcm.txt {name}.json"
                with (args.output / (name + ".log")).open("w") as log:
                    run("shell", command, stdout=log, stderr=subprocess.STDOUT)
                run(
                    "pull",
                    device + "/" + name + ".json",
                    str(args.output / (name + ".json")),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                (args.output / (name + "-after.txt")).write_text(
                    shell("dumpsys thermalservice")
                )
                print(name, "complete", flush=True)
                # Leave normal Android thermal and frequency control intact.
                time.sleep(20)
        results = []
        for repeat in range(args.rounds):
            reference = json.loads(
                (args.output / f"{repeat}-reference-independent.json").read_text()
            )
            native = json.loads((args.output / f"{repeat}-native.json").read_text())
            assert len(reference["clips"]) == len(native["clips"])
            for a, b in zip(reference["clips"], native["clips"]):
                assert a["path"] == b["path"]
                results.append(
                    {
                        "round": repeat,
                        "path": a["path"],
                        "final_exact": a["text"] == b["text"],
                        "updates_exact": [
                            (u["samples"], u["text"]) for u in a["updates"]
                        ]
                        == [(u["samples"], u["text"]) for u in b["updates"]],
                    }
                )
        (args.output / "parity.json").write_text(json.dumps(results, indent=2) + "\n")
        if not all(v["final_exact"] and v["updates_exact"] for v in results):
            raise RuntimeError("device transcript mismatch")
    finally:
        run("shell", "rm", "-rf", device)
        run("shell", "test", "!", "-e", device)
        (args.output / "cleanup.json").write_text(
            json.dumps(
                {"device_directory_removed": True, "app_or_settings_changed": False}
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
