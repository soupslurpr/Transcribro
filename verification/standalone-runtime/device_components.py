#!/usr/bin/env python3
"""Run the standalone component executable over USB; remove its device files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shlex
import time
import uuid
from device_quiet import SpeechActivity


def main():
    p = argparse.ArgumentParser()
    for name in ["adb", "build", "archive", "encoder", "decoder", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--serial", required=True)
    args = p.parse_args()
    if args.serial.startswith("emulator-"):
        raise ValueError("Physical USB device required for this study")
    args.output.mkdir(parents=True, exist_ok=False)
    device = "/data/local/tmp/transcribro-standalone-" + uuid.uuid4().hex
    adb = [str(args.adb), "-s", args.serial]

    def run(*a, **kw):
        return subprocess.run(adb + list(a), check=True, text=True, **kw)

    observer = SpeechActivity(adb)

    def stop_ours():
        listing = run("shell", "ps -A -o PID,ARGS", capture_output=True).stdout
        for line in listing.splitlines()[1:]:
            fields = line.strip().split(None, 1)
            if len(fields) == 2 and fields[1].startswith(device + "/"):
                run("shell", "kill -TERM " + fields[0])

    def component(command, output):
        process = subprocess.Popen(adb + ["shell", command], stdout=output)
        deadline = time.monotonic() + 600
        try:
            while process.poll() is None:
                if observer.busy():
                    raise RuntimeError("Another speech experiment acquired the device")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Component check exceeded its deadline")
                time.sleep(5)
            if process.returncode:
                raise RuntimeError("Component executable failed")
        finally:
            if process.poll() is None:
                stop_ours()
                process.terminate()
                process.wait(timeout=30)

    artifacts = [
        args.archive,
        args.build / "run_components",
        args.build / "libtranscribro_standalone.so",
    ]
    identity = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in artifacts}
    try:
        time.sleep(30)
        if observer.busy():
            raise RuntimeError("Another speech experiment is active")
        run("shell", "mkdir", device)
        for f in artifacts:
            run(
                "push",
                str(f),
                device + "/",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        for kind in ["decoder", "encoder"]:
            run(
                "push",
                str(getattr(args, kind)),
                device + "/" + kind + "-inputs",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            name = "decoder_kv" if kind == "decoder" else kind
            with (args.output / (kind + ".json")).open("w") as f:
                command = [
                    device + "/run_components",
                    device + "/" + args.archive.name,
                    name,
                    device + "/" + kind + "-inputs",
                    device + "/" + kind + "-outputs",
                ]
                component(
                    "LD_LIBRARY_PATH="
                    + shlex.quote(device)
                    + " "
                    + shlex.join(command),
                    f,
                )
            run(
                "pull",
                device + "/" + kind + "-outputs",
                str(args.output / kind),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(kind, "complete", flush=True)
    finally:
        stop_ours()
        run("shell", "rm", "-rf", device)
        run("shell", "test", "!", "-e", device)
        (args.output / "artifacts.json").write_text(
            json.dumps(
                {
                    "device": args.serial,
                    "sha256": identity,
                    "device_files_removed": True,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
