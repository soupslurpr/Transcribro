#!/usr/bin/env python3
"""Prepare checksum-locked native inputs and original Medium assets for Gradle.

Downloads and reproducible intermediates live outside build outputs. Offline
mode applies to every acquisition; conversion, wheel installation and native
configuration never consult a package index or use CMake FetchContent.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEECH = ROOT / "speech"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path, record):
    return (path.is_file() and path.stat().st_size == record["bytes"]
            and sha256(path) == record["sha256"])


def acquire(record, cache, offline):
    target = cache / "downloads" / record.get("filename", record["sha256"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if matches(target, record):
        return target
    if offline:
        raise RuntimeError(f"Missing or invalid pinned input {target.name}; run once without --offline")
    if not record["url"].startswith("https://"):
        raise RuntimeError("Pinned downloads require HTTPS")
    print("Downloading pinned input: " + target.name, flush=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
        partial = Path(output.name)
        try:
            with urllib.request.urlopen(record["url"], timeout=180) as source:
                shutil.copyfileobj(source, output)
            output.close()
            if not matches(partial, record):
                raise RuntimeError("Size or SHA-256 mismatch: " + target.name)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    return target


def unpack(archive, output, includes=None):
    output.mkdir(parents=True)
    with tarfile.open(archive) as source:
        for member in source:
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            relative = Path(*parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Unsafe native archive path")
            if includes and not any(str(relative) == p or str(relative).startswith(p + "/") for p in includes):
                continue
            # Only regular files/directories are used; do not follow source symlinks.
            destination = output / relative
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                destination.chmod(member.mode & 0o777)


def sources(args, lock):
    archives = {name: acquire(record, args.cache, args.offline)
                for name, record in lock["native"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        prepared = Path(temporary) / "sources"
        prepared.mkdir()
        for name, archive in archives.items():
            includes = None
            if name == "mlas":
                includes = ["LICENSE", "ThirdPartyNotices.txt", "cmake", "include",
                            "onnxruntime/core/mlas", "onnxruntime/core/common", "onnxruntime/core/platform"]
            elif name == "moonshine":
                includes = ["LICENSE", "core/silero-vad-model-data.h", "core/moonshine-c-api.h"]
            unpack(archive, prepared / name, includes)
        if sha256(prepared / "moonshine/core/silero-vad-model-data.h") != lock["vad"]["header_sha256"]:
            raise RuntimeError("Pinned inner VAD source changed")
        (prepared / "inputs.json").write_text(json.dumps(lock["native"], sort_keys=True, indent=2) + "\n")
        if args.output.exists():
            shutil.rmtree(args.output)
        prepared.replace(args.output)
    print("Pinned native sources prepared", flush=True)


def converter_python(args, lock):
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise RuntimeError("The locked converter currently supports Linux x86_64 with Python 3.11")
    wheels = [acquire(record, args.cache, args.offline) for record in lock["python"].values()]
    identity = hashlib.sha256(json.dumps(lock["python"], sort_keys=True).encode()).hexdigest()
    environment = args.cache / "python" / identity
    marker = environment / "complete"
    if not marker.exists():
        if environment.exists():
            shutil.rmtree(environment)
        # Explicit interpreter, no pip bootstrap or package resolution/network.
        subprocess.run(["python3.11", "-m", "venv", "--without-pip", str(environment)], check=True)
        site = environment / "lib/python3.11/site-packages"
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as source:
                source.extractall(site)
        marker.write_text(identity + "\n")
    python = environment / "bin/python"
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONHASHSEED="0")
    env.pop("PYTHONPATH", None)
    # Check that the complete transitive dependency closure is in the wheel lock.
    subprocess.run([str(python), "-c", """
from importlib.metadata import distributions, version
from packaging.requirements import Requirement
for d in distributions():
    for text in d.requires or []:
        r = Requirement(text)
        if r.marker is None or r.marker.evaluate({'extra': ''}):
            assert r.specifier.contains(version(r.name)), text
"""], env=env, check=True)
    schema = environment / "lib/python3.11/site-packages" / lock["schema"]["path"]
    if sha256(schema) != lock["schema"]["sha256"]:
        raise RuntimeError("Converter flatbuffer schema changed")
    return python, env


def zstd_program(args, lock):
    directory = args.cache / "zstd-host" / lock["native"]["zstd"]["sha256"]
    executable = directory / "programs/zstd"
    if not executable.is_file():
        subprocess.run(["cmake", "-S", str(args.sources / "zstd/build/cmake"), "-B", str(directory),
                        "-DCMAKE_BUILD_TYPE=Release", "-DZSTD_BUILD_TESTS=OFF", "-DZSTD_BUILD_SHARED=OFF",
                        "-DZSTD_BUILD_PROGRAMS=ON", "-DZSTD_MULTITHREAD_SUPPORT=ON"], check=True)
        subprocess.run(["cmake", "--build", str(directory), "--target", "zstd", "--parallel", "2"], check=True)
    return executable


def model(args, lock):
    if sha256(ROOT / lock["outer_vad"]["path"]) != lock["outer_vad"]["sha256"]:
        raise RuntimeError("Pinned outer Silero model changed")
    for name, expected in lock["converter_inputs"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError("Converter input differs from native-inputs.lock.json: " + name)
    pinned = json.loads((ROOT / "docs/moonshine-model.json").read_text())
    manifest = json.loads((SPEECH / "medium-model.json").read_text())
    identity = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    artifacts = args.cache / "models" / identity
    valid = all(matches(artifacts / f["file"], f) for f in manifest["files"])
    archive = artifacts / "medium.trm"
    valid = valid and matches(archive, {"bytes": manifest["archive_bytes"], "sha256": manifest["archive_sha256"]})
    if not valid:
        # Acquire originals even if no network is permitted. A cache miss fails here.
        originals = [(f, acquire(dict(f, url=pinned["base_url"] + "/" + f["file"],
                                      filename=f["sha256"]), args.cache, args.offline)) for f in pinned["files"]]
        python, env = converter_python(args, lock)
        zstd = zstd_program(args, lock)
        artifacts.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts.parent) as temporary:
            temporary = Path(temporary)
            originals_dir = temporary / "originals"
            originals_dir.mkdir()
            for record, path in originals:
                os.link(path, originals_dir / record["file"])
            converted = temporary / "converted"
            converted.mkdir()
            output = converted / "medium.trm"
            subprocess.run([str(python), str(SPEECH / "tools/import_model.py"),
                            "--models", str(originals_dir), "--output", str(output),
                            "--vad-header", str(args.sources / "moonshine/core/silero-vad-model-data.h"),
                            "--packed-head"], check=True, env=env)
            if not matches(output, {"bytes": manifest["archive_bytes"], "sha256": manifest["archive_sha256"]}):
                raise RuntimeError("Original Medium archive changed")
            subprocess.run([str(zstd), "-T1", "-9", str(output), "-o", str(converted / "medium.trm.zst")], check=True)
            shutil.copyfile(originals_dir / "tokenizer.bin", converted / "tokenizer.bin")
            for record in manifest["files"]:
                if not matches(converted / record["file"], record):
                    raise RuntimeError("Converted artifact differs from pinned model: " + record["file"])
            if artifacts.exists():
                shutil.rmtree(artifacts)
            converted.replace(artifacts)
    destination = args.output / "models/standalone"
    if args.output.exists():
        shutil.rmtree(args.output)
    destination.mkdir(parents=True)
    for record in manifest["files"]:
        shutil.copyfile(artifacts / record["file"], destination / record["file"])
    shutil.copyfile(SPEECH / "medium-model.json", destination / "model.json")
    print("Exact original Medium assets prepared: " + manifest["archive_sha256"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["sources", "model"])
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    if args.sources:
        args.sources = args.sources.resolve()
    if args.mode == "model" and not args.sources:
        parser.error("model preparation requires --sources")
    lock = json.loads((SPEECH / "native-inputs.lock.json").read_text())
    args.cache.mkdir(parents=True, exist_ok=True)
    # Debug/staging/release share the same persistent cache and may build together.
    with (args.cache / "prepare.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        (sources if args.mode == "sources" else model)(args, lock)


if __name__ == "__main__":
    main()
