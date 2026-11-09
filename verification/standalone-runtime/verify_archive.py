#!/usr/bin/env python3
"""Reject malformed archive descriptors/metadata with the sanitizer executable."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("report already exists")
    with args.archive.open("rb") as f:
        header = f.read(64)
        count = struct.unpack_from("<I", header, 12)[0]
        directory = f.read(count * 160)
    records = {}
    for i in range(count):
        record = directory[i * 160 : (i + 1) * 160]
        name = record[:80].split(b"\0")[0].decode()
        records[name] = (64 + i * 160, struct.unpack_from("<Q", record, 120)[0])
    shape = records["frontend/linear/quantized"][0]
    gain = records["encoder/norm/0"][1]
    scale = records["decoder/embedding/scale"][1]
    matrix = records["decoder/linear/0/weight"][0]
    packed_embedding = "decoder/embedding/packed64" in records
    embedding = records[
        (
            "decoder/embedding/packed64"
            if packed_embedding
            else "decoder/embedding/quantized"
        )
    ][0]
    mutations = [
        ("magic", 0, b"INVALID!"),
        ("version", 8, struct.pack("<I", 2)),
        ("directory_count", 12, struct.pack("<I", 0xFFFFFFFF)),
        ("file_length", 16, struct.pack("<Q", 64)),
        ("directory_offset", 24, struct.pack("<Q", 128)),
        ("tensor_type", shape + 80, struct.pack("<I", 99)),
        ("tensor_rank", shape + 84, struct.pack("<I", 5)),
        ("shape_overflow", shape + 88, struct.pack("<Q", 0xFFFFFFFFFFFFFFFF)),
        ("same_size_wrong_shape", shape + 88, struct.pack("<QQ", 768, 80)),
        ("wrong_matrix_shape", matrix + 88, struct.pack("<QQ", 320, 1280)),
        ("overlapping_payload", shape + 120, struct.pack("<Q", 64)),
        (
            "unaligned_payload",
            shape + 120,
            struct.pack("<Q", records["frontend/linear/quantized"][1] + 1),
        ),
        ("oversized_payload", shape + 128, struct.pack("<Q", 0xFFFFFFFFFFFFFFFF)),
        ("missing_tensor", shape, b"absent" + bytes(74)),
        ("nonfinite_gain", gain, struct.pack("<f", float("nan"))),
        ("negative_scale", scale, struct.pack("<f", -1)),
        ("duplicate_name", 64 + 160, directory[:80]),
        (
            "wrong_embedding_layout",
            embedding + 88,
            (
                struct.pack("<4Q", 512, 5, 256, 32)
                if packed_embedding
                else struct.pack("<2Q", 640, 32768)
            ),
        ),
        ("signed_embedding_codes", embedding + 80, struct.pack("<I", 3)),
        (
            "unknown_embedding_layout",
            embedding,
            b"decoder/embedding/packed65".ljust(80, b"\0"),
        ),
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="transcribro-invalid-") as tmp:
        path = Path(tmp) / "model.trm"
        shutil.copyfile(args.archive, path)
        with path.open("r+b", buffering=0) as f:
            for name, at, replacement in mutations:
                original = os.pread(f.fileno(), len(replacement), at)
                os.pwrite(f.fileno(), replacement, at)
                try:
                    run = subprocess.run(
                        [str(args.runner.resolve()), str(path), "--reject-archive"],
                        capture_output=True,
                        text=True,
                        env={
                            **os.environ,
                            "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
                            "UBSAN_OPTIONS": "halt_on_error=1",
                        },
                    )
                    if run.returncode or run.stderr:
                        raise RuntimeError(
                            (name, run.returncode, run.stdout, run.stderr)
                        )
                    results.append(
                        {"case": name, "rejected": True, "reason": run.stdout.strip()}
                    )
                finally:
                    os.pwrite(f.fileno(), original, at)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(len(results), "malformed archives rejected without sanitizer diagnostics")


if __name__ == "__main__":
    main()
