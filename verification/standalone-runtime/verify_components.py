#!/usr/bin/env python3
"""Compare independent native components against pinned real ORT inputs."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np


class FrontendState(ct.Structure):
    _fields_ = [
        ("sample_buffer", ct.c_float * 79),
        ("sample_len", ct.c_int64),
        ("conv1", ct.c_float * (768 * 4)),
        ("conv2", ct.c_float * (1536 * 4)),
        ("frame_count", ct.c_int64),
    ]


def read_tensor(path):
    raw = path.read_bytes()
    assert raw[:8] == b"TRINPUT1"
    kind, rank = struct.unpack_from("<II", raw, 8)
    shape = struct.unpack_from("<" + "q" * rank, raw, 16)
    return (
        np.frombuffer(raw, dtype={1: "<f4", 7: "<i8"}[kind], offset=16 + 8 * rank)
        .reshape(shape)
        .copy()
    )


class Model:
    def __init__(self, library, archive):
        self.library = ct.CDLL(str(library.resolve()))
        self.library.tm_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_size_t]
        self.library.tm_create.restype = ct.c_void_p
        self.library.tm_destroy.argtypes = [ct.c_void_p]
        self.library.tm_error.restype = ct.c_char_p
        p, n = ct.c_void_p, ct.c_size_t
        self.library.tm_encoder.argtypes = [p, p, n, p]
        self.library.tm_adapter.argtypes = [p, p, n, n, p]
        self.library.tm_cross.argtypes = [p, p, n, p, p]
        self.library.tm_decoder.argtypes = [p, p, n, p, p, n, p, p, n, p, p, p]
        self.library.tm_frontend.argtypes = [p, p, n, p, p, n, p]
        error = ct.create_string_buffer(1024)
        self.handle = self.library.tm_create(
            str(archive.resolve()).encode(), error, len(error)
        )
        if not self.handle:
            raise RuntimeError(error.value.decode())

    def close(self):
        if self.handle:
            self.library.tm_destroy(self.handle)
            self.handle = None

    def run(self, kind, inputs):
        def ptr(array):
            assert array.flags.c_contiguous
            return array.ctypes.data

        if kind == "encoder":
            x = inputs["features"]
            outputs = [np.empty_like(x)]
            status = self.library.tm_encoder(
                self.handle, ptr(x), x.shape[1], ptr(outputs[0])
            )
        elif kind == "adapter":
            x = inputs["encoded"]
            outputs = [np.empty((1, x.shape[1], 640), dtype="f4")]
            status = self.library.tm_adapter(
                self.handle,
                ptr(x),
                x.shape[1],
                int(inputs["pos_offset"][0]),
                ptr(outputs[0]),
            )
        elif kind == "cross_kv":
            x = inputs["memory"]
            outputs = [
                np.empty((14, 1, 10, x.shape[1], 64), dtype="f4") for _ in range(2)
            ]
            status = self.library.tm_cross(
                self.handle, ptr(x), x.shape[1], *map(ptr, outputs)
            )
        elif kind == "decoder_kv":
            tokens = inputs["token"]
            rows, past, frames = (
                tokens.shape[1],
                inputs["k_self"].shape[3],
                inputs["out_k_cross"].shape[3],
            )
            outputs = [np.empty((1, rows, 32768), dtype="f4")]
            outputs += [
                np.empty((14, 1, 10, past + rows, 64), dtype="f4") for _ in range(2)
            ]
            status = self.library.tm_decoder(
                self.handle,
                ptr(tokens),
                rows,
                ptr(inputs["k_self"]),
                ptr(inputs["v_self"]),
                past,
                ptr(inputs["out_k_cross"]),
                ptr(inputs["out_v_cross"]),
                frames,
                *map(ptr, outputs),
            )
            outputs += [inputs["out_k_cross"], inputs["out_v_cross"]]
        else:
            raise ValueError(kind)
        if status:
            raise RuntimeError(self.library.tm_error().decode())
        return outputs

    def frontend(self, audio, state):
        assert audio.dtype == np.float32 and audio.flags.c_contiguous
        features = np.empty(
            ((audio.size + state.sample_len + 319) // 320, 768), dtype="f4"
        )
        count = ct.c_size_t()
        status = self.library.tm_frontend(
            self.handle,
            audio.ctypes.data,
            audio.size,
            ct.byref(state),
            features.ctypes.data,
            features.shape[0],
            ct.byref(count),
        )
        if status:
            raise RuntimeError(self.library.tm_error().decode())
        return features[: count.value].reshape(1, count.value, 768)


def main():
    import onnxruntime as ort

    p = argparse.ArgumentParser()
    for name in ["library", "archive", "original", "inputs", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--kind",
        choices=["encoder", "decoder_kv", "adapter", "cross_kv"],
        required=True,
    )
    args = p.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    reference = ort.InferenceSession(
        str(args.original), options, providers=["CPUExecutionProvider"]
    )
    started = time.monotonic()
    model = Model(args.library, args.archive)
    load_ms = (time.monotonic() - started) * 1000
    records = []
    try:
        for directory in sorted(args.inputs.iterdir(), key=lambda d: int(d.name)):
            names = (directory / "inputs.txt").read_text().splitlines()
            inputs = {
                name: read_tensor(directory / f"input-{i}.bin")
                for i, name in enumerate(names)
            }
            expected = reference.run(None, inputs)
            started = time.monotonic()
            actual = model.run(args.kind, inputs)
            elapsed = (time.monotonic() - started) * 1000
            checks = []
            for a, b in zip(expected, actual, strict=True):
                assert a.shape == b.shape and np.all(np.isfinite(b))
                difference = a.astype("f8") - b.astype("f8")
                checks.append(
                    {
                        "shape": list(a.shape),
                        "exact": bool(np.array_equal(a.view("u4"), b.view("u4"))),
                        "max_error": float(np.max(np.abs(difference), initial=0)),
                        "rmse": (
                            float(np.sqrt(np.mean(difference * difference)))
                            if a.size
                            else 0
                        ),
                    }
                )
            greedy_equal = (
                bool(np.array_equal(expected[0].argmax(-1), actual[0].argmax(-1)))
                if args.kind == "decoder_kv"
                else None
            )
            record = {
                "case": directory.name,
                "milliseconds": elapsed,
                "checks": checks,
                "greedy_equal": greedy_equal,
            }
            records.append(record)
            print(
                directory.name,
                [round(x["max_error"], 7) for x in checks],
                "greedy",
                greedy_equal,
                "ms",
                round(elapsed, 2),
                flush=True,
            )
    finally:
        model.close()
    args.output.write_text(
        json.dumps(
            {
                "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
                "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                "ort": ort.__version__,
                "load_ms": load_ms,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    if not all(check["exact"] for record in records for check in record["checks"]):
        raise RuntimeError("component outputs differ from the pinned reference")


if __name__ == "__main__":
    main()
