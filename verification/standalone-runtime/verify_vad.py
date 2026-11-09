#!/usr/bin/env python3
"""Compare the independent fixed 16 kHz Silero path on supplied PCM."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import wave

import numpy as np
import onnxruntime as ort
from verify_components import Model


class VadState(ct.Structure):
    _fields_ = [
        ("context", ct.c_float * 64),
        ("hidden", ct.c_float * 128),
        ("cell", ct.c_float * 128),
    ]


def main():
    p = argparse.ArgumentParser()
    for name in ["library", "archive", "original", "audio", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    reference = ort.InferenceSession(args.original.read_bytes(), options)
    with wave.open(str(args.audio)) as f:
        assert (
            f.getnchannels() == 1
            and f.getframerate() == 16000
            and f.getsampwidth() == 2
        )
        audio = np.frombuffer(f.readframes(f.getnframes()), "<i2").astype("f4") / 32768
    audio = np.pad(audio, (16000, 16000))
    model = Model(args.library, args.archive)
    model.library.tm_vad.argtypes = [ct.c_void_p] * 4
    state = VadState()
    context, reference_state = np.zeros(64, "f4"), np.zeros((2, 1, 128), "f4")
    records = []
    try:
        for offset in range(0, audio.size - 511, 512):
            chunk = audio[offset : offset + 512]
            probability, reference_state = reference.run(
                None,
                {
                    "input": np.concatenate([context, chunk])[None],
                    "state": reference_state,
                    "sr": np.array(16000, "i8"),
                },
            )
            result = ct.c_float()
            status = model.library.tm_vad(
                model.handle, chunk.ctypes.data, ct.byref(state), ct.byref(result)
            )
            if status:
                raise RuntimeError(model.library.tm_error().decode())
            actual_state = np.concatenate(
                [np.array(state.hidden), np.array(state.cell)]
            ).reshape(2, 1, 128)
            context = chunk[-64:].copy()
            records.append(
                {
                    "offset": offset,
                    "reference": float(probability.item()),
                    "candidate": result.value,
                    "probability_error": abs(float(probability.item()) - result.value),
                    "state_error": float(
                        np.max(np.abs(reference_state - actual_state))
                    ),
                    "decision_equal": bool(
                        (probability.item() >= 0.5) == (result.value >= 0.5)
                    ),
                }
            )
    finally:
        model.close()
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps(
            {
                "frames": len(records),
                "max_probability_error": max(v["probability_error"] for v in records),
                "max_state_error": max(v["state_error"] for v in records),
                "decisions_equal": all(v["decision_equal"] for v in records),
            },
            indent=2,
        )
    )
    if any(v["probability_error"] or v["state_error"] for v in records):
        raise RuntimeError("VAD values or state differ from the reference")


if __name__ == "__main__":
    main()
