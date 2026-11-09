#!/usr/bin/env python3
"""Compare frontend features and recurrent state on silent supplied PCM."""
import argparse
import json
from pathlib import Path
import wave

import numpy as np
import onnxruntime as ort
from verify_components import Model, FrontendState


def main():
    p = argparse.ArgumentParser()
    for name in ["library", "archive", "models", "audio", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    weights = ort.InferenceSession(str(args.models / "frontend.weights.ort"), options)
    weight_values = dict(
        zip([v.name for v in weights.get_outputs()], weights.run(None, {}))
    )
    reference = ort.InferenceSession(str(args.models / "frontend.model.ort"), options)
    state = {
        "sample_buffer": np.zeros((1, 79), "f4"),
        "sample_len": np.zeros((1,), "i8"),
        "conv1_buffer": np.zeros((1, 768, 4), "f4"),
        "conv2_buffer": np.zeros((1, 1536, 4), "f4"),
        "frame_count": np.zeros((1,), "i8"),
    }
    with wave.open(str(args.audio)) as f:
        assert (
            f.getnchannels() == 1
            and f.getsampwidth() == 2
            and f.getframerate() == 16000
        )
        audio = np.frombuffer(f.readframes(f.getnframes()), "<i2").astype("f4") / 32768
    audio = np.concatenate([np.zeros(640, "f4"), audio, np.zeros(16000, "f4")])
    model = Model(args.library, args.archive)
    candidate_state = FrontendState()
    records, offset = [], 0
    try:
        sizes = [640, 1600, 2560, 5120, 1293, 3200, 16000]
        while offset < audio.size:
            size = min(sizes[len(records) % len(sizes)], audio.size - offset)
            chunk = audio[offset : offset + size]
            if size < 640:
                chunk = np.pad(chunk, (0, 640 - size))
            expected = reference.run(
                None, {"audio_chunk": chunk[None], **state, **weight_values}
            )
            features = model.frontend(chunk, candidate_state)
            actual = [
                features,
                np.array(candidate_state.sample_buffer),
                np.array([candidate_state.sample_len]),
                np.array(candidate_state.conv1),
                np.array(candidate_state.conv2),
                np.array([candidate_state.frame_count]),
            ]
            checks = []
            for a, b in zip(expected, actual, strict=True):
                a, b = a.reshape(-1), b.reshape(-1)
                assert a.size == b.size and np.all(np.isfinite(b))
                checks.append(
                    {
                        "exact": bool(np.array_equal(a, b)),
                        "max_error": float(np.max(np.abs(a.astype("f8") - b))),
                    }
                )
            state = {
                name: value for name, value in zip(state, expected[1:], strict=True)
            }
            records.append({"offset": offset, "samples": chunk.size, "checks": checks})
            print(offset, chunk.size, [v["max_error"] for v in checks], flush=True)
            offset += size
    finally:
        model.close()
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    if not all(check["exact"] for record in records for check in record["checks"]):
        raise RuntimeError("frontend values or state differ from the reference")


if __name__ == "__main__":
    main()
