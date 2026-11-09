#!/usr/bin/env python3
"""Locate numerical divergence without changing the deployed component API."""
import argparse
import ctypes as ct
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from oracle import reconstruct
from verify_components import Model, read_tensor


def main():
    p = argparse.ArgumentParser()
    for name in ["library", "archive", "original", "case", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--kind", choices=["encoder", "decoder_kv"], required=True)
    args = p.parse_args()
    prefix = "decoder" if args.kind == "decoder_kv" else "encoder"
    graph, info = reconstruct(args.original)
    nodes = graph.graph.node
    by_output = {name: node for node in nodes for name in node.output}
    mapping = {}
    for node in nodes:
        if node.op_type in {"MatMulIntegerToFloat", "DynamicQuantizeMatMul"}:
            output = node.output[0]
            i = int(output.split("_")[1]) if "_" in output else 0
            mapping[f"{prefix}/linear/{i}/output"] = output
            if node.op_type == "DynamicQuantizeMatMul":
                mapping[f"{prefix}/linear/{i}/input"] = node.input[0]
        if node.op_type == "Mul" and node.input[0].startswith("layer_norm"):
            name = node.input[0].removeprefix("layer_norm")
            i = int(name[1:]) if name else 0
            mapping[f"{prefix}/norm/{i}"] = node.output[0]
    names = sorted(set(mapping.values()))
    del graph.graph.output[:]
    graph.graph.output.extend(info[n] for n in names)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    reference = ort.InferenceSession(
        graph.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    input_names = (args.case / "inputs.txt").read_text().splitlines()
    inputs = {
        name: read_tensor(args.case / f"input-{i}.bin")
        for i, name in enumerate(input_names)
    }
    expected = dict(zip(names, reference.run(names, inputs), strict=True))
    model = Model(args.library, args.archive)
    callback_type = ct.CFUNCTYPE(
        None, ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_float), ct.c_size_t
    )
    actual = {}

    def collect(context, name, data, count):
        actual[name.decode()] = np.ctypeslib.as_array(data, (count,)).copy()

    callback = callback_type(collect)
    model.library.tm_set_trace.argtypes = [callback_type, ct.c_void_p]
    model.library.tm_set_trace(callback, None)
    try:
        model.run(args.kind, inputs)
    finally:
        model.library.tm_set_trace(callback_type(), None)
        model.close()
    records = []
    for name, a in actual.items():
        b = expected[mapping[name]].reshape(-1)
        diff = a.astype("f8") - b.astype("f8")
        result = {
            "name": name,
            "reference": mapping[name],
            "max_error": float(np.max(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "exact": bool(np.array_equal(a, b)),
        }
        records.append(result)
        print(name, f"{result['max_error']:.9g}", flush=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    np.savez(args.output.with_suffix(".npz"), **actual)
    np.savez(
        args.output.with_suffix(".reference.npz"),
        **{k: expected[v] for k, v in mapping.items()},
    )


if __name__ == "__main__":
    main()
