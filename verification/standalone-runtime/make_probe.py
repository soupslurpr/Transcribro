#!/usr/bin/env python3
"""Export an evaluation-only ORT model exposing the native trace boundaries."""
import argparse
import json
from pathlib import Path
import onnxruntime as ort
from oracle import reconstruct


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--kind", choices=["encoder", "decoder"], required=True)
    args = p.parse_args()
    model, info = reconstruct(args.source)
    mapping = {}
    for node in model.graph.node:
        if node.op_type in {"MatMulIntegerToFloat", "DynamicQuantizeMatMul"}:
            name = node.output[0]
            i = int(name.split("_")[1]) if "_" in name else 0
            mapping[f"{args.kind}/linear/{i}/output"] = name
            if node.op_type == "DynamicQuantizeMatMul":
                mapping[f"{args.kind}/linear/{i}/input"] = node.input[0]
        if node.op_type == "Mul" and node.input[0].startswith("layer_norm"):
            name = node.input[0].removeprefix("layer_norm")
            i = int(name[1:]) if name else 0
            mapping[f"{args.kind}/norm/{i}"] = node.output[0]
    original = [value.name for value in model.graph.output]
    model.graph.output.extend(
        info[name] for name in sorted(set(mapping.values()) - set(original))
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.optimized_model_filepath = str(args.output)
    options.add_session_config_entry("session.save_model_format", "ORT")
    ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {"mapping": mapping, "outputs": [v.name for v in model.graph.output]},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
