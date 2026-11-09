#!/usr/bin/env python3
"""Import the pinned Medium values into a fixed, graph-free tensor archive.

ONNX Runtime's Python package is a conversion dependency only. Deployment
contains neither its graph format nor its operator registry. Matrix packing
is reversible; no weight, scale, bias, or activation precision is reduced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import re

import numpy as np
from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.InferenceSession import (
    InferenceSession,
)

if not __debug__:
    raise RuntimeError("Model conversion requires validation; do not use Python -O")


DTYPES = {1: "<f4", 2: "u1", 3: "i1", 6: "<i4", 7: "<i8"}


class Source:
    def __init__(self, path, expected, branch=None):
        data = path if isinstance(path, bytes) else path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        g = InferenceSession.GetRootAsInferenceSession(data, 0).Model().Graph()
        if branch:
            node = next(
                g.Nodes(i)
                for i in range(g.NodesLength())
                if g.Nodes(i).OpType() == b"If"
            )
            g = next(
                node.Attributes(i).G()
                for i in range(node.AttributesLength())
                if node.Attributes(i).Name() == branch.encode()
            )
        self.values = {}
        self.nodes = []
        for i in range(g.InitializersLength()):
            t = g.Initializers(i)
            self.values[t.Name().decode()] = np.frombuffer(
                t.RawDataAsNumpy(), dtype=DTYPES[t.DataType()]
            ).reshape(tuple(map(int, t.DimsAsNumpy())))
        for i in range(g.NodesLength()):
            n = g.Nodes(i)
            self.nodes.append(
                (
                    n.OpType().decode(),
                    [n.Inputs(j).decode() for j in range(n.InputsLength())],
                    [n.Outputs(j).decode() for j in range(n.OutputsLength())],
                )
            )

    def producing(self, name):
        return next(n for n in self.nodes if name in n[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vad-header", type=Path, required=True)
    parser.add_argument("--packed-head", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists(), "Refuse to overwrite an archive"
    root = Path(__file__).resolve().parents[2]
    pinned = json.loads((root / "docs/moonshine-model.json").read_text())
    sources = {
        v["file"].removesuffix(".ort"): Source(args.models / v["file"], v["sha256"])
        for v in pinned["files"]
        if v["file"].endswith(".ort")
    }
    vad_data = bytes(
        int(v, 16)
        for v in re.findall(r"0x([0-9a-fA-F]{2})", args.vad_header.read_text())
    )
    vad_hash = "a620103ef77fbdae23c7798b9271506a48a883997e55f851d2884551e0947437"
    sources["silero-16k"] = Source(vad_data, vad_hash, "then_branch")
    values = {}
    provenance = {}

    def add(name, data, source=None):
        assert name not in values
        data = np.ascontiguousarray(data)
        assert data.dtype.str in DTYPES.values() or data.dtype.name in {"uint8", "int8"}
        if data.dtype.kind == "f":
            assert np.all(np.isfinite(data)), name
        values[name] = data
        if source:
            provenance[name] = source

    def copy(name, graph, key):
        add(name, sources[graph].values[key], [graph, key])

    def projection(name, graph, output):
        source = sources[graph]
        op, inputs, _ = source.producing(output)
        if op == "DynamicQuantizeMatMul":
            weight, scale, zero = inputs[1:4]
            bias = source.values[inputs[4]] if len(inputs) > 4 and inputs[4] else None
        else:
            assert op == "MatMulIntegerToFloat" and len(inputs) == 6
            weight, scale, zero = inputs[1], inputs[3], inputs[5]
            bias = None
        w = source.values[weight]
        k, n = w.shape
        assert w.dtype == np.uint8 and k % 16 == 0 and n % 16 == 0
        signed = (w.astype(np.int16) - 128).astype(np.int8)
        packed = signed.reshape(k // 8, 8, n // 16, 8, 2).transpose(2, 0, 3, 4, 1)
        restored = packed.transpose(1, 4, 0, 2, 3).reshape(k, n)
        assert np.array_equal(restored, signed)
        add(name + "/weight", packed.reshape(k, n), [graph, weight, "reversible i8mm8"])
        add(name + "/sums", signed.astype(np.int32).sum(axis=0).astype("<i4"))
        add(name + "/scale", np.broadcast_to(source.values[scale], (n,)))
        add(name + "/zero", np.broadcast_to(source.values[zero], (n,)))
        add(name + "/bias", np.zeros(n, dtype="<f4") if bias is None else bias)

    for graph, count in [("encoder", 84), ("decoder_kv", 112)]:
        prefix = "encoder" if graph == "encoder" else "decoder"
        for i in range(count):
            projection(f"{prefix}/linear/{i}", graph, "linear" + (f"_{i}" if i else ""))
        norms = 29 if graph == "encoder" else 43
        for i in range(norms):
            output = "layer_norm" + (f"_{i}" if i else "")
            _, inputs, _ = next(
                n for n in sources[graph].nodes if n[0] == "Mul" and output in n[1]
            )
            gain = next(k for k in inputs if k != output)
            copy(f"{prefix}/norm/{i}", graph, gain)
    d = sources["decoder_kv"].values
    assert np.array_equal(d["embedding.weight_quantized"].T, d["t_quantized"])
    assert sources["decoder_kv"].producing("t")[:2] == (
        "DequantizeLinear",
        ["t_quantized", "embedding.weight_scale", "embedding.weight_zero_point"],
    )
    for suffix in ["quantized", "scale", "zero_point"]:
        copy("decoder/embedding/" + suffix, "decoder_kv", "embedding.weight_" + suffix)
    if args.packed_head:
        name = "decoder/embedding/quantized"
        original = values.pop(name)
        source = provenance.pop(name)
        packed = original.reshape(512, 4, 16, 5, 128).transpose(0, 3, 1, 4, 2)
        restored = packed.transpose(0, 2, 4, 1, 3).reshape(32768, 640)
        assert np.array_equal(original, restored)
        add(
            "decoder/embedding/packed64",
            packed.reshape(512, 5, 512, 16),
            source + ["reversible packed64 uint8 codes; no new quantization"],
        )
    copy("decoder/rotary", "decoder_kv", "rotary.inv_freq")
    projection("adapter/projection", "adapter", "memory")
    for suffix in ["quantized", "scale", "zero_point"]:
        copy("adapter/positions/" + suffix, "adapter", "pos_embed.weight_" + suffix)
    for i in range(28):
        projection(f"cross/{i}", "cross_kv", "linear" + (f"_{i}" if i else ""))
    for label, key in [
        ("linear", "onnx::MatMul_128_add_tensor"),
        ("conv1", "/weight/weight.0/Mul_output_0"),
        ("conv2", "/weight/weight.0_1/Mul_output_0"),
    ]:
        for suffix in ["quantized", "scale", "zero_point"]:
            copy(f"frontend/{label}/{suffix}", "frontend.weights", key + "_" + suffix)
    for label in ["conv1", "conv2"]:
        copy(f"frontend/{label}/bias", "frontend.model", label + ".conv.bias")
    for name in [
        "stft.forward_basis_buffer",
        "decoder.decoder.2.weight",
        "decoder.decoder.2.bias",
        "decoder.rnn.weight_ih",
        "decoder.rnn.weight_hh",
        "decoder.rnn.bias_ih",
        "decoder.rnn.bias_hh",
    ] + [
        f"encoder.{i}.reparam_conv.{suffix}"
        for i in range(4)
        for suffix in ["weight", "bias"]
    ]:
        copy("vad/" + name, "silero-16k", "If_0_then_branch__Inline_0__" + name)

    # Fixed directory records, followed by 64-byte-aligned payloads. Tensor
    # names are semantic roles; no operators, shape graph or model code ships.
    header_size, record_size = 64, 160
    offset = header_size + record_size * len(values)
    entries = []
    payloads = []
    for name, data in sorted(values.items()):
        encoded_name = name.encode()
        assert len(encoded_name) < 80 and data.ndim <= 4
        kind = next(k for k, v in DTYPES.items() if np.dtype(v) == data.dtype)
        offset = (offset + 63) & ~63
        raw = data.tobytes()
        record = struct.pack(
            "<80sII4QQQ24s",
            encoded_name,
            kind,
            data.ndim,
            *(list(data.shape) + [0] * (4 - data.ndim)),
            offset,
            len(raw),
            bytes(24),
        )
        assert len(record) == record_size
        entries.append(record)
        payloads.append((offset, raw))
        offset += len(raw)
    header = struct.pack(
        "<8sIIQQ32s", b"TRMED001", 1, len(entries), offset, header_size, bytes(32)
    )
    with args.output.open("xb") as f:
        f.write(header)
        f.write(b"".join(entries))
        for position, raw in payloads:
            f.write(bytes(position - f.tell()))
            f.write(raw)
    report = {
        "format": "TRMED001",
        "source_model": pinned,
        "bytes": args.output.stat().st_size,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "projection_count": 225,
        "tensors": len(entries),
        "original_values_preserved": True,
        "tied_embedding_transpose_verified": True,
        "vad_source_sha256": vad_hash,
        "provenance": provenance,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"provenance", "source_model"}}
        )
    )


if __name__ == "__main__":
    main()
