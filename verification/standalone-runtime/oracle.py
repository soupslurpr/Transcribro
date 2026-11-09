#!/usr/bin/env python3
"""Evaluation-only reconstruction of pinned ORT graphs for tensor probes.

This module is never imported by the native runtime or shipped in the app.
"""
import hashlib
import json
from pathlib import Path
import onnx
import numpy as np
from onnx import helper, TensorProto
import onnxruntime as ort
from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.InferenceSession import (
    InferenceSession,
)
from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.TensorTypeAndShape import (
    TensorTypeAndShape,
)


def reconstruct(source):
    assert ort.__version__ == "1.23.2"
    data = source.read_bytes()
    model = InferenceSession.GetRootAsInferenceSession(data, 0).Model()
    graph = model.Graph()
    tensors = []
    identities = {}
    for i in range(graph.InitializersLength()):
        t = graph.Initializers(i)
        tensor = TensorProto(
            name=t.Name().decode(),
            data_type=t.DataType(),
            dims=list(map(int, t.DimsAsNumpy())),
        )
        tensor.raw_data = t.RawDataAsNumpy().tobytes()
        tensors.append(tensor)
        identities[tensor.name] = hashlib.sha256(tensor.raw_data).hexdigest()
    args = {}
    for i in range(graph.NodeArgsLength()):
        v = graph.NodeArgs(i)
        if v.Type() is None:
            continue
        assert v.Type().ValueType() == 1
        union = v.Type().Value()
        t = TensorTypeAndShape()
        t.Init(union.Bytes, union.Pos)
        shape = None
        if t.Shape() is not None:
            shape = []
            for j in range(t.Shape().DimLength()):
                d = t.Shape().Dim(j).Value()
                shape.append(
                    d.DimParam().decode()
                    if d.DimParam()
                    else int(d.DimValue()) if d.DimType() == 1 else None
                )
        args[v.Name().decode()] = helper.make_tensor_value_info(
            v.Name().decode(), t.ElemType(), shape
        )
    inputs = [graph.Inputs(i).decode() for i in range(graph.InputsLength())]
    outputs = [graph.Outputs(i).decode() for i in range(graph.OutputsLength())]
    base = []
    for i in range(graph.NodesLength()):
        n = graph.Nodes(i)
        attrs = {}
        for j in range(n.AttributesLength()):
            a = n.Attributes(j)
            if a.Type() == 1:
                value = a.F()
            elif a.Type() == 2:
                value = a.I()
            elif a.Type() == 3:
                value = a.S()
            elif a.Type() == 7:
                value = list(map(int, a.IntsAsNumpy()))
            else:
                raise ValueError((n.Name(), a.Name(), a.Type()))
            attrs[a.Name().decode()] = value
        base.append(
            helper.make_node(
                n.OpType().decode(),
                [n.Inputs(j).decode() for j in range(n.InputsLength())],
                [n.Outputs(j).decode() for j in range(n.OutputsLength())],
                name=n.Name().decode(),
                domain=(n.Domain() or b"").decode(),
                **attrs
            )
        )
    available = set(inputs) | set(identities) | {""}
    ordered = []
    pending = base[:]
    while pending:
        ready = [n for n in pending if set(n.input) <= available]
        if not ready:
            raise ValueError("graph has unresolved inputs")
        for n in ready:
            ordered.append(n)
            available.update(n.output)
            pending.remove(n)
    imports = [
        helper.make_opsetid(
            (model.OpsetImport(i).Domain() or b"").decode(),
            model.OpsetImport(i).Version(),
        )
        for i in range(model.OpsetImportLength())
    ]
    return (
        helper.make_model(
            helper.make_graph(
                ordered,
                "reference",
                [args[n] for n in inputs],
                [args[n] for n in outputs],
                tensors,
                value_info=[v for k, v in args.items() if k not in inputs + outputs],
            ),
            opset_imports=imports,
            ir_version=10,
        ),
        args,
    )
