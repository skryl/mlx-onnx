import json

import numpy as np

import mlx.core as mx
import mlx_onnx as ir

from _contract_harness import assert_close, nodes, run_ort, stub
from _op_harness import run_case


def test_flatten_mlx_to_onnx_and_parity():
    run_case("Flatten")


def test_flatten_after_add_static_shape_is_supported_for_stub_conversion():
    x = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    y = mx.array([[0.5, 1.5], [2.5, 3.5]], dtype=mx.float32)
    payload = ir.export_ir(lambda a, b: mx.flatten(a + b, start_axis=0, end_axis=1), x, y)

    report = json.loads(ir.ir_compatibility_report_json(payload))
    assert report["unsupported_nodes"] == 0
    assert report["unsupported_ops"] == []
    assert report["ready_for_stub_conversion"] is True

    onnx_stub = stub(payload, model_name="contract_flatten_after_add")
    assert [node["op_type"] for node in nodes(onnx_stub)] == ["Add", "Reshape"]

    feeds = {
        payload["inputs"][0]["name"]: np.asarray(x),
        payload["inputs"][1]["name"]: np.asarray(y),
    }
    out = run_ort(payload, feeds, model_name="parity_flatten_after_add")
    expected = np.asarray(mx.flatten(x + y, start_axis=0, end_axis=1))
    assert_close(expected, out[0])
