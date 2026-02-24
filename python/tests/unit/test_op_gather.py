import numpy as np

import mlx.core as mx
import mlx_onnx as ir

from _contract_harness import assert_close, nodes, run_ort, stub
from _op_harness import run_case


def test_gather_mlx_to_onnx_and_parity():
    run_case("Gather")


def test_take_axis1_lowers_gather_with_axis_attribute():
    x = mx.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=mx.float32)
    idx = mx.array([2, 0], dtype=mx.int32)
    payload = ir.export_ir(lambda a, b: mx.take(a, b, axis=1), x, idx)

    onnx_stub = stub(payload, model_name="contract_gather_axis1")
    gather_nodes = [node for node in nodes(onnx_stub) if node["op_type"] == "Gather"]
    assert len(gather_nodes) == 1
    assert gather_nodes[0]["attributes"] == {"axis": 1}

    feeds = {
        payload["inputs"][0]["name"]: np.asarray(x),
        payload["inputs"][1]["name"]: np.asarray(idx),
    }
    out = run_ort(payload, feeds, model_name="parity_gather_axis1")
    expected = np.asarray(mx.take(x, idx, axis=1))
    assert_close(expected, out[0])
