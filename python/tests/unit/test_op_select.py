import numpy as np

import mlx.core as mx
import mlx_onnx as ir

from _contract_harness import assert_close, op_types, run_ort, stub
from _op_harness import run_case


def test_select_mlx_to_onnx_and_parity():
    run_case("Select")


def test_where_greater_pattern_rewrite_contract_and_parity():
    x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)
    payload = ir.export_ir(lambda a: mx.where(a > 3.0, a, mx.zeros_like(a)), x)

    onnx_stub = stub(payload, model_name="contract_where_greater_pattern")
    assert op_types(onnx_stub) == ["Expand", "Greater", "Expand", "Identity", "Where"]

    feeds = {payload["inputs"][0]["name"]: np.asarray(x)}
    out = run_ort(payload, feeds, model_name="parity_where_greater_pattern")
    expected = np.where(np.asarray(x) > 3.0, np.asarray(x), np.zeros_like(np.asarray(x)))
    assert_close(expected, out[0])
