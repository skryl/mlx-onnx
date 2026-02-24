import numpy as np
import pytest

import mlx.core as mx
import mlx_onnx as ir

from _contract_harness import assert_close, initializers, nodes, op_types, run_ort, stub
from _op_harness import run_case


@pytest.mark.parametrize("case_id", ["ArgReduceMin", "ArgReduceMax"])
def test_argreduce_mlx_to_onnx_and_parity(case_id):
    run_case(case_id)


@pytest.mark.parametrize(
    ("name", "fun", "expected_op", "expected_axis", "expected_values"),
    [
        (
            "argmax_axis1",
            lambda a: mx.argmax(a, axis=1, keepdims=False),
            "ArgMax",
            1,
            np.asarray([2, 2], dtype=np.uint32),
        ),
        (
            "argmin_axis0",
            lambda a: mx.argmin(a, axis=0, keepdims=False),
            "ArgMin",
            0,
            np.asarray([0, 0, 0], dtype=np.uint32),
        ),
    ],
)
def test_argreduce_exact_rewrite_contract_and_parity(
    name: str,
    fun,
    expected_op: str,
    expected_axis: int,
    expected_values: np.ndarray,
):
    x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)
    payload = ir.export_ir(fun, x)

    onnx_stub = stub(payload, model_name=f"contract_{name}")
    assert op_types(onnx_stub) == [expected_op, "Cast", "Squeeze"]
    lowered_nodes = nodes(onnx_stub)
    assert lowered_nodes[0]["attributes"] == {"axis": expected_axis, "keepdims": 1}
    assert lowered_nodes[1]["attributes"] == {"to": "UINT32"}
    squeeze_axes = initializers(onnx_stub)[lowered_nodes[2]["inputs"][1]]
    assert squeeze_axes["dtype"] == "int64"
    assert squeeze_axes["shape"] == [1]
    assert squeeze_axes["values"] == [expected_axis]

    feeds = {payload["inputs"][0]["name"]: np.asarray(x)}
    out = run_ort(payload, feeds, model_name=f"parity_{name}")
    assert_close(expected_values, out[0])
