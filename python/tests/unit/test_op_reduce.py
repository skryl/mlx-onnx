import numpy as np
import pytest

import mlx.core as mx
import mlx_onnx as ir

from _contract_harness import assert_close, initializers, nodes, op_types, run_ort, stub
from _op_harness import run_case


@pytest.mark.parametrize("case_id", ["ReduceSum", "ReduceProd", "ReduceMin", "ReduceMax"])
def test_reduce_mlx_to_onnx_and_parity(case_id):
    run_case(case_id)


@pytest.mark.parametrize(
    ("reduce_name", "reduce_op", "numpy_reduce"),
    [
        ("all", "ReduceMin", np.all),
        ("any", "ReduceMax", np.any),
    ],
)
def test_boolean_reduce_lowers_with_cast_reduce_and_squeeze_contract(
    reduce_name: str,
    reduce_op: str,
    numpy_reduce,
):
    x = mx.array([[True, False, True], [True, True, True]], dtype=mx.bool_)
    if reduce_name == "all":
        payload = ir.export_ir(lambda a: mx.all(a, axis=1, keepdims=False), x)
    else:
        payload = ir.export_ir(lambda a: mx.any(a, axis=1, keepdims=False), x)

    onnx_stub = stub(payload, model_name=f"contract_reduce_bool_{reduce_name}")
    assert op_types(onnx_stub) == ["Cast", "Cast", reduce_op, "Cast", "Squeeze"]
    lowered_nodes = nodes(onnx_stub)
    assert lowered_nodes[0]["attributes"] == {"to": "BOOL"}
    assert lowered_nodes[1]["attributes"] == {"to": "INT64"}
    assert lowered_nodes[2]["attributes"] == {"keepdims": 1}
    assert lowered_nodes[3]["attributes"] == {"to": "BOOL"}

    inits = initializers(onnx_stub)
    reduce_axes = inits[lowered_nodes[2]["inputs"][1]]
    squeeze_axes = inits[lowered_nodes[4]["inputs"][1]]
    assert reduce_axes["dtype"] == "int64"
    assert reduce_axes["shape"] == [1]
    assert reduce_axes["values"] == [1]
    assert squeeze_axes["dtype"] == "int64"
    assert squeeze_axes["shape"] == [1]
    assert squeeze_axes["values"] == [1]

    feeds = {payload["inputs"][0]["name"]: np.asarray(x)}
    out = run_ort(payload, feeds, model_name=f"parity_reduce_bool_{reduce_name}")
    expected = numpy_reduce(np.asarray(x), axis=1)
    assert_close(expected, out[0])


def test_mean_axis1_lowers_reduce_squeeze_expand_mul_contract_and_parity():
    x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)
    payload = ir.export_ir(lambda a: mx.mean(a, axis=1), x)

    onnx_stub = stub(payload, model_name="contract_reduce_mean_axis1")
    assert op_types(onnx_stub) == ["ReduceSum", "Squeeze", "Expand", "Mul"]

    lowered_nodes = nodes(onnx_stub)
    inits = initializers(onnx_stub)
    reduce = lowered_nodes[0]
    squeeze = lowered_nodes[1]
    expand = lowered_nodes[2]
    assert reduce["attributes"] == {"keepdims": 1}

    reduce_axes = inits[reduce["inputs"][1]]
    squeeze_axes = inits[squeeze["inputs"][1]]
    expand_shape = inits[expand["inputs"][1]]
    assert reduce_axes["dtype"] == "int64"
    assert reduce_axes["shape"] == [1]
    assert reduce_axes["values"] == [1]
    assert squeeze_axes["dtype"] == "int64"
    assert squeeze_axes["shape"] == [1]
    assert squeeze_axes["values"] == [1]
    assert expand_shape["dtype"] == "int64"
    assert expand_shape["shape"] == [1]
    assert expand_shape["values"] == [2]

    feeds = {payload["inputs"][0]["name"]: np.asarray(x)}
    out = run_ort(payload, feeds, model_name="parity_reduce_mean_axis1")
    expected = np.mean(np.asarray(x), axis=1)
    assert_close(expected, out[0])
