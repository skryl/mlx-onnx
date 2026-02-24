import numpy as np

import mlx.core as mx
import mlx_onnx as ir

from _contract_harness import assert_close, initializers, nodes, op_types, run_ort, stub
from _op_harness import run_case


def test_expand_dims_mlx_to_onnx_and_parity():
    run_case("ExpandDims")


def test_reshape_then_expand_dims_uses_shape_and_axes_initializers():
    x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)
    payload = ir.export_ir(
        lambda a: mx.expand_dims(mx.reshape(a, (3, 2)), axis=[0]),
        x,
    )

    onnx_stub = stub(payload, model_name="contract_reshape_expand_dims")
    assert op_types(onnx_stub) == ["Reshape", "Unsqueeze"]

    lowered_nodes = nodes(onnx_stub)
    inits = initializers(onnx_stub)
    reshape_shape = inits[lowered_nodes[0]["inputs"][1]]
    unsqueeze_axes = inits[lowered_nodes[1]["inputs"][1]]
    assert reshape_shape["dtype"] == "int64"
    assert reshape_shape["shape"] == [2]
    assert reshape_shape["values"] == [3, 2]
    assert unsqueeze_axes["dtype"] == "int64"
    assert unsqueeze_axes["shape"] == [1]
    assert unsqueeze_axes["values"] == [0]

    feeds = {payload["inputs"][0]["name"]: np.asarray(x)}
    out = run_ort(payload, feeds, model_name="parity_reshape_expand_dims")
    expected = np.expand_dims(np.reshape(np.asarray(x), (3, 2)), axis=0)
    assert_close(expected, out[0])
