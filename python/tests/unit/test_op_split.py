import numpy as np

import mlx.core as mx
import mlx_onnx as ir

from _contract_harness import assert_close, initializers, nodes, run_ort, stub
from _op_harness import run_case


def test_split_mlx_to_onnx_and_parity():
    run_case("Split")


def test_equal_split_lowers_with_split_lengths_initializer():
    x = mx.arange(0, 8, 1, dtype=mx.float32).reshape((4, 2))
    payload = ir.export_ir(lambda a: mx.split(a, 2, axis=0), x)

    onnx_stub = stub(payload, model_name="contract_split_equal_axis0")
    split_node = nodes(onnx_stub)[0]
    assert split_node["op_type"] == "Split"
    assert split_node["attributes"] == {"axis": 0}

    split_init = initializers(onnx_stub)[split_node["inputs"][1]]
    assert split_init["dtype"] == "int64"
    assert split_init["shape"] == [2]
    assert split_init["values"] == [2, 2]


def test_index_split_lowers_with_expected_lengths_and_runtime_parity():
    x = np.asarray(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        dtype=np.float32,
    )
    x_mx = mx.array(x, dtype=mx.float32)
    payload = ir.export_ir(lambda a: mx.split(a, [1, 3], axis=0), x_mx)

    onnx_stub = stub(payload, model_name="contract_split_indices_axis0")
    split_node = nodes(onnx_stub)[0]
    assert split_node["op_type"] == "Split"
    assert split_node["attributes"] == {"axis": 0}
    split_init = initializers(onnx_stub)[split_node["inputs"][1]]
    assert split_init["dtype"] == "int64"
    assert split_init["shape"] == [3]
    assert split_init["values"] == [1, 2, 1]

    feeds = {payload["inputs"][0]["name"]: x}
    out = run_ort(payload, feeds, model_name="parity_split_indices_axis0")
    expected = np.split(x, [1, 3], axis=0)
    assert len(out) == 3
    assert_close(expected[0], out[0])
    assert_close(expected[1], out[1])
    assert_close(expected[2], out[2])
