import numpy as np

from _contract_harness import assert_close, initializers, nodes, run_ort, stub
from _op_harness import run_case


def test_reshape_mlx_to_onnx_and_parity():
    run_case("Reshape")


def test_reshape_normalizes_wrapped_negative_one_to_int64():
    wrapped_negative_one_uint64 = 18_446_744_073_709_551_615
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [4], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [{"name": "y", "shape": [2, 2], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {
                "op": "Reshape",
                "inputs": ["x"],
                "outputs": ["y"],
                "arguments": [[wrapped_negative_one_uint64, 2]],
            }
        ],
    }

    onnx_stub = stub(payload, model_name="contract_reshape_wrapped_neg_one")
    reshape_node = nodes(onnx_stub)[0]
    assert reshape_node["op_type"] == "Reshape"
    shape_init = initializers(onnx_stub)[reshape_node["inputs"][1]]
    assert shape_init["dtype"] == "int64"
    assert shape_init["shape"] == [2]
    assert shape_init["values"] == [-1, 2]

    x = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = run_ort(payload, {"x": x}, model_name="parity_reshape_wrapped_neg_one")
    assert_close(np.reshape(x, (2, 2)), out[0])
