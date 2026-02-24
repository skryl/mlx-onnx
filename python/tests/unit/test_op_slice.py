import numpy as np

from _contract_harness import assert_close, initializers, nodes, run_ort, stub
from _op_harness import run_case


def test_slice_mlx_to_onnx_and_parity():
    run_case("Slice")


def test_slice_lowers_with_exact_starts_ends_axes_steps_initializers():
    x = np.asarray(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [3, 3], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [{"name": "y", "shape": [1, 3], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {
                "op": "Slice",
                "inputs": ["x"],
                "outputs": ["y"],
                "arguments": [[1, 0], [2, 3], [1, 1]],
            }
        ],
    }
    onnx_stub = stub(payload, model_name="contract_slice_exact_inputs")
    slice_node = nodes(onnx_stub)[0]
    assert slice_node["op_type"] == "Slice"
    assert len(slice_node["inputs"]) == 5

    inits = initializers(onnx_stub)
    starts = inits[slice_node["inputs"][1]]
    ends = inits[slice_node["inputs"][2]]
    axes = inits[slice_node["inputs"][3]]
    steps = inits[slice_node["inputs"][4]]
    assert starts["dtype"] == "int64"
    assert starts["shape"] == [2]
    assert starts["values"] == [1, 0]
    assert ends["dtype"] == "int64"
    assert ends["shape"] == [2]
    assert ends["values"] == [2, 3]
    assert axes["dtype"] == "int64"
    assert axes["shape"] == [2]
    assert axes["values"] == [0, 1]
    assert steps["dtype"] == "int64"
    assert steps["shape"] == [2]
    assert steps["values"] == [1, 1]

    out = run_ort(payload, {"x": x}, model_name="parity_slice_exact_inputs")
    expected = x[1:2:1, 0:3:1]
    assert_close(expected, out[0])
