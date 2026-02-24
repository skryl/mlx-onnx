import numpy as np

from _contract_harness import assert_close, run_ort, stub
from _op_harness import run_case


def test_arange_mlx_to_onnx_and_parity():
    run_case("Arange")


def test_arange_with_float_arguments_is_stub_compatible():
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [],
        "keyword_inputs": [],
        "outputs": [{"name": "A", "shape": [5], "dtype": "float32"}],
        "constants": [],
        "nodes": [{"op": "Arange", "inputs": [], "outputs": ["A"], "arguments": [0.0, 1.0, 0.2]}],
    }

    onnx_stub = stub(payload, model_name="contract_arange_float_args")
    assert onnx_stub["graph"]["nodes"] == []
    assert len(onnx_stub["graph"]["initializers"]) == 1
    init = onnx_stub["graph"]["initializers"][0]
    assert init["name"] == "A"
    assert init["dtype"] == "float32"
    assert init["shape"] == [5]
    np.testing.assert_allclose(
        np.asarray(init["values"], dtype=np.float32),
        np.asarray([0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float32),
    )

    out = run_ort(payload, {}, model_name="parity_arange_float_args")
    assert len(out) == 1
    assert_close(np.asarray(init["values"], dtype=np.float32), out[0])
