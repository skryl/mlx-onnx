from _op_harness import run_case


def test_greater_equal_mlx_to_onnx_and_parity():
    run_case("GreaterEqual")
