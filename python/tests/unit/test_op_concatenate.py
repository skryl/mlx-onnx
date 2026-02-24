from _op_harness import run_case


def test_concatenate_mlx_to_onnx_and_parity():
    run_case("Concatenate")
