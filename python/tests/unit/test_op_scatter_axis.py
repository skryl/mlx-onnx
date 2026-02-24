from _op_harness import run_case


def test_scatter_axis_mlx_to_onnx_and_parity():
    run_case("ScatterAxis")
