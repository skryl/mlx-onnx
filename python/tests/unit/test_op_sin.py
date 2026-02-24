from _contract_harness import nodes, stub
from _op_cases import build_case
from _op_harness import run_case


def test_sin_mlx_to_onnx_and_parity():
    run_case("Sin")


def test_sin_lowers_with_empty_attributes():
    case = build_case("Sin")
    onnx_stub = stub(case.payload, model_name="contract_sin")
    sin_nodes = [node for node in nodes(onnx_stub) if node["op_type"] == "Sin"]
    assert len(sin_nodes) == 1
    assert sin_nodes[0]["attributes"] == {}
