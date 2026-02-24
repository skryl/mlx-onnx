from _contract_harness import nodes, stub
from _op_cases import build_case
from _op_harness import run_case


def test_less_mlx_to_onnx_and_parity():
    run_case("Less")


def test_less_lowers_with_empty_attributes():
    case = build_case("Less")
    onnx_stub = stub(case.payload, model_name="contract_less")
    less_nodes = [node for node in nodes(onnx_stub) if node["op_type"] == "Less"]
    assert len(less_nodes) == 1
    assert less_nodes[0]["attributes"] == {}
