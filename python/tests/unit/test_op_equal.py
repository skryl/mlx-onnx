from _contract_harness import nodes, stub
from _op_cases import build_case
from _op_harness import run_case


def test_equal_mlx_to_onnx_and_parity():
    run_case("Equal")


def test_equal_lowers_with_empty_attributes():
    case = build_case("Equal")
    onnx_stub = stub(case.payload, model_name="contract_equal")
    equal_nodes = [node for node in nodes(onnx_stub) if node["op_type"] == "Equal"]
    assert len(equal_nodes) == 1
    assert equal_nodes[0]["attributes"] == {}
