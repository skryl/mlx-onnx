from _contract_harness import nodes, stub
from _op_cases import build_case
from _op_harness import run_case


def test_cos_mlx_to_onnx_and_parity():
    run_case("Cos")


def test_cos_lowers_with_empty_attributes():
    case = build_case("Cos")
    onnx_stub = stub(case.payload, model_name="contract_cos")
    cos_nodes = [node for node in nodes(onnx_stub) if node["op_type"] == "Cos"]
    assert len(cos_nodes) == 1
    assert cos_nodes[0]["attributes"] == {}
