from _contract_harness import nodes, stub
from _op_cases import build_case
from _op_harness import run_case


def test_erf_mlx_to_onnx_and_parity():
    run_case("Erf")


def test_erf_lowers_with_empty_attributes():
    case = build_case("Erf")
    onnx_stub = stub(case.payload, model_name="contract_erf")
    erf_nodes = [node for node in nodes(onnx_stub) if node["op_type"] == "Erf"]
    assert len(erf_nodes) == 1
    assert erf_nodes[0]["attributes"] == {}
