from _contract_harness import nodes, stub
from _op_cases import build_case
from _op_harness import run_case


def test_softmax_mlx_to_onnx_and_parity():
    run_case("Softmax")


def test_softmax_lowers_with_empty_attributes():
    case = build_case("Softmax")
    onnx_stub = stub(case.payload, model_name="contract_softmax")
    softmax_nodes = [node for node in nodes(onnx_stub) if node["op_type"] == "Softmax"]
    assert len(softmax_nodes) == 1
    assert softmax_nodes[0]["attributes"] == {}
