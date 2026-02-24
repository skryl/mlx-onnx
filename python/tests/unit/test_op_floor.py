from _contract_harness import nodes, stub
from _op_cases import build_case
from _op_harness import run_case


def test_floor_mlx_to_onnx_and_parity():
    run_case("Floor")


def test_floor_lowers_with_empty_attributes():
    case = build_case("Floor")
    onnx_stub = stub(case.payload, model_name="contract_floor")
    floor_nodes = [node for node in nodes(onnx_stub) if node["op_type"] == "Floor"]
    assert len(floor_nodes) == 1
    assert floor_nodes[0]["attributes"] == {}
