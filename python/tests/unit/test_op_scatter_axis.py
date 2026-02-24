from _contract_harness import nodes, stub
from _op_cases import build_case
from _op_harness import run_case


def test_scatter_axis_mlx_to_onnx_and_parity():
    run_case("ScatterAxis")


def test_scatter_axis_lowers_to_scatter_elements_with_axis_attribute():
    case = build_case("ScatterAxis")
    onnx_stub = stub(case.payload, model_name="contract_scatter_axis")
    scatter_nodes = [node for node in nodes(onnx_stub) if node["op_type"] == "ScatterElements"]
    assert len(scatter_nodes) == 1
    assert scatter_nodes[0]["attributes"] == {"axis": 1}
