from _contract_harness import nodes, stub
from _op_cases import build_case
from _op_harness import run_case


def test_concatenate_mlx_to_onnx_and_parity():
    run_case("Concatenate")


def test_concatenate_axis_is_lowered_to_concat_attribute():
    case = build_case("Concatenate")
    onnx_stub = stub(case.payload, model_name="contract_concatenate_axis")
    concat_nodes = [node for node in nodes(onnx_stub) if node["op_type"] == "Concat"]
    assert len(concat_nodes) == 1
    assert concat_nodes[0]["attributes"] == {"axis": 1}
