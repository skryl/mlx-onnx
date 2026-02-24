from _contract_harness import nodes, stub
from _op_cases import build_case
from _op_harness import run_case


def test_as_type_mlx_to_onnx_and_parity():
    run_case("AsType")


def test_as_type_lowers_to_cast_with_expected_to_attribute():
    case = build_case("AsType")
    onnx_stub = stub(case.payload, model_name="contract_as_type")
    cast_nodes = [node for node in nodes(onnx_stub) if node["op_type"] == "Cast"]
    assert len(cast_nodes) == 1
    assert cast_nodes[0]["attributes"] == {"to": "FLOAT16"}
