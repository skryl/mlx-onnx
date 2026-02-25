import json

import mlx_onnx as ir


def _single_node_payload(node, inputs, outputs):
    return {
        "ir_version": 1,
        "shapeless": False,
        "inputs": inputs,
        "keyword_inputs": [],
        "outputs": outputs,
        "constants": [],
        "nodes": [node],
    }


def _lowered_op_types(payload, model_name):
    stub = json.loads(ir.ir_to_onnx_json(payload, opset=18, model_name=model_name))
    return [node["op_type"] for node in stub["graph"]["nodes"]]


def _assert_supported(payload, expected_op):
    report = json.loads(ir.ir_compatibility_report_json(payload))
    assert report["unsupported_nodes"] == 0
    assert report["ready_for_stub_conversion"] is True
    assert len(report["nodes"]) == 1
    assert report["nodes"][0]["op"] == expected_op
    assert report["nodes"][0]["supported"] is True


def test_slice_update_mapping_supported_and_lowered():
    payload = _single_node_payload(
        node={
            "op": "SliceUpdate",
            "inputs": ["x", "u"],
            "outputs": ["y"],
            "arguments": [[1], [3], [1]],
        },
        inputs=[
            {"name": "x", "shape": [4], "dtype": "float32"},
            {"name": "u", "shape": [2], "dtype": "float32"},
        ],
        outputs=[{"name": "y", "shape": [4], "dtype": "float32"}],
    )

    _assert_supported(payload, "SliceUpdate")
    op_types = _lowered_op_types(payload, model_name="unit_slice_update")
    assert "ScatterND" in op_types


def test_layernorm_mapping_supported_and_lowered():
    payload = _single_node_payload(
        node={
            "op": "LayerNorm",
            "inputs": ["x", "w", "b"],
            "outputs": ["y"],
            "arguments": [1e-5],
        },
        inputs=[
            {"name": "x", "shape": [2, 4], "dtype": "float32"},
            {"name": "w", "shape": [4], "dtype": "float32"},
            {"name": "b", "shape": [4], "dtype": "float32"},
        ],
        outputs=[{"name": "y", "shape": [2, 4], "dtype": "float32"}],
    )

    _assert_supported(payload, "LayerNorm")
    op_types = _lowered_op_types(payload, model_name="unit_layernorm")
    assert "ReduceMean" in op_types
    assert "Sqrt" in op_types


def test_rope_mapping_supported_and_lowered():
    payload = _single_node_payload(
        node={
            "op": "RoPE",
            "inputs": ["x", "offset"],
            "outputs": ["y"],
            "arguments": [4, True, 10000.0, 1.0, True],
        },
        inputs=[
            {"name": "x", "shape": [1, 1, 4, 8], "dtype": "float32"},
            {"name": "offset", "shape": [], "dtype": "int32"},
        ],
        outputs=[{"name": "y", "shape": [1, 1, 4, 8], "dtype": "float32"}],
    )

    _assert_supported(payload, "RoPE")
    op_types = _lowered_op_types(payload, model_name="unit_rope")
    assert "Sin" in op_types
    assert "Cos" in op_types


def test_argpartition_mapping_supported_and_lowered():
    payload = _single_node_payload(
        node={
            "op": "ArgPartition",
            "inputs": ["x"],
            "outputs": ["y"],
            "arguments": [1, 1],
        },
        inputs=[
            {"name": "x", "shape": [2, 4], "dtype": "float32"},
        ],
        outputs=[{"name": "y", "shape": [2, 2], "dtype": "uint32"}],
    )

    _assert_supported(payload, "ArgPartition")
    op_types = _lowered_op_types(payload, model_name="unit_argpartition")
    assert "TopK" in op_types
    assert "Cast" in op_types


def test_gathermm_mapping_supported_and_lowered():
    payload = _single_node_payload(
        node={
            "op": "GatherMM",
            "inputs": ["a", "b", "lhs", "rhs"],
            "outputs": ["y"],
            "arguments": [False, False],
        },
        inputs=[
            {"name": "a", "shape": [4, 1, 3], "dtype": "float32"},
            {"name": "b", "shape": [2, 3, 2], "dtype": "float32"},
            {"name": "lhs", "shape": [4], "dtype": "int64"},
            {"name": "rhs", "shape": [4], "dtype": "int64"},
        ],
        outputs=[{"name": "y", "shape": [4, 1, 2], "dtype": "float32"}],
    )

    _assert_supported(payload, "GatherMM")
    op_types = _lowered_op_types(payload, model_name="unit_gathermm")
    assert "Reshape" in op_types
    assert "Gather" in op_types
    assert "MatMul" in op_types
