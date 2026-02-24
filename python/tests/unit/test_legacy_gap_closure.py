from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

import mlx.core as mx
import mlx_onnx as ir


def _stub(payload: dict, model_name: str) -> dict:
    return json.loads(ir.ir_to_onnx_json(payload, opset=18, model_name=model_name))


def _run_ort(payload: dict, feeds: dict[str, np.ndarray], model_name: str) -> list[np.ndarray]:
    ort = pytest.importorskip("onnxruntime")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, f"{model_name}.onnx")
        ir.ir_to_onnx(path, payload, opset=18, model_name=model_name)
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        runtime_feeds = {
            input_info.name: np.asarray(feeds[input_info.name])
            for input_info in session.get_inputs()
        }
        return [np.asarray(out) for out in session.run(None, runtime_feeds)]


def _assert_close(expected: np.ndarray, actual: np.ndarray, *, atol: float = 1e-5):
    if expected.dtype == np.bool_ or np.issubdtype(expected.dtype, np.integer):
        np.testing.assert_array_equal(expected, actual)
    else:
        np.testing.assert_allclose(expected, actual, rtol=1e-4, atol=atol)


def test_shapeless_dynamic_arity_multiple_outputs_contract_and_parity():
    def fun(a, b, c):
        total = a + b + c
        return total, mx.sum(total, axis=1, keepdims=True)

    x = mx.arange(0, 6, dtype=mx.float32).reshape((2, 3))
    y = mx.ones((2, 3), dtype=mx.float32)
    z = mx.full((2, 3), 2.0, dtype=mx.float32)
    payload = ir.export_ir(fun, x, y, z, shapeless=True)
    assert payload["shapeless"] is True
    assert len(payload["inputs"]) == 3
    assert len(payload["outputs"]) == 2
    assert [node["op"] for node in payload["nodes"]] == ["Add", "Add", "Reduce"]

    stub = _stub(payload, model_name="legacy_shapeless_dynamic_arity")
    assert [node["op_type"] for node in stub["graph"]["nodes"]] == ["Add", "Add", "ReduceSum"]

    feeds = {
        payload["inputs"][0]["name"]: np.asarray(x),
        payload["inputs"][1]["name"]: np.asarray(y),
        payload["inputs"][2]["name"]: np.asarray(z),
    }
    out = _run_ort(payload, feeds, model_name="legacy_shapeless_dynamic_arity")
    total = np.asarray(x) + np.asarray(y) + np.asarray(z)
    _assert_close(total, out[0])
    _assert_close(np.sum(total, axis=1, keepdims=True), out[1])


def test_shapeless_add_reduce_nodes_stable_across_shapes_and_parity():
    def fun(a, b):
        return mx.sum(a + b, axis=1, keepdims=True)

    x1 = mx.arange(0, 6, dtype=mx.float32).reshape((2, 3))
    y1 = mx.ones((2, 3), dtype=mx.float32)
    x2 = mx.arange(0, 12, dtype=mx.float32).reshape((4, 3))
    y2 = mx.ones((4, 3), dtype=mx.float32)

    payload1 = ir.export_ir(fun, x1, y1, shapeless=True)
    payload2 = ir.export_ir(fun, x2, y2, shapeless=True)
    assert payload1["shapeless"] is True
    assert payload2["shapeless"] is True
    assert payload1["nodes"] == payload2["nodes"]

    out1 = _run_ort(
        payload1,
        {
            payload1["inputs"][0]["name"]: np.asarray(x1),
            payload1["inputs"][1]["name"]: np.asarray(y1),
        },
        model_name="legacy_shapeless_baseline_1",
    )
    out2 = _run_ort(
        payload2,
        {
            payload2["inputs"][0]["name"]: np.asarray(x2),
            payload2["inputs"][1]["name"]: np.asarray(y2),
        },
        model_name="legacy_shapeless_baseline_2",
    )
    _assert_close(np.sum(np.asarray(x1) + np.asarray(y1), axis=1, keepdims=True), out1[0])
    _assert_close(np.sum(np.asarray(x2) + np.asarray(y2), axis=1, keepdims=True), out2[0])


def test_shapeless_ops_reshape_gatheraxis_full_like_matmul_contracts():
    x = mx.arange(0, 6, dtype=mx.float32).reshape((2, 3))
    idx = mx.array([[0, 1, 2], [2, 1, 0]], dtype=mx.int32)
    payload = ir.export_ir(
        lambda a, b: mx.reshape(mx.take_along_axis(a, b, axis=1), (6,)),
        x,
        idx,
        shapeless=True,
    )
    assert payload["shapeless"] is True
    assert [node["op"] for node in payload["nodes"]] == ["GatherAxis", "Reshape"]
    stub = _stub(payload, model_name="legacy_shapeless_gatheraxis_reshape")
    assert [node["op_type"] for node in stub["graph"]["nodes"]] == ["GatherElements", "Reshape"]

    full_like_payload = ir.export_ir(
        lambda a: mx.broadcast_to(mx.full((1, 1), 2.0, dtype=mx.float32), a.shape),
        x,
        shapeless=True,
    )
    assert full_like_payload["shapeless"] is True
    assert [node["op"] for node in full_like_payload["nodes"]] == ["Broadcast", "Full", "Broadcast"]
    full_like_stub = _stub(full_like_payload, model_name="legacy_shapeless_full_like")
    assert [node["op_type"] for node in full_like_stub["graph"]["nodes"]] == ["Expand", "Identity", "Expand"]

    m = mx.arange(0, 8, dtype=mx.float32).reshape((2, 4))
    n = mx.arange(0, 12, dtype=mx.float32).reshape((4, 3))
    matmul_payload = ir.export_ir(lambda a, b: a @ b, m, n, shapeless=True)
    assert matmul_payload["shapeless"] is True
    assert [node["op"] for node in matmul_payload["nodes"]] == ["Matmul"]
    matmul_stub = _stub(matmul_payload, model_name="legacy_shapeless_matmul")
    assert [node["op_type"] for node in matmul_stub["graph"]["nodes"]] == ["MatMul"]


def test_shapeless_sliceupdate_manual_payload_parity():
    payload = {
        "ir_version": 1,
        "shapeless": True,
        "inputs": [
            {"name": "x", "shape": [4], "dtype": "float32"},
            {"name": "u", "shape": [2], "dtype": "float32"},
        ],
        "keyword_inputs": [],
        "outputs": [{"name": "y", "shape": [4], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {
                "op": "SliceUpdate",
                "inputs": ["x", "u"],
                "outputs": ["y"],
                "arguments": [[1], [3], [1]],
            }
        ],
    }
    stub = _stub(payload, model_name="legacy_shapeless_sliceupdate")
    assert [node["op_type"] for node in stub["graph"]["nodes"]] == ["Reshape", "ScatterND"]
    out = _run_ort(
        payload,
        {
            "x": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "u": np.asarray([9.0, 8.0], dtype=np.float32),
        },
        model_name="legacy_shapeless_sliceupdate",
    )
    _assert_close(np.asarray([1.0, 9.0, 8.0, 4.0], dtype=np.float32), out[0])


def test_complex64_initializer_marker_and_literal_string_are_accepted():
    onnx = pytest.importorskip("onnx")
    marker_payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [2], "dtype": "complex64"}],
        "keyword_inputs": [],
        "outputs": [{"name": "y", "shape": [2], "dtype": "complex64"}],
        "constants": [
            {
                "name": "c",
                "shape": [2],
                "dtype": "complex64",
                "values": [
                    {"__mlx_complex__": [1.0, -2.0]},
                    {"__mlx_complex__": [0.5, 0.25]},
                ],
            }
        ],
        "nodes": [{"op": "Add", "inputs": ["x", "c"], "outputs": ["y"]}],
    }
    string_payload = {
        **marker_payload,
        "constants": [
            {
                "name": "c",
                "shape": [2],
                "dtype": "complex64",
                "values": ["1-2i", "0.5+0.25i"],
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        marker_path = os.path.join(tmp_dir, "marker.onnx")
        string_path = os.path.join(tmp_dir, "string.onnx")
        ir.ir_to_onnx(marker_path, marker_payload, model_name="legacy_complex_marker")
        ir.ir_to_onnx(string_path, string_payload, model_name="legacy_complex_string")
        marker_model = onnx.load(marker_path)
        string_model = onnx.load(string_path)
        marker_init = marker_model.graph.initializer[0]
        string_init = string_model.graph.initializer[0]
        assert marker_init.data_type == onnx.TensorProto.COMPLEX64
        assert string_init.data_type == onnx.TensorProto.COMPLEX64
        assert list(marker_init.float_data) == [1.0, -2.0, 0.5, 0.25]
        assert list(string_init.float_data) == [1.0, -2.0, 0.5, 0.25]


def test_complex64_initializer_rejects_invalid_marker_and_literal():
    base = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [1], "dtype": "complex64"}],
        "keyword_inputs": [],
        "outputs": [{"name": "y", "shape": [1], "dtype": "complex64"}],
        "nodes": [{"op": "Add", "inputs": ["x", "c"], "outputs": ["y"]}],
    }
    invalid_marker = {
        **base,
        "constants": [
            {
                "name": "c",
                "shape": [1],
                "dtype": "complex64",
                "values": [{"__mlx_complex__": [1.0]}],
            }
        ],
    }
    invalid_literal = {
        **base,
        "constants": [
            {"name": "c", "shape": [1], "dtype": "complex64", "values": ["bad_literal"]}
        ],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(RuntimeError, match="invalid complex marker"):
            ir.ir_to_onnx(
                os.path.join(tmp_dir, "bad_marker.onnx"),
                invalid_marker,
                model_name="legacy_complex_bad_marker",
            )
        with pytest.raises(RuntimeError, match="unsupported complex literal"):
            ir.ir_to_onnx(
                os.path.join(tmp_dir, "bad_literal.onnx"),
                invalid_literal,
                model_name="legacy_complex_bad_literal",
            )


def test_export_ir_captures_astype_target_dtype_and_legacy_fallback_behavior():
    x = mx.array([1.25, 2.5], dtype=mx.float32)
    payload = ir.export_ir(lambda a: a.astype(mx.int32), x)
    assert payload["nodes"][0]["op"] == "AsType"
    assert payload["nodes"][0]["arguments"] == ["int32"]

    missing_arg_payload = json.loads(json.dumps(payload))
    missing_arg_payload["nodes"][0]["arguments"] = []
    stub = _stub(missing_arg_payload, model_name="legacy_astype_missing_arg")
    cast = stub["graph"]["nodes"][0]
    assert cast["op_type"] == "Cast"
    assert cast["attributes"]["to"] == "INT32"

    invalid_arg_payload = json.loads(json.dumps(payload))
    invalid_arg_payload["nodes"][0]["arguments"] = [123]
    with pytest.raises(NotImplementedError, match="unsupported AsType arguments"):
        ir.ir_to_onnx_json(invalid_arg_payload, opset=18, model_name="legacy_astype_invalid")


def test_mean_decomposition_reduce_expand_mul_contract_and_parity():
    x = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    payload = ir.export_ir(lambda a: mx.mean(a, axis=1, keepdims=True), x)
    stub = _stub(payload, model_name="legacy_mean_contract")
    assert [node["op_type"] for node in stub["graph"]["nodes"]] == ["ReduceSum", "Expand", "Mul"]

    reduce = stub["graph"]["nodes"][0]
    assert reduce["attributes"] == {"keepdims": 1}
    initializers = {item["name"]: item for item in stub["graph"]["initializers"]}
    assert initializers[reduce["inputs"][1]]["dtype"] == "int64"
    assert initializers[reduce["inputs"][1]]["values"] == [1]
    assert initializers["C"]["values"] == 0.5

    out = _run_ort(
        payload,
        {payload["inputs"][0]["name"]: np.asarray(x)},
        model_name="legacy_mean_contract",
    )
    _assert_close(np.mean(np.asarray(x), axis=1, keepdims=True), out[0])


def test_arange_float_arguments_and_reshape_int64_normalization_regression():
    arange_payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [],
        "keyword_inputs": [],
        "outputs": [{"name": "A", "shape": [5], "dtype": "float32"}],
        "constants": [],
        "nodes": [{"op": "Arange", "inputs": [], "outputs": ["A"], "arguments": [0.0, 1.0, 0.2]}],
    }
    arange_stub = _stub(arange_payload, model_name="legacy_arange_float")
    assert arange_stub["graph"]["nodes"] == []
    init = arange_stub["graph"]["initializers"][0]
    assert init["dtype"] == "float32"
    np.testing.assert_allclose(np.asarray(init["values"]), np.asarray([0.0, 0.2, 0.4, 0.6, 0.8]))

    reshape_payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [4], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [{"name": "y", "shape": [2, 2], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {
                "op": "Reshape",
                "inputs": ["x"],
                "outputs": ["y"],
                "arguments": [[2.0, 2.0]],
            }
        ],
    }
    reshape_stub = _stub(reshape_payload, model_name="legacy_reshape_shape_norm")
    node = reshape_stub["graph"]["nodes"][0]
    shape_init = {
        item["name"]: item for item in reshape_stub["graph"]["initializers"]
    }[node["inputs"][1]]
    assert shape_init["dtype"] == "int64"
    assert shape_init["values"] == [2, 2]


def test_stack_lowers_to_unsqueeze_unsqueeze_concat_with_axis_contract_and_parity():
    x = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    y = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)
    payload = ir.export_ir(lambda a, b: mx.stack([a, b], axis=1), x, y)
    stub = _stub(payload, model_name="legacy_stack")
    assert [node["op_type"] for node in stub["graph"]["nodes"]] == ["Unsqueeze", "Unsqueeze", "Concat"]
    assert stub["graph"]["nodes"][-1]["attributes"] == {"axis": 1}

    out = _run_ort(
        payload,
        {
            payload["inputs"][0]["name"]: np.asarray(x),
            payload["inputs"][1]["name"]: np.asarray(y),
        },
        model_name="legacy_stack",
    )
    _assert_close(np.stack([np.asarray(x), np.asarray(y)], axis=1), out[0])
