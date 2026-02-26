from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

import mlx.core as mx
import mlx_onnx as ir

from _op_cases import build_case


def _stub(payload: dict, model_name: str) -> dict:
    return json.loads(ir.ir_to_onnx_json(payload, opset=18, model_name=model_name))


def _nodes(stub: dict) -> list[dict]:
    return list(stub["graph"]["nodes"])


def _op_types(stub: dict) -> list[str]:
    return [node["op_type"] for node in _nodes(stub)]


def _initializers(stub: dict) -> dict[str, dict]:
    return {item["name"]: item for item in stub["graph"].get("initializers", [])}


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


@pytest.mark.parametrize(
    ("case_id", "arg_op"),
    [("ArgReduceMin", "ArgMin"), ("ArgReduceMax", "ArgMax")],
)
def test_argreduce_exact_rewrite_contract(case_id: str, arg_op: str):
    case = build_case(case_id)
    stub = _stub(case.payload, model_name=f"contract_{case_id.lower()}")
    nodes = _nodes(stub)
    assert len(nodes) == 2
    assert [node["op_type"] for node in nodes] == [arg_op, "Cast"]
    assert nodes[0]["attributes"] == {"axis": 1, "keepdims": 1}
    assert nodes[1]["attributes"] == {"to": "UINT32"}


@pytest.mark.parametrize(
    ("case_id", "reduce_op"),
    [
        ("ReduceSum", "ReduceSum"),
        ("ReduceProd", "ReduceProd"),
        ("ReduceMin", "ReduceMin"),
        ("ReduceMax", "ReduceMax"),
    ],
)
def test_reduce_axes_and_keepdims_contract(case_id: str, reduce_op: str):
    case = build_case(case_id)
    stub = _stub(case.payload, model_name=f"contract_{case_id.lower()}")
    nodes = _nodes(stub)
    assert len(nodes) == 1
    assert nodes[0]["op_type"] == reduce_op
    assert nodes[0]["attributes"] == {"keepdims": 1}
    assert len(nodes[0]["inputs"]) == 2
    axes_name = nodes[0]["inputs"][1]
    axes_init = _initializers(stub)[axes_name]
    assert axes_init["dtype"] == "int64"
    assert axes_init["shape"] == [1]
    assert axes_init["values"] == [1]


@pytest.mark.parametrize(("reduce_code", "reduce_op"), [(0, "ReduceMin"), (1, "ReduceMax")])
def test_reduce_bool_rewrite_contract_and_parity(reduce_code: int, reduce_op: str):
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [2, 3], "dtype": "bool"}],
        "keyword_inputs": [],
        "outputs": [{"name": "y", "shape": [2, 1], "dtype": "bool"}],
        "constants": [],
        "nodes": [
            {
                "op": "Reduce",
                "inputs": ["x"],
                "outputs": ["y"],
                "arguments": [reduce_code, [1]],
            }
        ],
    }
    stub = _stub(payload, model_name=f"contract_reduce_bool_{reduce_code}")
    nodes = _nodes(stub)
    assert [node["op_type"] for node in nodes] == ["Cast", "Cast", reduce_op, "Cast"]
    assert nodes[0]["attributes"] == {"to": "BOOL"}
    assert nodes[1]["attributes"] == {"to": "INT64"}
    assert nodes[2]["attributes"] == {"keepdims": 1}
    assert nodes[3]["attributes"] == {"to": "BOOL"}

    x = np.asarray([[True, False, True], [False, False, True]], dtype=bool)
    outputs = _run_ort(payload, {"x": x}, model_name=f"parity_reduce_bool_{reduce_code}")
    if reduce_code == 0:
        expected = np.all(x, axis=1, keepdims=True)
    else:
        expected = np.any(x, axis=1, keepdims=True)
    _assert_close(expected, outputs[0])


def test_transpose_perm_attribute_contract():
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [2, 3, 4], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [{"name": "y", "shape": [3, 4, 2], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {
                "op": "Transpose",
                "inputs": ["x"],
                "outputs": ["y"],
                "arguments": [[1, 2, 0]],
            }
        ],
    }
    stub = _stub(payload, model_name="contract_transpose")
    node = _nodes(stub)[0]
    assert node["op_type"] == "Transpose"
    assert node["attributes"]["perm"] == [1, 2, 0]

    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    out = _run_ort(payload, {"x": x}, model_name="parity_transpose")
    _assert_close(np.transpose(x, (1, 2, 0)), out[0])


def test_slice_five_input_contract_and_parity():
    case = build_case("Slice")
    stub = _stub(case.payload, model_name="contract_slice")
    node = _nodes(stub)[0]
    assert node["op_type"] == "Slice"
    assert len(node["inputs"]) == 5
    inits = _initializers(stub)
    for aux in node["inputs"][1:]:
        assert aux in inits
        assert inits[aux]["dtype"] == "int64"
    out = _run_ort(case.payload, case.feeds, model_name="parity_slice")
    _assert_close(case.expected_outputs[0], out[0])


def test_slice_update_rewrite_contract_and_parity():
    case = build_case("SliceUpdate")
    stub = _stub(case.payload, model_name="contract_slice_update")
    nodes = _nodes(stub)
    assert [node["op_type"] for node in nodes] == ["Reshape", "ScatterND"]
    reshape_out = nodes[0]["outputs"][0]
    assert nodes[1]["inputs"][2] == reshape_out
    inits = _initializers(stub)
    indices_name = nodes[1]["inputs"][1]
    assert inits[indices_name]["dtype"] == "int64"
    assert inits[indices_name]["shape"] == [2, 1]
    assert inits[indices_name]["values"] == [1, 2]

    out = _run_ort(case.payload, case.feeds, model_name="parity_slice_update")
    _assert_close(case.expected_outputs[0], out[0])


def test_split_equal_contract_and_parity():
    case = build_case("Split")
    stub = _stub(case.payload, model_name="contract_split_equal")
    nodes = _nodes(stub)
    assert len(nodes) == 1
    split = nodes[0]
    assert split["op_type"] == "Split"
    assert split["attributes"] == {"axis": 1}
    split_init = _initializers(stub)[split["inputs"][1]]
    assert split_init["dtype"] == "int64"
    assert split_init["values"] == [1, 1]

    outs = _run_ort(case.payload, case.feeds, model_name="parity_split_equal")
    assert len(outs) == 2
    _assert_close(case.expected_outputs[0], outs[0])
    _assert_close(case.expected_outputs[1], outs[1])


def test_split_indices_contract_and_parity():
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [2, 6], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [
            {"name": "y0", "shape": [2, 2], "dtype": "float32"},
            {"name": "y1", "shape": [2, 3], "dtype": "float32"},
            {"name": "y2", "shape": [2, 1], "dtype": "float32"},
        ],
        "constants": [],
        "nodes": [
            {
                "op": "Split",
                "inputs": ["x"],
                "outputs": ["y0", "y1", "y2"],
                "arguments": [[2, 5], 1],
            }
        ],
    }
    stub = _stub(payload, model_name="contract_split_indices")
    node = _nodes(stub)[0]
    assert node["op_type"] == "Split"
    assert node["attributes"] == {"axis": 1}
    split_init = _initializers(stub)[node["inputs"][1]]
    assert split_init["values"] == [2, 3, 1]

    x = np.arange(12, dtype=np.float32).reshape(2, 6)
    out = _run_ort(payload, {"x": x}, model_name="parity_split_indices")
    expected = np.split(x, [2, 5], axis=1)
    _assert_close(expected[0], out[0])
    _assert_close(expected[1], out[1])
    _assert_close(expected[2], out[2])


def test_scan_reverse_exclusive_contract_and_parity():
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [2, 4], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [{"name": "y", "shape": [2, 4], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {
                "op": "Scan",
                "inputs": ["x"],
                "outputs": ["y"],
                "arguments": [2, 1, True, False],
            }
        ],
    }
    stub = _stub(payload, model_name="contract_scan")
    node = _nodes(stub)[0]
    assert node["op_type"] == "CumSum"
    assert node["attributes"] == {"exclusive": 1, "reverse": 1}
    axis_init = _initializers(stub)[node["inputs"][1]]
    assert axis_init["dtype"] == "int64"
    assert axis_init["values"] == [1]

    x = np.asarray([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    out = _run_ort(payload, {"x": x}, model_name="parity_scan")
    rev = np.flip(x, axis=1)
    rev_cumsum = np.cumsum(rev, axis=1)
    rev_exclusive = np.concatenate(
        [np.zeros((x.shape[0], 1), dtype=np.float32), rev_cumsum[:, :-1]], axis=1
    )
    expected = np.flip(rev_exclusive, axis=1)
    _assert_close(expected, out[0])


def test_convolution_exact_rewrite_contract():
    case = build_case("Convolution")
    stub = _stub(case.payload, model_name="contract_conv")
    nodes = _nodes(stub)
    assert [node["op_type"] for node in nodes] == ["Transpose", "Transpose", "Conv", "Transpose"]
    assert nodes[0]["attributes"]["perm"] == [0, 3, 1, 2]
    assert nodes[1]["attributes"]["perm"] == [0, 3, 1, 2]
    assert nodes[2]["attributes"] == {
        "strides": [1, 1],
        "pads": [1, 1, 1, 1],
        "dilations": [1, 1],
        "group": 1,
    }
    assert nodes[3]["attributes"]["perm"] == [0, 2, 3, 1]


def test_convolution_transpose_exact_rewrite_contract():
    case = build_case("ConvolutionTranspose")
    stub = _stub(case.payload, model_name="contract_convt")
    nodes = _nodes(stub)
    assert [node["op_type"] for node in nodes] == [
        "Transpose",
        "Transpose",
        "ConvTranspose",
        "Transpose",
    ]
    assert nodes[0]["attributes"]["perm"] == [0, 3, 1, 2]
    assert nodes[1]["attributes"]["perm"] == [3, 0, 1, 2]
    assert nodes[2]["attributes"] == {
        "strides": [1, 1],
        "pads": [1, 1, 1, 1],
        "dilations": [1, 1],
        "group": 1,
        "output_padding": [0, 0],
    }
    assert nodes[3]["attributes"]["perm"] == [0, 2, 3, 1]


def test_convolution_rejects_non_unity_input_dilation_when_flip_false():
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [
            {"name": "A", "shape": [1, 4, 4, 2], "dtype": "float32"},
            {"name": "B", "shape": [3, 3, 3, 2], "dtype": "float32"},
        ],
        "keyword_inputs": [],
        "outputs": [{"name": "C", "shape": [1, 4, 4, 3], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {
                "op": "Convolution",
                "inputs": ["A", "B"],
                "outputs": ["C"],
                "arguments": [[1, 1], [1, 1], [1, 1], [1, 1], [2, 1], 1, False],
            }
        ],
    }
    with pytest.raises(RuntimeError, match="unsupported Convolution input_dilation"):
        ir.ir_to_onnx_json(payload, opset=18, model_name="contract_conv_bad_dilation")


def test_convolution_add_int_bias_cast_contract_and_parity():
    x = mx.arange(0, 1 * 4 * 4 * 2, dtype=mx.float32).reshape((1, 4, 4, 2)) / 10.0
    w = mx.arange(0, 3 * 3 * 3 * 2, dtype=mx.float32).reshape((3, 3, 3, 2)) / 20.0
    b = mx.array([1, 2, 3], dtype=mx.int32)

    def fun(a, weight, bias):
        return mx.conv2d(a, weight, stride=1, padding=1) + bias

    payload = ir.export_ir(fun, x, w, b)
    stub = _stub(payload, model_name="contract_conv_bias_cast")
    assert "Cast" in _op_types(stub)
    cast_nodes = [node for node in _nodes(stub) if node["op_type"] == "Cast"]
    assert any(node["attributes"] == {"to": "FLOAT"} for node in cast_nodes)

    feeds = {
        payload["inputs"][0]["name"]: np.asarray(x),
        payload["inputs"][1]["name"]: np.asarray(w),
        payload["inputs"][2]["name"]: np.asarray(b),
    }
    out = _run_ort(payload, feeds, model_name="parity_conv_bias_cast")
    expected = np.asarray(fun(x, w, b))
    _assert_close(expected, out[0], atol=1e-4)


def test_gather_axis_contract_includes_unsqueeze_and_squeeze():
    case = build_case("Gather")
    stub = _stub(case.payload, model_name="contract_gather")
    nodes = _nodes(stub)
    assert [node["op_type"] for node in nodes] == ["Gather", "Unsqueeze", "Squeeze"]
    assert nodes[0]["attributes"] == {"axis": 0}


def test_gatheraxis_expand_rewrite_contract_and_parity():
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [
            {"name": "x", "shape": [2, 1, 3], "dtype": "float32"},
            {"name": "idx", "shape": [2, 4, 3], "dtype": "int32"},
        ],
        "keyword_inputs": [],
        "outputs": [{"name": "y", "shape": [2, 4, 3], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {
                "op": "GatherAxis",
                "inputs": ["x", "idx"],
                "outputs": ["y"],
                "arguments": [2],
            }
        ],
    }
    stub = _stub(payload, model_name="contract_gatheraxis_expand")
    assert _op_types(stub) == ["Expand", "GatherElements"]
    assert _nodes(stub)[1]["attributes"] == {"axis": 2}

    x = np.asarray([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)
    idx = np.asarray(
        [
            [[2, 1, 0], [0, 1, 2], [1, 2, 0], [0, 2, 1]],
            [[0, 2, 1], [1, 0, 2], [2, 1, 0], [1, 2, 0]],
        ],
        dtype=np.int32,
    )
    out = _run_ort(payload, {"x": x, "idx": idx}, model_name="parity_gatheraxis_expand")
    expected = np.take_along_axis(np.broadcast_to(x, (2, 4, 3)), idx, axis=2)
    _assert_close(expected, out[0])


def test_argpartition_topk_rewrite_contract_and_parity():
    x = mx.array([[0.2, -1.0, 3.0, 0.5], [2.0, 0.1, -0.4, 1.7]], dtype=mx.float32)
    payload = ir.export_ir(
        lambda a: mx.argpartition(a * -1.0, kth=1, axis=-1)[:, :2],
        x,
    )
    stub = _stub(payload, model_name="contract_argpartition_topk")
    nodes = _nodes(stub)
    assert "TopK" in _op_types(stub)
    topk = next(node for node in nodes if node["op_type"] == "TopK")
    cast = next(node for node in nodes if node["op_type"] == "Cast")

    assert topk["attributes"] == {"axis": 1, "largest": 0, "sorted": 0}
    k_init = _initializers(stub)[topk["inputs"][1]]
    assert k_init["dtype"] == "int64"
    assert k_init["shape"] == [1]
    assert k_init["values"] == [2]
    assert cast["attributes"] == {"to": "UINT32"}

    feeds = {payload["inputs"][0]["name"]: np.asarray(x)}
    out = _run_ort(payload, feeds, model_name="parity_argpartition_topk")
    expected = np.asarray(mx.argpartition(x * -1.0, kth=1, axis=-1)[:, :2])
    np.testing.assert_array_equal(np.sort(expected, axis=-1), np.sort(out[0], axis=-1))


def test_gathermm_rewrite_contract_and_parity():
    x = mx.array(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0]],
            [[2.0, 1.0, 0.5]],
        ],
        dtype=mx.float32,
    )
    w = mx.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
            [[0.5, 1.0], [1.5, -0.5], [0.0, 2.0]],
        ],
        dtype=mx.float32,
    )
    rhs = mx.array([0, 1, 0, 1], dtype=mx.uint32)

    payload = ir.export_ir(lambda a, b, c: mx.gather_mm(a, b, rhs_indices=c), x, w, rhs)
    stub = _stub(payload, model_name="contract_gathermm")
    nodes = _nodes(stub)
    op_types = [node["op_type"] for node in nodes]

    assert op_types.count("Reshape") == 2
    assert op_types.count("Gather") == 2
    assert op_types[-1] == "MatMul"
    assert any(node["op_type"] == "Cast" and node["attributes"] == {"to": "INT64"} for node in nodes)
    for gather in [node for node in nodes if node["op_type"] == "Gather"]:
        assert gather["attributes"] == {"axis": 0}

    feeds = {
        payload["inputs"][0]["name"]: np.asarray(x),
        payload["inputs"][1]["name"]: np.asarray(w),
        payload["inputs"][2]["name"]: np.asarray(rhs),
    }
    out = _run_ort(payload, feeds, model_name="parity_gathermm")
    expected = np.asarray(mx.gather_mm(x, w, rhs_indices=rhs))
    _assert_close(expected, out[0], atol=1e-4)


def test_pad_constant_mode_contract_and_parity():
    case = build_case("Pad")
    stub = _stub(case.payload, model_name="contract_pad")
    node = _nodes(stub)[0]
    assert node["op_type"] == "Pad"
    assert node["attributes"] == {"mode": "constant"}
    pads_init = _initializers(stub)[node["inputs"][1]]
    assert pads_init["dtype"] == "int64"
    assert pads_init["shape"] == [4]
    const_init = _initializers(stub)[node["inputs"][2]]
    assert const_init["shape"] == []

    out = _run_ort(case.payload, case.feeds, model_name="parity_pad")
    _assert_close(case.expected_outputs[0], out[0])


def test_logsumexp_axes_keepdims_contract():
    case = build_case("LogSumExp")
    stub = _stub(case.payload, model_name="contract_logsumexp")
    node = _nodes(stub)[0]
    assert node["op_type"] == "ReduceLogSumExp"
    assert node["attributes"] == {"keepdims": 1}
    axes_init = _initializers(stub)[node["inputs"][1]]
    assert axes_init["dtype"] == "int64"
    assert axes_init["values"] == [1]


def test_flatten_after_matmul_rewrite_contract_and_parity():
    x = mx.arange(0, 8, dtype=mx.float32).reshape((2, 4))
    y = mx.arange(0, 12, dtype=mx.float32).reshape((4, 3))
    payload = ir.export_ir(
        lambda a, b: mx.flatten(a @ b, start_axis=0, end_axis=1),
        x,
        y,
    )
    stub = _stub(payload, model_name="contract_matmul_flatten")
    assert _op_types(stub) == ["MatMul", "Reshape"]
    reshape = _nodes(stub)[1]
    shape_init = _initializers(stub)[reshape["inputs"][1]]
    assert shape_init["dtype"] == "int64"
    assert shape_init["values"] == [6]

    feeds = {
        payload["inputs"][0]["name"]: np.asarray(x),
        payload["inputs"][1]["name"]: np.asarray(y),
    }
    out = _run_ort(payload, feeds, model_name="parity_matmul_flatten")
    expected = np.asarray(mx.flatten(x @ y, start_axis=0, end_axis=1))
    _assert_close(expected, out[0])


def test_concatenate_default_flatten_rewrite_contract_and_parity():
    x = mx.arange(0, 4, dtype=mx.float32).reshape((2, 2))
    y = mx.arange(4, 8, dtype=mx.float32).reshape((2, 2))
    payload = ir.export_ir(
        lambda a, b: mx.concatenate([mx.flatten(a), mx.flatten(b)], axis=0),
        x,
        y,
    )
    stub = _stub(payload, model_name="contract_concat_flatten")
    assert _op_types(stub) == ["Reshape", "Reshape", "Concat"]
    assert _nodes(stub)[2]["attributes"] == {"axis": 0}

    feeds = {
        payload["inputs"][0]["name"]: np.asarray(x),
        payload["inputs"][1]["name"]: np.asarray(y),
    }
    out = _run_ort(payload, feeds, model_name="parity_concat_flatten")
    expected = np.concatenate([np.asarray(mx.flatten(x)), np.asarray(mx.flatten(y))], axis=0)
    _assert_close(expected, out[0])


def test_where_greater_pattern_rewrite_contract_and_parity():
    x = mx.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=mx.float32)
    payload = ir.export_ir(lambda a: mx.where(a > 0, a, mx.zeros_like(a)), x)
    stub = _stub(payload, model_name="contract_where_pattern")
    assert _op_types(stub) == ["Expand", "Greater", "Identity", "Where"]

    feeds = {payload["inputs"][0]["name"]: np.asarray(x)}
    out = _run_ort(payload, feeds, model_name="parity_where_pattern")
    expected = np.where(np.asarray(x) > 0, np.asarray(x), np.zeros_like(np.asarray(x)))
    _assert_close(expected, out[0])


def test_expm1_rewrite_contract_and_parity():
    case = build_case("Expm1")
    stub = _stub(case.payload, model_name="contract_expm1")
    assert _op_types(stub) == ["Exp", "Sub"]

    sub_node = _nodes(stub)[1]
    one_init = _initializers(stub)[sub_node["inputs"][1]]
    assert one_init["dtype"] == "float32"
    assert one_init["shape"] == [1]
    assert one_init["values"] == [1.0]

    out = _run_ort(case.payload, case.feeds, model_name="parity_expm1")
    _assert_close(case.expected_outputs[0], out[0])


def test_logaddexp_rewrite_contract_and_parity():
    case = build_case("LogAddExp")
    stub = _stub(case.payload, model_name="contract_logaddexp")
    assert _op_types(stub) == ["Max", "Sub", "Sub", "Exp", "Exp", "Add", "Log", "Add"]

    out = _run_ort(case.payload, case.feeds, model_name="parity_logaddexp")
    _assert_close(case.expected_outputs[0], out[0])


def test_less_equal_contract_and_parity():
    case = build_case("LessEqual")
    stub = _stub(case.payload, model_name="contract_less_equal")
    assert _op_types(stub) == ["LessOrEqual"]

    out = _run_ort(case.payload, case.feeds, model_name="parity_less_equal")
    _assert_close(case.expected_outputs[0], out[0])


def test_logical_and_cast_contract_and_parity():
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [
            {"name": "x", "shape": [2, 2], "dtype": "int32"},
            {"name": "y", "shape": [2, 2], "dtype": "int32"},
        ],
        "keyword_inputs": [],
        "outputs": [{"name": "z", "shape": [2, 2], "dtype": "bool"}],
        "constants": [],
        "nodes": [
            {
                "op": "LogicalAnd",
                "inputs": ["x", "y"],
                "outputs": ["z"],
                "arguments": [],
            }
        ],
    }
    stub = _stub(payload, model_name="contract_logical_and_cast")
    assert _op_types(stub) == ["Cast", "Cast", "And"]
    assert _nodes(stub)[0]["attributes"] == {"to": "BOOL"}
    assert _nodes(stub)[1]["attributes"] == {"to": "BOOL"}

    x = np.asarray([[1, 0], [2, 3]], dtype=np.int32)
    y = np.asarray([[1, 4], [0, 5]], dtype=np.int32)
    out = _run_ort(payload, {"x": x, "y": y}, model_name="parity_logical_and_cast")
    expected = np.logical_and(x, y)
    _assert_close(expected, out[0])


def test_bitwise_binary_shift_contract_and_parity():
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [
            {"name": "x", "shape": [2, 2], "dtype": "uint32"},
            {"name": "y", "shape": [2, 2], "dtype": "uint32"},
        ],
        "keyword_inputs": [],
        "outputs": [{"name": "z", "shape": [2, 2], "dtype": "uint32"}],
        "constants": [],
        "nodes": [
            {
                "op": "BitwiseBinary",
                "inputs": ["x", "y"],
                "outputs": ["z"],
                "arguments": [3],
            }
        ],
    }
    stub = _stub(payload, model_name="contract_bitwise_shift")
    assert _op_types(stub) == ["BitShift"]
    assert _nodes(stub)[0]["attributes"] == {"direction": "LEFT"}

    x = np.asarray([[1, 2], [3, 4]], dtype=np.uint32)
    y = np.asarray([[1, 2], [0, 1]], dtype=np.uint32)
    out = _run_ort(payload, {"x": x, "y": y}, model_name="parity_bitwise_shift")
    expected = np.left_shift(x, y)
    _assert_close(expected, out[0])


def test_bitwise_binary_signed_shift_cast_contract_and_parity():
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [
            {"name": "x", "shape": [2, 2], "dtype": "int32"},
            {"name": "y", "shape": [2, 2], "dtype": "int32"},
        ],
        "keyword_inputs": [],
        "outputs": [{"name": "z", "shape": [2, 2], "dtype": "int32"}],
        "constants": [],
        "nodes": [
            {
                "op": "BitwiseBinary",
                "inputs": ["x", "y"],
                "outputs": ["z"],
                "arguments": [4],
            }
        ],
    }
    stub = _stub(payload, model_name="contract_bitwise_signed_shift")
    assert _op_types(stub) == ["Cast", "Cast", "BitShift", "Cast"]
    assert _nodes(stub)[0]["attributes"] == {"to": "UINT32"}
    assert _nodes(stub)[1]["attributes"] == {"to": "UINT32"}
    assert _nodes(stub)[2]["attributes"] == {"direction": "RIGHT"}
    assert _nodes(stub)[3]["attributes"] == {"to": "INT32"}

    x = np.asarray([[16, 9], [8, 4]], dtype=np.int32)
    y = np.asarray([[1, 2], [3, 1]], dtype=np.int32)
    out = _run_ort(payload, {"x": x, "y": y}, model_name="parity_bitwise_signed_shift")
    expected = np.right_shift(x, y)
    _assert_close(expected, out[0])


def test_compat_report_expm1_flatten_cascade_resolved():
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [2, 3, 4], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [{"name": "z", "shape": [2, 12], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {"op": "Expm1", "inputs": ["x"], "outputs": ["y"], "arguments": []},
            {"op": "Flatten", "inputs": ["y"], "outputs": ["z"], "arguments": [1, 2]},
        ],
    }
    report = json.loads(ir.ir_compatibility_report_json(payload))
    assert report["unsupported_nodes"] == 0
    assert report["unsupported_ops"] == []


def test_compat_report_expm1_pad_cascade_resolved():
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [
            {"name": "x", "shape": [2, 2], "dtype": "float32"},
            {"name": "c", "shape": [], "dtype": "float32"},
        ],
        "keyword_inputs": [],
        "outputs": [{"name": "z", "shape": [4, 2], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {"op": "Expm1", "inputs": ["x"], "outputs": ["y"], "arguments": []},
            {"op": "Pad", "inputs": ["y", "c"], "outputs": ["z"], "arguments": [[0], [1], [1]]},
        ],
    }
    report = json.loads(ir.ir_compatibility_report_json(payload))
    assert report["unsupported_nodes"] == 0
    assert report["unsupported_ops"] == []


def test_compat_report_expm1_split_cascade_resolved():
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [4, 2], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [
            {"name": "a", "shape": [2, 2], "dtype": "float32"},
            {"name": "b", "shape": [2, 2], "dtype": "float32"},
        ],
        "constants": [],
        "nodes": [
            {"op": "Expm1", "inputs": ["x"], "outputs": ["y"], "arguments": []},
            {"op": "Split", "inputs": ["y"], "outputs": ["a", "b"], "arguments": [[2], 0]},
        ],
    }
    report = json.loads(ir.ir_compatibility_report_json(payload))
    assert report["unsupported_nodes"] == 0
    assert report["unsupported_ops"] == []


def test_flatten_dynamic_input_shape_contract_and_compat():
    payload = {
        "ir_version": 1,
        "shapeless": True,
        "inputs": [{"name": "x", "shape": [2, -1], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [{"name": "z", "shape": [-1], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {"op": "Split", "inputs": ["x"], "outputs": ["a", "b"], "arguments": [[1], 1]},
            {"op": "Flatten", "inputs": ["b"], "outputs": ["z"], "arguments": [0, 1]},
        ],
    }
    report = json.loads(ir.ir_compatibility_report_json(payload))
    assert report["unsupported_nodes"] == 0
    assert report["unsupported_ops"] == []

    stub = _stub(payload, model_name="contract_flatten_dynamic_shape")
    flatten_nodes = [node for node in _nodes(stub) if "_Flatten" in node["name"]]
    assert [node["op_type"] for node in flatten_nodes] == [
        "Shape",
        "Slice",
        "Slice",
        "ReduceProd",
        "Slice",
        "Concat",
        "Reshape",
    ]


def test_split_dynamic_equal_parts_contract_and_compat():
    payload = {
        "ir_version": 1,
        "shapeless": True,
        "inputs": [{"name": "x", "shape": [2, -1], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [
            {"name": "a", "shape": [2, -1], "dtype": "float32"},
            {"name": "b", "shape": [2, -1], "dtype": "float32"},
            {"name": "c", "shape": [2, -1], "dtype": "float32"},
        ],
        "constants": [],
        "nodes": [
            {
                "op": "Split",
                "inputs": ["x"],
                "outputs": ["a", "b", "c"],
                "arguments": [[3], 1],
            },
        ],
    }
    report = json.loads(ir.ir_compatibility_report_json(payload))
    assert report["unsupported_nodes"] == 0
    assert report["unsupported_ops"] == []

    stub = _stub(payload, model_name="contract_split_dynamic_equal")
    split = _nodes(stub)[0]
    assert split["op_type"] == "Split"
    assert split["attributes"] == {"axis": 1, "num_outputs": 3}
    assert len(split["inputs"]) == 1


def test_split_dynamic_boundaries_contract_and_compat():
    payload = {
        "ir_version": 1,
        "shapeless": True,
        "inputs": [{"name": "x", "shape": [2, -1], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [
            {"name": "a", "shape": [2, 4], "dtype": "float32"},
            {"name": "b", "shape": [2, 8], "dtype": "float32"},
            {"name": "c", "shape": [2, -1], "dtype": "float32"},
        ],
        "constants": [],
        "nodes": [
            {
                "op": "Split",
                "inputs": ["x"],
                "outputs": ["a", "b", "c"],
                "arguments": [[4, 12], 1],
            },
        ],
    }
    report = json.loads(ir.ir_compatibility_report_json(payload))
    assert report["unsupported_nodes"] == 0
    assert report["unsupported_ops"] == []

    stub = _stub(payload, model_name="contract_split_dynamic_boundaries")
    assert _op_types(stub) == ["Shape", "Gather", "Sub", "Concat", "Split"]
    split = _nodes(stub)[-1]
    assert split["attributes"] == {"axis": 1}
    assert len(split["inputs"]) == 2


def test_pad_dynamic_rank_without_constant_contract_and_compat():
    payload = {
        "ir_version": 1,
        "shapeless": True,
        "inputs": [{"name": "x", "shape": [2, -1], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [{"name": "z", "shape": [-1, -1], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {"op": "Split", "inputs": ["x"], "outputs": ["a", "b"], "arguments": [[1], 1]},
            {
                "op": "Pad",
                "inputs": ["b"],
                "outputs": ["z"],
                "arguments": [[0, 1], [1, 0], [0, 2]],
            },
        ],
    }
    report = json.loads(ir.ir_compatibility_report_json(payload))
    assert report["unsupported_nodes"] == 0
    assert report["unsupported_ops"] == []

    stub = _stub(payload, model_name="contract_pad_dynamic_rank")
    pad = next(node for node in _nodes(stub) if node["op_type"] == "Pad")
    assert pad["attributes"] == {"mode": "constant"}
    assert len(pad["inputs"]) == 4
    assert pad["inputs"][2] == ""
