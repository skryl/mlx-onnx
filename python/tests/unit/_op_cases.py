from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx_onnx as ir


@dataclass
class OpCase:
    case_id: str
    payload: dict[str, Any]
    feeds: dict[str, np.ndarray]
    expected_outputs: list[np.ndarray] | None
    expected_onnx_ops: tuple[str | tuple[str, ...], ...]
    random_bounds: tuple[int, int] | None = None
    rtol: float = 1e-4
    atol: float = 1e-5


def _f32(data: Any) -> mx.array:
    return mx.array(data, dtype=mx.float32)


def _i32(data: Any) -> mx.array:
    return mx.array(data, dtype=mx.int32)


def _to_numpy_outputs(value: Any) -> list[np.ndarray]:
    if isinstance(value, (list, tuple)):
        return [np.asarray(item) for item in value]
    return [np.asarray(value)]


def _feeds_from_payload(payload: dict[str, Any], args: tuple[Any, ...]) -> dict[str, np.ndarray]:
    input_specs = payload.get("inputs", [])
    if len(input_specs) != len(args):
        raise AssertionError(
            f"inputs/args mismatch: payload has {len(input_specs)} inputs, case has {len(args)} args"
        )
    feeds: dict[str, np.ndarray] = {}
    for spec, arg in zip(input_specs, args):
        feeds[spec["name"]] = np.asarray(arg)
    return feeds


def _trace_case(
    case_id: str,
    target_ir_op: str | tuple[str, ...],
    expected_onnx_ops: tuple[str | tuple[str, ...], ...],
    fun: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    random_bounds: tuple[int, int] | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> OpCase:
    payload = ir.export_ir(fun, *args)
    ir_ops = [node["op"] for node in payload["nodes"]]
    expected_ir_ops = (
        (target_ir_op,) if isinstance(target_ir_op, str) else target_ir_op
    )
    if not any(op in ir_ops for op in expected_ir_ops):
        expected_label = (
            f"one of {list(expected_ir_ops)}"
            if len(expected_ir_ops) > 1
            else f"'{expected_ir_ops[0]}'"
        )
        raise AssertionError(
            f"{case_id}: expected IR op {expected_label} in trace nodes, got {ir_ops}"
        )

    expected_outputs: list[np.ndarray] | None
    if random_bounds is None:
        expected_value = fun(*args)
        mx.eval(expected_value)
        expected_outputs = _to_numpy_outputs(expected_value)
    else:
        expected_outputs = None

    feeds = _feeds_from_payload(payload, args)
    return OpCase(
        case_id=case_id,
        payload=payload,
        feeds=feeds,
        expected_outputs=expected_outputs,
        expected_onnx_ops=expected_onnx_ops,
        random_bounds=random_bounds,
        rtol=rtol,
        atol=atol,
    )


def _manual_case(
    case_id: str,
    target_ir_op: str,
    expected_onnx_ops: tuple[str | tuple[str, ...], ...],
    payload: dict[str, Any],
    feeds: dict[str, np.ndarray],
    expected_outputs: list[np.ndarray],
) -> OpCase:
    ir_ops = [node["op"] for node in payload["nodes"]]
    if target_ir_op not in ir_ops:
        raise AssertionError(
            f"{case_id}: expected IR op '{target_ir_op}' in payload nodes, got {ir_ops}"
        )
    return OpCase(
        case_id=case_id,
        payload=payload,
        feeds=feeds,
        expected_outputs=expected_outputs,
        expected_onnx_ops=expected_onnx_ops,
        rtol=1e-4,
        atol=1e-5,
    )


def _build_add() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    y = _f32([[0.5, 1.5], [2.5, 3.5]])
    return _trace_case("Add", "Add", ("Add",), lambda a, b: a + b, (x, y))


def _build_addmm() -> OpCase:
    bias = _f32([0.1, -0.2])
    mat_a = _f32([[1.0, 2.0], [3.0, 4.0]])
    mat_b = _f32([[0.5, 1.5], [2.5, 3.5]])
    return _trace_case(
        "AddMM",
        "AddMM",
        ("Gemm",),
        lambda b, a, c: mx.addmm(b, a, c),
        (bias, mat_a, mat_b),
    )


def _build_subtract() -> OpCase:
    x = _f32([[4.0, 3.0], [2.0, 1.0]])
    y = _f32([[0.5, 1.5], [1.0, 0.5]])
    return _trace_case(
        "Subtract", "Subtract", ("Sub",), lambda a, b: a - b, (x, y)
    )


def _build_multiply() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    y = _f32([[0.5, 1.5], [2.0, 0.25]])
    return _trace_case(
        "Multiply", "Multiply", ("Mul",), lambda a, b: a * b, (x, y)
    )


def _build_square() -> OpCase:
    x = _f32([[-2.0, -1.0], [2.0, 3.0]])
    return _trace_case("Square", "Square", ("Mul",), lambda a: mx.square(a), (x,))


def _build_divide() -> OpCase:
    x = _f32([[2.0, 4.0], [9.0, 12.0]])
    y = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case("Divide", "Divide", ("Div",), lambda a, b: a / b, (x, y))


def _build_astype() -> OpCase:
    x = _f32([[1.25, 2.5], [3.75, 4.5]])
    return _trace_case(
        "AsType", "AsType", ("Cast",), lambda a: a.astype(mx.float16), (x,)
    )


def _build_exp() -> OpCase:
    x = _f32([[0.0, 1.0], [2.0, 3.0]])
    return _trace_case("Exp", "Exp", ("Exp",), lambda a: mx.exp(a), (x,))


def _build_log() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case("Log", "Log", ("Log",), lambda a: mx.log(a), (x,))


def _build_sin() -> OpCase:
    x = _f32([[0.0, 0.5], [1.0, 1.5]])
    return _trace_case("Sin", "Sin", ("Sin",), lambda a: mx.sin(a), (x,))


def _build_cos() -> OpCase:
    x = _f32([[0.0, 0.5], [1.0, 1.5]])
    return _trace_case("Cos", "Cos", ("Cos",), lambda a: mx.cos(a), (x,))


def _build_erf() -> OpCase:
    x = _f32([[-1.0, -0.5], [0.5, 1.0]])
    return _trace_case("Erf", "Erf", ("Erf",), lambda a: mx.erf(a), (x,))


def _build_erfinv() -> OpCase:
    x = _f32([[-0.7, -0.2], [0.2, 0.7]])
    return _trace_case(
        "ErfInv",
        "ErfInv",
        ("Where", "Sqrt", "Log"),
        lambda a: mx.erfinv(a),
        (x,),
        rtol=1e-3,
        atol=2e-4,
    )


def _build_sqrt() -> OpCase:
    x = _f32([[1.0, 4.0], [9.0, 16.0]])
    return _trace_case("Sqrt", "Sqrt", ("Sqrt",), lambda a: mx.sqrt(a), (x,))


def _build_abs() -> OpCase:
    x = _f32([[-1.0, 2.0], [-3.0, 4.0]])
    return _trace_case("Abs", "Abs", ("Abs",), lambda a: mx.abs(a), (x,))


def _build_floor() -> OpCase:
    x = _f32([[1.2, 2.8], [-1.2, -2.8]])
    return _trace_case("Floor", "Floor", ("Floor",), lambda a: mx.floor(a), (x,))


def _build_negative() -> OpCase:
    x = _f32([[1.0, -2.0], [3.0, -4.0]])
    return _trace_case("Negative", "Negative", ("Neg",), lambda a: -a, (x,))


def _build_relu() -> OpCase:
    x = np.asarray([[-1.0, 0.5], [2.0, -3.0]], dtype=np.float32)
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [2, 2], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [{"name": "y", "shape": [2, 2], "dtype": "float32"}],
        "constants": [],
        "nodes": [{"op": "Relu", "inputs": ["x"], "outputs": ["y"]}],
    }
    expected = [np.maximum(x, 0.0)]
    feeds = {"x": x}
    return _manual_case("Relu", "Relu", ("Relu",), payload, feeds, expected)


def _build_sigmoid() -> OpCase:
    x = _f32([[-2.0, -1.0], [1.0, 2.0]])
    return _trace_case(
        "Sigmoid", "Sigmoid", ("Sigmoid",), lambda a: mx.sigmoid(a), (x,)
    )


def _build_tanh() -> OpCase:
    x = _f32([[-2.0, -1.0], [1.0, 2.0]])
    return _trace_case("Tanh", "Tanh", ("Tanh",), lambda a: mx.tanh(a), (x,))


def _build_layernorm() -> OpCase:
    layer = nn.LayerNorm(4)
    mx.eval(layer.parameters())
    x = _f32([[0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0]])
    return _trace_case(
        "LayerNorm",
        ("LayerNorm", "Reduce", "Sqrt"),
        (("ReduceMean", "ReduceSum"), "Sqrt"),
        lambda a: layer(a),
        (x,),
    )


def _build_softmax() -> OpCase:
    x = _f32([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    return _trace_case(
        "Softmax", "Softmax", ("Softmax",), lambda a: mx.softmax(a, axis=-1), (x,)
    )


def _build_greater() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    y = _f32([[0.5, 2.5], [3.5, 1.0]])
    return _trace_case("Greater", "Greater", ("Greater",), lambda a, b: a > b, (x, y))


def _build_greater_equal() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    y = _f32([[0.5, 2.5], [3.5, 1.0]])
    return _trace_case(
        "GreaterEqual", "GreaterEqual", ("GreaterOrEqual",), lambda a, b: a >= b, (x, y)
    )


def _build_less() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    y = _f32([[0.5, 2.5], [3.5, 1.0]])
    return _trace_case("Less", "Less", ("Less",), lambda a, b: a < b, (x, y))


def _build_equal() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    y = _f32([[1.0, 2.5], [3.0, 0.0]])
    return _trace_case("Equal", "Equal", ("Equal",), lambda a, b: a == b, (x, y))


def _build_select() -> OpCase:
    cond = mx.array([[True, False], [False, True]])
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    y = _f32([[10.0, 20.0], [30.0, 40.0]])
    return _trace_case(
        "Select", "Select", ("Where",), lambda c, a, b: mx.where(c, a, b), (cond, x, y)
    )


def _build_full() -> OpCase:
    return _trace_case(
        "Full",
        "Full",
        ("Identity",),
        lambda: mx.full((), 3.5, dtype=mx.float32),
        (),
    )


def _build_randombits() -> OpCase:
    return _trace_case(
        "RandomBits",
        "RandomBits",
        ("RandomUniform",),
        lambda: mx.random.randint(0, 16, shape=(2, 2), dtype=mx.int32),
        (),
        random_bounds=(0, 16),
    )


def _build_matmul() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    y = _f32([[0.5, 1.5], [2.0, 3.0]])
    return _trace_case("Matmul", "Matmul", ("MatMul",), lambda a, b: a @ b, (x, y))


def _build_reshape() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "Reshape", "Reshape", ("Reshape",), lambda a: mx.reshape(a, (4,)), (x,)
    )


def _build_flatten() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "Flatten",
        "Flatten",
        ("Reshape",),
        lambda a: mx.flatten(a, start_axis=0, end_axis=1),
        (x,),
    )


def _build_unflatten() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "Unflatten",
        "Unflatten",
        ("Reshape",),
        lambda a: mx.unflatten(mx.reshape(a, (4,)), axis=0, shape=(2, 2)),
        (x,),
    )


def _build_transpose() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "Transpose", "Transpose", ("Transpose",), lambda a: mx.transpose(a), (x,)
    )


def _build_squeeze() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "Squeeze",
        "Squeeze",
        ("Squeeze",),
        lambda a: mx.squeeze(mx.expand_dims(a, 0), axis=0),
        (x,),
    )


def _build_expanddims() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "ExpandDims", "ExpandDims", ("Unsqueeze",), lambda a: mx.expand_dims(a, 0), (x,)
    )


def _build_broadcast() -> OpCase:
    x = _f32([[1.0, 2.0]])
    return _trace_case(
        "Broadcast",
        "Broadcast",
        ("Expand",),
        lambda a: mx.broadcast_to(a, (3, 2)),
        (x,),
    )


def _build_arange() -> OpCase:
    return _trace_case(
        "Arange",
        "Arange",
        (),
        lambda: mx.arange(0, 6, 2, dtype=mx.int32),
        (),
    )


def _build_asstrided() -> OpCase:
    x = mx.arange(0, 6, dtype=mx.float32)
    return _trace_case(
        "AsStrided",
        "AsStrided",
        ("Gather",),
        lambda a: mx.as_strided(a, shape=(2, 2), strides=(2, 1), offset=1),
        (x,),
    )


def _build_rope() -> OpCase:
    rope = nn.RoPE(dims=4, traditional=True, base=10000.0, scale=1.0)
    x = mx.arange(0, 32, dtype=mx.float32).reshape((1, 1, 4, 8)) / 10.0
    offset = mx.array(0, dtype=mx.int32)
    return _trace_case(
        "RoPE",
        ("RoPE", "Sin", "Cos"),
        ("Sin", "Cos"),
        lambda a, b: rope(a, b),
        (x, offset),
    )


def _build_concatenate() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    y = _f32([[5.0, 6.0], [7.0, 8.0]])
    return _trace_case(
        "Concatenate",
        "Concatenate",
        ("Concat",),
        lambda a, b: mx.concatenate([a, b], axis=1),
        (x, y),
    )


def _build_convolution() -> OpCase:
    x = mx.arange(0, 1 * 4 * 4 * 2, dtype=mx.float32).reshape((1, 4, 4, 2)) / 10.0
    w = mx.arange(0, 3 * 3 * 3 * 2, dtype=mx.float32).reshape((3, 3, 3, 2)) / 20.0
    return _trace_case(
        "Convolution",
        "Convolution",
        ("Conv",),
        lambda a, b: mx.conv2d(a, b, stride=1, padding=1),
        (x, w),
    )


def _build_convolutiontranspose() -> OpCase:
    x = mx.arange(0, 1 * 4 * 4 * 2, dtype=mx.float32).reshape((1, 4, 4, 2)) / 10.0
    w = mx.arange(0, 3 * 3 * 3 * 2, dtype=mx.float32).reshape((3, 3, 3, 2)) / 20.0
    return _trace_case(
        "ConvolutionTranspose",
        "Convolution",
        ("ConvTranspose",),
        lambda a, b: mx.conv_transpose2d(a, b, stride=1, padding=1),
        (x, w),
    )


def _build_gather() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    indices = _i32([0, 1])
    return _trace_case(
        "Gather", "Gather", ("Gather",), lambda a, b: mx.take(a, b, axis=0), (x, indices)
    )


def _build_gatheraxis() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    indices = _i32([[0, 1], [1, 0]])
    return _trace_case(
        "GatherAxis",
        "GatherAxis",
        ("GatherElements",),
        lambda a, b: mx.take_along_axis(a, b, axis=1),
        (x, indices),
    )


def _build_slice() -> OpCase:
    x = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype=np.float32,
    )
    payload = {
        "ir_version": 1,
        "shapeless": False,
        "inputs": [{"name": "x", "shape": [3, 4], "dtype": "float32"}],
        "keyword_inputs": [],
        "outputs": [{"name": "y", "shape": [3, 2], "dtype": "float32"}],
        "constants": [],
        "nodes": [
            {
                "op": "Slice",
                "inputs": ["x"],
                "outputs": ["y"],
                "arguments": [[0, 1], [3, 4], [1, 2]],
            }
        ],
    }
    expected = [x[0:3:1, 1:4:2]]
    feeds = {"x": x}
    return _manual_case("Slice", "Slice", ("Slice",), payload, feeds, expected)


def _build_sliceupdate() -> OpCase:
    x = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    u = np.asarray([9.0, 8.0], dtype=np.float32)
    payload = {
        "ir_version": 1,
        "shapeless": False,
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
    expected_value = x.copy()
    expected_value[1:3] = u
    return _manual_case(
        "SliceUpdate",
        "SliceUpdate",
        ("ScatterND",),
        payload,
        {"x": x, "u": u},
        [expected_value],
    )


def _build_split() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "Split", "Split", ("Split",), lambda a: mx.split(a, 2, axis=1), (x,)
    )


def _build_logsumexp() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "LogSumExp",
        "LogSumExp",
        ("ReduceLogSumExp",),
        lambda a: mx.logsumexp(a, axis=1, keepdims=True),
        (x,),
    )


def _build_pad() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "Pad",
        "Pad",
        ("Pad",),
        lambda a: mx.pad(a, ((1, 1), (1, 1)), mode="constant", constant_values=0.5),
        (x,),
    )


def _build_scan() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case("Scan", "Scan", ("CumSum",), lambda a: mx.cumsum(a, axis=1), (x,))


def _build_scatteraxis() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    indices = _i32([[0, 1], [1, 0]])
    updates = _f32([[9.0, 8.0], [7.0, 6.0]])
    return _trace_case(
        "ScatterAxis",
        "ScatterAxis",
        ("ScatterElements",),
        lambda a, b, c: mx.put_along_axis(a, b, c, axis=1),
        (x, indices, updates),
    )


def _build_maximum() -> OpCase:
    x = _f32([[1.0, 5.0], [3.0, 2.0]])
    y = _f32([[2.0, 4.0], [1.0, 6.0]])
    return _trace_case(
        "Maximum", "Maximum", ("Max",), lambda a, b: mx.maximum(a, b), (x, y)
    )


def _build_minimum() -> OpCase:
    x = _f32([[1.0, 5.0], [3.0, 2.0]])
    y = _f32([[2.0, 4.0], [1.0, 6.0]])
    return _trace_case(
        "Minimum", "Minimum", ("Min",), lambda a, b: mx.minimum(a, b), (x, y)
    )


def _build_power() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    y = _f32([[2.0, 2.0], [2.0, 2.0]])
    return _trace_case("Power", "Power", ("Pow",), lambda a, b: mx.power(a, b), (x, y))


def _build_reduce_sum() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "ReduceSum", "Reduce", ("ReduceSum",), lambda a: mx.sum(a, axis=1, keepdims=True), (x,)
    )


def _build_reduce_prod() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "ReduceProd",
        "Reduce",
        ("ReduceProd",),
        lambda a: mx.prod(a, axis=1, keepdims=True),
        (x,),
    )


def _build_reduce_min() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "ReduceMin", "Reduce", ("ReduceMin",), lambda a: mx.min(a, axis=1, keepdims=True), (x,)
    )


def _build_reduce_max() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "ReduceMax", "Reduce", ("ReduceMax",), lambda a: mx.max(a, axis=1, keepdims=True), (x,)
    )


def _build_argreduce_min() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "ArgReduceMin",
        "ArgReduce",
        ("ArgMin",),
        lambda a: mx.argmin(a, axis=1, keepdims=True),
        (x,),
    )


def _build_argreduce_max() -> OpCase:
    x = _f32([[1.0, 2.0], [3.0, 4.0]])
    return _trace_case(
        "ArgReduceMax",
        "ArgReduce",
        ("ArgMax",),
        lambda a: mx.argmax(a, axis=1, keepdims=True),
        (x,),
    )


CASE_BUILDERS: dict[str, Callable[[], OpCase]] = {
    "Add": _build_add,
    "AddMM": _build_addmm,
    "Subtract": _build_subtract,
    "Multiply": _build_multiply,
    "Square": _build_square,
    "Divide": _build_divide,
    "AsType": _build_astype,
    "Exp": _build_exp,
    "Log": _build_log,
    "Sin": _build_sin,
    "Cos": _build_cos,
    "Erf": _build_erf,
    "ErfInv": _build_erfinv,
    "Sqrt": _build_sqrt,
    "Abs": _build_abs,
    "Floor": _build_floor,
    "Negative": _build_negative,
    "Relu": _build_relu,
    "Sigmoid": _build_sigmoid,
    "Tanh": _build_tanh,
    "LayerNorm": _build_layernorm,
    "Softmax": _build_softmax,
    "Greater": _build_greater,
    "GreaterEqual": _build_greater_equal,
    "Less": _build_less,
    "Equal": _build_equal,
    "Select": _build_select,
    "Full": _build_full,
    "RandomBits": _build_randombits,
    "Matmul": _build_matmul,
    "Reshape": _build_reshape,
    "Flatten": _build_flatten,
    "Unflatten": _build_unflatten,
    "Transpose": _build_transpose,
    "Squeeze": _build_squeeze,
    "ExpandDims": _build_expanddims,
    "Broadcast": _build_broadcast,
    "Arange": _build_arange,
    "AsStrided": _build_asstrided,
    "RoPE": _build_rope,
    "Concatenate": _build_concatenate,
    "Convolution": _build_convolution,
    "ConvolutionTranspose": _build_convolutiontranspose,
    "Gather": _build_gather,
    "GatherAxis": _build_gatheraxis,
    "Slice": _build_slice,
    "SliceUpdate": _build_sliceupdate,
    "Split": _build_split,
    "LogSumExp": _build_logsumexp,
    "Pad": _build_pad,
    "Scan": _build_scan,
    "ScatterAxis": _build_scatteraxis,
    "Maximum": _build_maximum,
    "Minimum": _build_minimum,
    "Power": _build_power,
    "ReduceSum": _build_reduce_sum,
    "ReduceProd": _build_reduce_prod,
    "ReduceMin": _build_reduce_min,
    "ReduceMax": _build_reduce_max,
    "ArgReduceMin": _build_argreduce_min,
    "ArgReduceMax": _build_argreduce_max,
}


def build_case(case_id: str) -> OpCase:
    try:
        builder = CASE_BUILDERS[case_id]
    except KeyError as exc:
        raise KeyError(f"unknown op case '{case_id}'") from exc
    return builder()
