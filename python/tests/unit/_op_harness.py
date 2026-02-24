from __future__ import annotations

import json
import os
import tempfile
from typing import Sequence

import numpy as np
import pytest

import mlx_onnx as ir

from _op_cases import build_case


def _require_onnxruntime():
    pytest.importorskip("numpy")
    ort = pytest.importorskip("onnxruntime")
    return ort


def _lowered_op_types(payload: dict, model_name: str) -> list[str]:
    stub = json.loads(ir.ir_to_onnx_json(payload, opset=18, model_name=model_name))
    return [node["op_type"] for node in stub["graph"]["nodes"]]


def _run_onnxruntime(payload: dict, feeds: dict[str, np.ndarray], model_name: str):
    ort = _require_onnxruntime()
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, f"{model_name}.onnx")
        ir.ir_to_onnx(onnx_path, payload, opset=18, model_name=model_name)
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        runtime_feeds = {
            input_info.name: np.asarray(feeds[input_info.name])
            for input_info in session.get_inputs()
        }
        return session.run(None, runtime_feeds)


def _assert_outputs_close(
    expected_outputs: Sequence[np.ndarray],
    actual_outputs,
    *,
    rtol: float,
    atol: float,
):
    if len(expected_outputs) != len(actual_outputs):
        raise AssertionError(
            f"output count mismatch: expected {len(expected_outputs)}, got {len(actual_outputs)}"
        )

    for expected, actual in zip(expected_outputs, actual_outputs):
        expected_arr = np.asarray(expected)
        actual_arr = np.asarray(actual)
        if expected_arr.dtype == np.bool_ or np.issubdtype(expected_arr.dtype, np.integer):
            np.testing.assert_array_equal(expected_arr, actual_arr)
            continue
        if expected_arr.dtype == np.float16:
            np.testing.assert_allclose(expected_arr, actual_arr, rtol=5e-3, atol=5e-3)
            continue
        np.testing.assert_allclose(expected_arr, actual_arr, rtol=rtol, atol=atol)


def _assert_random_outputs(actual_outputs, bounds: tuple[int, int]):
    lower, upper = bounds
    if len(actual_outputs) != 1:
        raise AssertionError(
            f"RandomBits case expects single output, got {len(actual_outputs)} outputs"
        )
    actual = np.asarray(actual_outputs[0])
    if not np.issubdtype(actual.dtype, np.integer):
        raise AssertionError(f"RandomBits output must be integer dtype, got {actual.dtype}")
    if actual.min() < lower:
        raise AssertionError(f"RandomBits output min {actual.min()} < expected lower {lower}")
    if actual.max() >= upper:
        raise AssertionError(f"RandomBits output max {actual.max()} >= expected upper {upper}")


def run_case(case_id: str):
    case = build_case(case_id)
    model_name = f"unit_{case_id.lower()}"
    op_types = _lowered_op_types(case.payload, model_name=model_name)
    for expected_onnx_op in case.expected_onnx_ops:
        if isinstance(expected_onnx_op, tuple):
            assert any(candidate in op_types for candidate in expected_onnx_op), (
                f"{case_id}: expected one of ONNX ops {list(expected_onnx_op)} "
                f"in lowered graph, got {op_types}"
            )
            continue
        assert expected_onnx_op in op_types, (
            f"{case_id}: expected ONNX op '{expected_onnx_op}' in lowered graph, got {op_types}"
        )

    try:
        actual_outputs = _run_onnxruntime(case.payload, case.feeds, model_name=model_name)
    except Exception as exc:
        message = str(exc)
        if case.random_bounds is not None and (
            "NOT_IMPLEMENTED" in message or "Could not find an implementation" in message
        ):
            pytest.skip(f"{case_id}: ORT backend does not implement required random op: {message}")
        raise
    if case.random_bounds is not None:
        _assert_random_outputs(actual_outputs, case.random_bounds)
        return

    if case.expected_outputs is None:
        raise AssertionError(f"{case_id}: expected outputs missing for deterministic case")
    _assert_outputs_close(
        case.expected_outputs, actual_outputs, rtol=case.rtol, atol=case.atol
    )
