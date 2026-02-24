from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

import mlx_onnx as ir


def stub(payload: dict, model_name: str) -> dict:
    return json.loads(ir.ir_to_onnx_json(payload, opset=18, model_name=model_name))


def nodes(onnx_stub: dict) -> list[dict]:
    return list(onnx_stub["graph"]["nodes"])


def op_types(onnx_stub: dict) -> list[str]:
    return [node["op_type"] for node in nodes(onnx_stub)]


def initializers(onnx_stub: dict) -> dict[str, dict]:
    return {item["name"]: item for item in onnx_stub["graph"].get("initializers", [])}


def run_ort(payload: dict, feeds: dict[str, np.ndarray], model_name: str) -> list[np.ndarray]:
    ort = pytest.importorskip("onnxruntime")
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, f"{model_name}.onnx")
        ir.ir_to_onnx(model_path, payload, opset=18, model_name=model_name)
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        runtime_feeds = {
            input_info.name: np.asarray(feeds[input_info.name])
            for input_info in session.get_inputs()
        }
        return [np.asarray(out) for out in session.run(None, runtime_feeds)]


def assert_close(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
):
    expected_arr = np.asarray(expected)
    actual_arr = np.asarray(actual)
    if expected_arr.dtype == np.bool_ or np.issubdtype(expected_arr.dtype, np.integer):
        np.testing.assert_array_equal(expected_arr, actual_arr)
        return
    if expected_arr.dtype == np.float16:
        np.testing.assert_allclose(expected_arr, actual_arr, rtol=5e-3, atol=5e-3)
        return
    np.testing.assert_allclose(expected_arr, actual_arr, rtol=rtol, atol=atol)
