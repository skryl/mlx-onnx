"""Export a tiny MLX MLP model to ONNX and verify ONNX Runtime parity."""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import mlx.core as mx
import mlx_onnx as mxonnx


def build_model():
    # Tiny 2-layer MLP weights captured as constants in the exported graph.
    w1 = mx.array(
        [
            [0.2, -0.1, 0.4, 0.0, 0.3, -0.2],
            [-0.3, 0.5, 0.1, -0.4, 0.2, 0.1],
            [0.6, 0.2, -0.5, 0.3, -0.1, 0.2],
            [0.1, -0.2, 0.2, 0.5, 0.4, -0.3],
        ],
        dtype=mx.float32,
    )
    b1 = mx.array([0.1, -0.1, 0.05, 0.0, 0.2, -0.05], dtype=mx.float32)

    w2 = mx.array(
        [
            [0.3, -0.4],
            [0.1, 0.2],
            [-0.2, 0.5],
            [0.4, -0.1],
            [0.2, 0.3],
            [-0.5, 0.2],
        ],
        dtype=mx.float32,
    )
    b2 = mx.array([0.05, -0.02], dtype=mx.float32)

    def model(x):
        h = mx.maximum(x @ w1 + b1, 0.0)
        return h @ w2 + b2

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="tiny_mlp.onnx",
        help="Path to write ONNX model",
    )
    args = parser.parse_args()

    model = build_model()
    x = mx.array([[1.0, -2.0, 0.5, 3.0]], dtype=mx.float32)

    mxonnx.export_onnx(args.output, model, x, model_name="tiny_mlp", opset=18)
    print(f"Exported: {args.output}")

    # Optional parity check with ONNX Runtime.
    mlx_out = np.array(model(x))
    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: np.array(x)})[0]
    print("MLX:", mlx_out)
    print("ONNX:", onnx_out)
    print("max abs diff:", np.max(np.abs(mlx_out - onnx_out)))


if __name__ == "__main__":
    main()
