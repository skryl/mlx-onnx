# Supported MLX Ops

This document lists the MLX Graph IR ops currently supported by `mlx-onnx` for ONNX conversion in this repository.

Source of truth:
- `src/mappings.cpp` (`kOnnxOpPairs`, `kReduceCodeToOnnxOp`, `kArgReduceCodeToOnnxOp`)
- `src/lowering.cpp` dynamic op-specific lowering paths

Support is argument/shape/dtype dependent for some ops. For a concrete graph, use `export_onnx_compatibility_report` (Python) or `ir_compatibility_report_json`.

## Direct Op Mappings

| MLX IR op | ONNX op type |
| --- | --- |
| `Add` | `Add` |
| `AddMM` | `Gemm` |
| `Subtract` | `Sub` |
| `Multiply` | `Mul` |
| `Square` | `Mul` |
| `Divide` | `Div` |
| `AsType` | `Cast` |
| `Exp` | `Exp` |
| `Log` | `Log` |
| `Sin` | `Sin` |
| `Cos` | `Cos` |
| `Erf` | `Erf` |
| `ErfInv` | `ErfInv` |
| `Sqrt` | `Sqrt` |
| `Abs` | `Abs` |
| `Floor` | `Floor` |
| `Negative` | `Neg` |
| `Relu` | `Relu` |
| `Sigmoid` | `Sigmoid` |
| `Tanh` | `Tanh` |
| `LayerNorm` | `LayerNormalization` |
| `Softmax` | `Softmax` |
| `Greater` | `Greater` |
| `Less` | `Less` |
| `Equal` | `Equal` |
| `Select` | `Where` |
| `Full` | `Identity` |
| `RandomBits` | `RandomUniform` |
| `Matmul` | `MatMul` |
| `Reshape` | `Reshape` |
| `Flatten` | `Reshape` |
| `Unflatten` | `Reshape` |
| `Transpose` | `Transpose` |
| `Squeeze` | `Squeeze` |
| `ExpandDims` | `Unsqueeze` |
| `Broadcast` | `Expand` |
| `Arange` | `Constant` |
| `AsStrided` | `Gather` |
| `RoPE` | `Identity` |
| `Concatenate` | `Concat` |
| `Convolution` | `Conv` |
| `ConvolutionTranspose` | `ConvTranspose` |
| `Gather` | `Gather` |
| `GatherAxis` | `GatherElements` |
| `Slice` | `Slice` |
| `SliceUpdate` | `ScatterND` |
| `Split` | `Split` |
| `LogSumExp` | `ReduceLogSumExp` |
| `Pad` | `Pad` |
| `Scan` | `CumSum` |
| `ScatterAxis` | `ScatterElements` |
| `Maximum` | `Max` |
| `Minimum` | `Min` |
| `Power` | `Pow` |

## Dynamic Reduction Mappings

`Reduce` and `ArgReduce` map by encoded reducer mode rather than a single static op type.

### `Reduce` mode -> ONNX op

| Reduce mode | ONNX op type |
| --- | --- |
| `0` | `ReduceMin` |
| `1` | `ReduceMax` |
| `2` | `ReduceSum` |
| `3` | `ReduceProd` |
| `4` | `ReduceMin` |
| `5` | `ReduceMax` |

### `ArgReduce` mode -> ONNX op

| ArgReduce mode | ONNX op type |
| --- | --- |
| `0` | `ArgMin` |
| `1` | `ArgMax` |

## Important Lowering Notes

- `Convolution` can lower to `ConvTranspose` when IR arguments indicate flipped convolution semantics (`flip=true`).
- Some ops above are rewritten into multi-node ONNX subgraphs during lowering (for example `Flatten`, `Select`, `LayerNorm`, and `RoPE` paths).
- Compatibility is validated per-node during report generation; unsupported argument combinations are surfaced as unsupported nodes.
