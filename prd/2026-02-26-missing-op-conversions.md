# PRD: ONNX Compat Gap Closure (Pass-2)

## Context

Latest compatibility report (generated `2026-02-26T22:00:19Z`) shows 24 models still missing support for 4 ops:

1. `BitwiseBinary`
2. `Flatten`
3. `Pad`
4. `Split`

The missing nodes are concentrated in dynamic-shape or dtype-sensitive lowering paths:

- `BitwiseBinary` misses are `BitShift` invocations in `bitnet`.
- `Flatten` misses are broad across decoder families, indicating shape-inference dependency in compatibility probing.
- `Pad` miss appears in `mamba` with sparse pad axes.
- `Split` misses appear in `baichuan_m1`, `mamba`, and `mamba2` with mixed split specs.

## Problem Statement

Compatibility probing (`ir_compatibility_report_json`) simulates lowering node-by-node. Ops that depend on fully-known intermediate shapes or strict dtype assumptions can be reported unsupported even when a runtime-valid ONNX rewrite is possible.

Current gap: several conversion paths are still too strict for dynamic graph state during probing.

## Goals

1. Make all 4 missing ops convertible in compatibility probing and stub lowering for reported invocation patterns.
2. Preserve deterministic lowering for already-supported static paths.
3. Add targeted contract tests for dynamic fallbacks and dtype edge paths.
4. Keep exporter API unchanged.

## Non-Goals

- Introducing custom ONNX domains.
- Changing public Python/C++ interfaces.
- Large lowering architecture refactors.

## Scope

### In Scope

- `src/lowering.cpp`
  - `BitwiseBinary` shift dtype policy update.
  - `Flatten` dynamic-shape lowering fallback.
  - `Pad` sparse-axis lowering with unknown rank.
  - `Split` dynamic axis-dimension lowering fallbacks.
- `python/tests/unit/test_lowering_rewrite_contracts.py`
  - Update signed-shift behavior contract.
  - Add dynamic `Flatten`/`Pad`/`Split` compatibility contracts.
- `docs/supported-mlx-ops.md`
  - Align `BitwiseBinary` shift note with implemented behavior.

### Out of Scope

- Model registry changes.
- External Ruby-based report harness changes.

## Functional Requirements

1. `BitwiseBinary` shift (`opcode` 3/4) supports integer/bool inputs by:
   - casting to unsigned compute dtype,
   - applying ONNX `BitShift`,
   - casting back to output/promoted dtype when needed.
2. `Flatten` supports unknown input shape (unknown intermediate rank/shape) via dynamic shape graph construction.
3. `Pad` supports unknown input rank for non-negative explicit axes using ONNX `Pad` optional `axes` input.
4. `Split` supports unknown axis dimension for:
   - equal-parts form (`[parts]` with `num_outputs`),
   - explicit lengths,
   - boundary indices (`[b0, b1, ...]`) by deriving final chunk from runtime `Shape`.

## Design

### 1. `BitwiseBinary` Shift

- Preserve opcode mapping:
  - `0 -> BitwiseAnd`
  - `1 -> BitwiseOr`
  - `2 -> BitwiseXor`
  - `3 -> BitShift(direction=LEFT)`
  - `4 -> BitShift(direction=RIGHT)`
- For shift modes, choose compute dtype:
  - unsigned integer: unchanged
  - signed integer: mapped unsigned partner (`int32 -> uint32`, etc.)
  - `bool -> uint8`
- Cast both inputs to compute dtype, emit `BitShift`, cast output back when required.

### 2. `Flatten` Dynamic Fallback

When input shape is unknown at lowering time:

1. `Shape(input)`
2. slice prefix dims `[0:start)`
3. slice middle dims `[start:end+1)`
4. `ReduceProd(middle, axes=[0], keepdims=1)`
5. slice suffix dims `[end+1:]`
6. `Concat(prefix, middle_prod, suffix)`
7. `Reshape(input, target_shape)`

Static-shape path remains unchanged.

### 3. `Pad` Unknown-Rank Handling

- Parse pad arguments with optional rank.
- If rank unknown:
  - reject negative axes,
  - keep provided non-negative sparse axes.
- Lower to ONNX `Pad` with `pads` input and `axes` input when needed.
- Preserve optional constant input position semantics.

### 4. `Split` Unknown-Axis-Dimension Handling

If axis dim is unknown:

- `spec=[parts]` and `parts==num_outputs`:
  - emit `Split` with `num_outputs` attribute.
- `spec` length equals `num_outputs`:
  - use explicit split lengths input.
- boundary form (`len(spec)=num_outputs-1`):
  - compute final chunk length at runtime from `Shape -> Gather(axis_dim) -> Sub(last_boundary)`,
  - `Concat` with static earlier chunk lengths,
  - feed resulting dynamic split tensor to `Split`.

## Acceptance Criteria

1. Targeted unit suites for rewritten contracts pass.
2. Dynamic compatibility contracts for `Flatten`/`Pad`/`Split` pass (`unsupported_nodes == 0`).
3. Signed `BitwiseBinary` shift contract/parity passes with cast-based path.
4. Full unit suite remains green.

## Validation Plan

1. `python -m pip install -e .`
2. `python -m pytest python/tests/unit/test_new_op_mappings.py python/tests/unit/test_lowering_rewrite_contracts.py -q`
3. `python -m pytest python/tests/unit/test_op_flatten.py python/tests/unit/test_op_split.py python/tests/unit/test_op_pad.py python/tests/unit/test_op_bitwise_binary.py -q`
4. `python -m pytest python/tests/unit -q`

## Execution Status

Implemented in this pass:

- `BitwiseBinary` shift cast-path support for signed/bool dtypes.
- `Flatten` dynamic-shape fallback lowering.
- `Pad` unknown-rank sparse-axis lowering path.
- `Split` unknown-axis-dimension dynamic fallback paths.
- `Split` ambiguity resolution: when `len(spec) == num_outputs - 1`, treat as boundary indices before equal-parts shorthand.
- Contract tests for dynamic `Flatten`/`Pad`/`Split` and signed shift behavior.
- Docs updated to reflect current `BitwiseBinary` shift semantics.
