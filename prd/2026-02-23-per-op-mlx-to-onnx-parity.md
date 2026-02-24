# PRD: Per-Op MLX->ONNX Unit Coverage with ONNXRuntime Parity

## Summary

Expand `python/tests/unit` to provide per-op coverage for MLX IR -> ONNX lowering.
For each mapped op, add a dedicated unit test file that verifies:

1. The MLX/IR op lowers to the expected ONNX op(s).
2. ONNXRuntime execution is numerically consistent with the MLX baseline.

This work also adds coverage for dynamic mapping paths (`Reduce`, `ArgReduce`).

Phase 2 extends this with strict lowering-contract validation and legacy-gap closure:

1. Exact ONNX attribute/rewrite contracts for critical lowerings (not only op-presence).
2. Coverage parity for legacy ONNX tests that were still missing/partial.

## Goals

- Add one test file per mapped op under `python/tests/unit`.
- Assert conversion correctness (`ir_to_onnx_json` contains expected ONNX op types).
- Assert numerical parity with ONNXRuntime for deterministic ops.
- Handle nondeterministic `RandomBits` via structural/runtime checks (shape, dtype, bounds).
- Centralize shared logic so per-op files stay small and maintainable.
- Add explicit contract tests for:
  - node sequence rewrites,
  - key attributes (`axis`, `perm`, `pads`, `keepdims`, `exclusive`, `reverse`, etc.),
  - expected auxiliary initializer tensors (shape/dtype/values),
  - failure contracts for unsupported argument combinations.
- Close missing legacy scenarios around shapeless behavior, complex64 initializers, and reduction edge paths.

## Non-Goals

- Changing production lowering behavior.
- Refactoring exporter internals.
- Covering every edge-case variant of each op in this phase.

## Scope

### In Scope

- Static mappings from `src/mappings.cpp:kOnnxOpPairs`.
- Dynamic mappings exercised by exported `Reduce` and `ArgReduce`.
- New shared test harness for:
  - exporting payloads,
  - ONNX stub inspection,
  - ONNXRuntime execution,
  - output comparison with appropriate tolerances.
- Contract-focused tests for critical rewrites:
  - `ArgReduce` -> `ArgMin/ArgMax + Cast`
  - `Reduce` axes/keepdims and bool decomposition rewrite
  - `Convolution`/`ConvolutionTranspose` transpose + kernel-layout rewrite details
  - `Slice`/`SliceUpdate`/`Split` initializer wiring contracts
  - `Gather`/`GatherAxis`/`ScatterAxis` axis and expand/cast behavior
  - `Pad`, `LogSumExp`, `Scan`, `Transpose`, `Flatten`, `Concatenate` rewrite details
- Legacy-gap closure tests:
  - shapeless multi-input/multi-output and mixed-op scenarios,
  - complex64 initializer acceptance (`__mlx_complex__` marker + string literal),
  - mean decomposition pattern (`ReduceSum + Broadcast + Multiply`),
  - AsType legacy fallback/validation behavior,
  - where-pattern rewrite assertions.

### Out of Scope

- Backward-compat matrix across multiple ONNXRuntime versions.
- Performance benchmarking.

## Proposed Structure

### New Shared Files

- `python/tests/unit/_op_cases.py`
  - Canonical case registry keyed by op name.
  - Case builders for traced MLX flows and explicit payload fallbacks.
- `python/tests/unit/_op_harness.py`
  - Reusable test execution utilities:
    - case execution,
    - ONNX op assertion,
    - ORT run,
    - parity comparison.
- `python/tests/unit/test_lowering_rewrite_contracts.py`
  - Strict per-op lowering contract assertions.
- `python/tests/unit/test_legacy_gap_closure.py`
  - Legacy parity coverage for missing scenarios.

### Per-Op Files

- `python/tests/unit/test_op_<op>.py` (one file per op).
- Each file runs exactly one case from the shared registry.
- Additional files for dynamic ops:
  - `test_op_reduce.py`
  - `test_op_argreduce.py`

## Functional Requirements

1. Each mapped op has a corresponding unit test file.
2. Each test must validate ONNX op lowering for the case.
3. Deterministic cases must compare ORT output to MLX output.
4. `RandomBits` must:
   - lower successfully,
   - execute in ORT,
   - satisfy shape/dtype/range assertions instead of value equality.
5. Tests skip cleanly when `onnxruntime` is unavailable.
6. Contract tests must assert exact rewrite details (node order/type/attributes and critical initializer tensors) for targeted operators.
7. Unsupported/invalid-lowering contracts must be asserted explicitly for known failure paths (e.g., unsupported convolution input dilation).
8. Legacy-gap scenarios must be represented by direct tests and pass.

## Case Design

- Prefer traced MLX callables for payload generation.
- Use explicit IR payloads only when traced form cannot reliably exercise the mapped op path.
- Include decomposed-lowering assertions for ops like `LayerNorm`, `RoPE`, `ErfInv` by asserting required ONNX op presence.

## Acceptance Criteria

- New files exist in `python/tests/unit` for each targeted op.
- Running `python -m pytest python/tests/unit -q` executes per-op tests without import/runtime errors.
- Deterministic op tests assert parity and pass with configured tolerances.
- RandomBits test validates structural/runtime constraints and passes.
- Contract tests validate exact rewrite/attribute semantics for targeted operators.
- Shapeless/complex64/mean/AsType legacy-gap tests pass and prevent regressions.

## Risks and Mitigations

- Risk: Some mapped ops are not emitted as standalone traced ops in current MLX tracing.
  - Mitigation: Use explicit payload case for those ops while preserving parity checks.
- Risk: Float precision differences between MLX and ORT.
  - Mitigation: dtype-aware tolerances in shared comparator.
- Risk: ORT dependency missing in certain environments.
  - Mitigation: `pytest.importorskip` gate in harness.

## Execution Plan

1. Add PRD document (this file).
2. Implement `_op_harness.py`.
3. Implement `_op_cases.py`.
4. Generate per-op test files wired to the registry.
5. Run `python/tests/unit` and fix failing cases iteratively.
6. Add strict lowering contract test suite.
7. Add legacy-gap closure tests for shapeless, complex64, reduce edge paths, and AsType behavior.
8. Re-run unit suite and close failures.

## Execution Status (2026-02-23)

- Completed:
  - Per-op test harness and case registry.
  - Per-op test files for mapped ops + dynamic `Reduce`/`ArgReduce`.
  - Strict rewrite/attribute contract suite (`test_lowering_rewrite_contracts.py`).
  - Legacy-gap closure suite (`test_legacy_gap_closure.py`), including:
    - shapeless dynamic/multi-output and stability checks,
    - complex64 marker/literal initializer acceptance and invalid input rejection,
    - AsType state/compat behavior,
    - mean decomposition contract,
    - arange/reshape regression contracts,
    - stack lowering contract.
- Validation:
  - `python -m pytest python/tests/unit -q` passes.
