# PRD: MoE ONNX Export Unblock (`ArgPartition`, `GatherMM`)

## Summary

Unblock ONNX export for MoE routing/compute paths by adding native lowering support for:

1. `ArgPartition` (routing top-k selection path).
2. `GatherMM` (per-expert batched matmul path).

`Flatten`, `Gather`, and `GatherAxis` are already supported in this repository for the argument patterns used by MoE codepaths.

## Goals

- Add lowering support for `ArgPartition` using ONNX-standard operators.
- Add lowering support for `GatherMM` without introducing custom ONNX domains.
- Keep compatibility report behavior accurate (`unsupported_ops` should no longer include these ops for supported argument/shape patterns).
- Add unit and rewrite-contract coverage with ORT parity checks.
- Document support and lowering semantics in repository docs.

## Non-Goals

- Adding custom ONNX runtime kernels or custom-op execution providers.
- Implementing a full `argpartition` semantic equivalent for arbitrary downstream uses beyond top-k routing-oriented patterns.
- Performance tuning parity versus MLX kernels.

## Scope

### In Scope

- `src/lowering.cpp`
  - argument parsing and op-type recognition for `ArgPartition` and `GatherMM`
  - lowering rewrites:
    - `ArgPartition` -> `TopK` + cast to `uint32`
    - `GatherMM` -> reshape/gather/matmul decomposition
- `python/tests/unit`
  - per-op parity tests for both ops
  - rewrite-contract tests for both ops
  - mapping/support tests for both ops
- `docs/supported-mlx-ops.md`
  - add/clarify support rows and lowering notes

### Out of Scope

- exporter ABI changes
- new public Python/C++ API surface
- broad refactor of lowering pipeline

## Functional Requirements

1. `ArgPartition`
   - accepts `[kth, axis]` arguments
   - lowers via ONNX `TopK` with:
     - `k = kth + 1`
     - `largest = 0`
     - `sorted = 0`
   - indices output is cast to `uint32` to match MLX IR expectations

2. `GatherMM`
   - accepts `[left_sorted, right_sorted]` arguments
   - lowers with standard ops:
     - optional index casts to `int64`
     - optional index broadcast via `Expand`
     - flatten batch dimensions for lhs/rhs via `Reshape`
     - gather selected matrices along batch axis via `Gather(axis=0)`
     - compute per-index matmul via `MatMul`

3. Compatibility reports mark both ops as supported when arguments/shapes satisfy lowering constraints.

## Acceptance Criteria

- `ArgPartition` and `GatherMM` no longer show as unsupported for covered MoE patterns.
- New tests pass:
  - per-op parity tests
  - mapping/support tests
  - rewrite-contract + ORT parity tests
- Supported-op documentation updated.

## Risks and Mitigations

- Risk: `ArgPartition` is not a 1:1 semantic equivalent with `TopK` for all possible downstream uses.
  - Mitigation: document lowering semantics and scope to top-k routing-oriented usage.
- Risk: `GatherMM` shape/index broadcast corner cases.
  - Mitigation: enforce static-shape checks and add ORT parity tests over representative gathered matmul cases.

## Execution Status (2026-02-25)

- Completed:
  - `ArgPartition` lowering (`TopK` + cast) in `src/lowering.cpp`
  - `GatherMM` decomposition lowering in `src/lowering.cpp`
  - unit per-op tests:
    - `python/tests/unit/test_op_argpartition.py`
    - `python/tests/unit/test_op_gather_mm.py`
  - mapping/support coverage updates in `python/tests/unit/test_new_op_mappings.py`
  - rewrite-contract/parity tests in `python/tests/unit/test_lowering_rewrite_contracts.py`
  - docs update in `docs/supported-mlx-ops.md`
