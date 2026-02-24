# mlx-onnx

[![Tests](https://github.com/skryl/mlx-onnx/actions/workflows/tests.yml/badge.svg)](https://github.com/skryl/mlx-onnx/actions/workflows/tests.yml)
![Version](https://img.shields.io/badge/version-0.30.7.1-blue)

`mlx-onnx` is a standalone IR/ONNX export library for MLX. It provides a Python package (`mlx_onnx`) and a native C++ library (`mlx_onnx`) for:

- exporting MLX callables to IR
- exporting MLX callables directly to ONNX
- converting IR payloads to ONNX

## Installation (pip)

Install from source:

```bash
pip install .
```

The installed Python package includes the required bundled `mlx` build/source files under `mlx_onnx/_vendor/mlx`.

Install in editable mode for local development:

```bash
pip install -e .
```

Install from a wheel:

```bash
python -m build --wheel
pip install dist/*.whl
```

## Python Quickstart

`mlx-onnx` builds and links against the bundled `mlx` submodule sources for Python bindings.
No external `mlx` install is required.

Example:

```python
import mlx.core as mx
import mlx_onnx as mxonnx

class MLP(mx.nn.Module):
    def __call__(self, x):
        ...

model = MLP()
x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)

def forward(x):
    return model(x)

mxonnx.export_onnx("model.onnx", forward, x)
```

You can also run a compatibility pre-check before writing the ONNX file:

```python
report = mxonnx.export_onnx_compatibility_report(forward, x)
```


## Python Interface

`mlx-onnx` is pip-installable as a standalone package:

```bash
pip install .
```

from the repository root (or `pip install -e .` for editable development).

The installed package exports:

- `export_ir`
- `export_ir_json`
- `export_onnx`
- `export_onnx_json`
- `ir_to_onnx`
- `ir_to_onnx_json`
- `ir_compatibility_report_json`
- `export_onnx_compatibility_report`

### Python API interface

All APIs are available as:

`import mlx_onnx as mxonnx`

All exported callables are expected to accept/return MLX arrays and must satisfy:

- Inputs to `export_*` and `export_onnx_*` trace wrappers can be positional arrays, keyword arrays, a single positional tuple, or a single positional dict.
- The callable output must be a single `mx.array` or a tuple/list of `mx.array`.

Function signatures:

- `export_ir(fun: Callable, *args, shapeless: bool = False, **kwargs) -> dict`
  - `fun`: callable that represents the forward pass.
  - `args`: positional MLX arrays (or tuple of arrays when a single positional arg is used).
  - `kwargs`: keyword MLX arrays.
  - `shapeless`: when `True`, ignores concrete input shapes during tracing.
  - Returns the traced IR payload as a Python `dict`.

- `export_ir_json(fun: Callable, *args, shapeless: bool = False, **kwargs) -> str`
  - Same inputs as `export_ir`.
  - Returns IR JSON text.

- `export_onnx_compatibility_report(fun: Callable, *args, shapeless: bool = False, **kwargs) -> dict`
  - Same inputs as `export_ir`.
  - Returns a compatibility report as `dict`.

- `export_onnx_json(fun: Callable, *args, shapeless: bool = False, opset: int = 18, model_name: str = "mlx_graph", **kwargs) -> str`
  - `opset`: positive integer ONNX opset.
  - `model_name`: non-empty model name.
  - Returns ONNX model JSON text.

- `export_onnx(target_path: Union[str, PathLike], fun: Callable, *args, shapeless: bool = False, opset: int = 18, model_name: str = "mlx_graph", external_data: bool = False, external_data_file: Optional[str] = None, external_data_size_threshold: int = 1024, **kwargs) -> str`
  - `target_path`: output file path.
  - `external_data`: enable external tensor storage.
  - `external_data_file`: explicit `.data` filename when `external_data=True`; defaults to `<stem>.data`.
  - `external_data_size_threshold`: byte threshold for deciding when to externalize tensor data.
  - Returns the output ONNX artifact path.

- `ir_to_onnx_json(ir_source: Union[dict, list, tuple, str, PathLike], *, opset: int = 18, model_name: str = "mlx_graph") -> str`
  - `ir_source`: MLX IR payload as in-memory object, tuple/list container, JSON string, or path to JSON file.
  - Returns ONNX model JSON text.

- `ir_to_onnx(target_path: Union[str, PathLike], ir_source: Union[dict, list, tuple, str, PathLike], *, opset: int = 18, model_name: str = "mlx_graph", external_data: bool = False, external_data_file: Optional[str] = None, external_data_size_threshold: int = 1024) -> str`
  - Same IR source and export controls as `ir_to_onnx_json`.
  - Writes ONNX binary artifact at `target_path`.
  - Returns the output ONNX artifact path.

- `ir_compatibility_report_json(ir_source: Union[dict, list, tuple, str, PathLike]) -> str`
  - `ir_source`: IR payload or serialized IR source.
  - Returns a compatibility report JSON string.

## C++ Quickstart

Example for consuming `mlx-onnx` from C++ (assuming you already have an upstream MLX
model built in C++ and a pair of inputs available):

```cpp
#include <iostream>

#include "mlx/array.h"
#include "mlx/ir.hpp"
#include "mlx/ops.h"

namespace mx = mlx::core;
namespace ir = mlx::onnx;

std::vector<mx::array> forward(const mx::Args& args, const mx::Kwargs&) {
  auto x = args.at(0);
  auto scale = args.at(1);
  return {x * scale};
}

int main() {
  mx::array input({1.0f, 2.0f, 3.0f});
  mx::array scale({2.0f, 2.0f, 2.0f});
  mx::Args args = {input, scale};
  mx::Kwargs kwargs{};

  ir::OnnxBinaryWriteOptions options;
  options.external_data = false;

  auto ir_json = ir::export_ir_json(forward, args, kwargs, /*shapeless=*/false);
  auto onnx_json = ir::export_onnx_json(
      forward, args, kwargs, /*shapeless=*/false, /*opset=*/18, "mlx_cpp_model");
  auto onnx_path = ir::export_onnx(
      "model.onnx", forward, args, kwargs, /*shapeless=*/false, 18, "mlx_cpp_model", options);

  std::cout << ir_json.size() << " bytes IR payload" << std::endl;
  std::cout << onnx_json.size() << " bytes ONNX JSON payload" << std::endl;
  std::cout << "Wrote: " << onnx_path << std::endl;

  ir::OnnxBinaryWriteOptions artifact_options;
  artifact_options.external_data = false;
  auto artifact = ir::build_onnx_binary_artifact_from_onnx_json(onnx_json, artifact_options);
  auto artifact_path = ir::write_onnx_binary_artifact_to_path(
      "model_artifact.onnx", artifact, artifact_options);
  std::cout << "Artifact written: " << artifact_path << std::endl;
  return 0;
}
```

### Consuming from CMake

If you are building `mlx-onnx` from source as part of your project, wire your target like this:

```cmake
add_subdirectory(/path/to/mlx-onnx mlx-onnx-build)

add_executable(onnx_exporter main.cpp)
target_link_libraries(onnx_exporter PRIVATE mlx_onnx)
target_include_directories(onnx_exporter PRIVATE /path/to/mlx-onnx/mlx)
```

If you are linking against an installed build, link the installed `mlx_onnx` shared/static library and ensure the include
paths for both `mlx` and `mlx-onnx` headers are available.

## C++ Interface

The standalone library exports the IR/ONNX API consumed by Python bindings
through `include/mlx/ir.hpp` under `mlx::onnx`:

- `export_ir_json`
- `export_onnx_compatibility_report_json`
- `export_onnx_json`
- `export_onnx`
- `ir_to_onnx_json`
- `ir_to_onnx`
- `ir_compatibility_report_json`
- `build_onnx_binary_artifact_from_onnx_json`
- `write_onnx_binary_artifact_to_path`
- `ir_is_unsupported_error_message`

`namespace`: `mlx::onnx` in `include/mlx/ir.hpp`

Shared types:

- `using IrCaptureFunction = std::function<std::vector<mlx::core::array>(const mlx::core::Args&, const mlx::core::Kwargs&)>;`
- `struct OnnxBinaryWriteOptions { bool external_data; std::string external_data_file; int64_t external_data_size_threshold; };`
  Defaults: `external_data=false`, `external_data_file="weights.bin"`, `external_data_size_threshold=1024`.
- `struct OnnxBinaryArtifact { std::string model_bytes; std::string external_data_bytes; bool has_external_data; };`

Function signatures and semantics:

- `std::string ir_to_onnx_json(const std::string& ir_json, int64_t opset, const std::string& model_name)`
  Input: IR JSON payload as string, ONNX opset (must be positive), model name (must be non-empty).
  Returns ONNX JSON string.

- `std::string ir_to_onnx(const std::string& target_path, const std::string& ir_json, int64_t opset, const std::string& model_name, const OnnxBinaryWriteOptions& options)`
  Same IR/metadata inputs as above plus output path and write options.
  Returns the written ONNX artifact path.

- `std::string ir_compatibility_report_json(const std::string& ir_json)`
  Input: IR JSON string.
  Returns compatibility report JSON string.

- `std::string export_ir_json(const IrCaptureFunction& fun, const mlx::core::Args& args, const mlx::core::Kwargs& kwargs, bool shapeless)`
  Inputs: captured function and concrete positional/keyword tracing inputs.
  `shapeless=true` allows shape-agnostic tracing behavior in supported paths.
  Returns traced MLX IR JSON string.

- `std::string export_onnx_compatibility_report_json(const IrCaptureFunction& fun, const mlx::core::Args& args, const mlx::core::Kwargs& kwargs, bool shapeless)`
  Same function/input contract as `export_ir_json`.
  Returns compatibility report JSON string.

- `std::string export_onnx_json(const IrCaptureFunction& fun, const mlx::core::Args& args, const mlx::core::Kwargs& kwargs, bool shapeless, int64_t opset, const std::string& model_name)`
  Traces and directly emits ONNX JSON.
  Returns ONNX JSON string.

- `std::string export_onnx(const std::string& target_path, const IrCaptureFunction& fun, const mlx::core::Args& args, const mlx::core::Kwargs& kwargs, bool shapeless, int64_t opset, const std::string& model_name, const OnnxBinaryWriteOptions& options)`
  Same tracing and conversion inputs as above plus output path and write options.
  Returns the written ONNX artifact path.

- `OnnxBinaryArtifact build_onnx_binary_artifact_from_onnx_json(const std::string& onnx_json, const OnnxBinaryWriteOptions& options)`
  Input: ONNX JSON text and write options.
  Returns `OnnxBinaryArtifact` containing serialized model bytes and optional external tensor bytes.

- `std::string write_onnx_binary_artifact_to_path(const std::string& target_path, const OnnxBinaryArtifact& artifact, const OnnxBinaryWriteOptions& options)`
  Writes a binary ONNX artifact plus optional external tensor file(s) as configured.
  Returns final written ONNX path.

- `bool ir_is_unsupported_error_message(const std::string& message)`
  Returns `true` when `message` represents an unsupported-operator IR export error.

## Development

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip build
pip install -e ".[test]"
```

### Run tests

```bash
python -m unittest python/tests/test_ir.py
```

The test extra includes ONNX parity dependencies (`numpy`, `onnx`, `onnxruntime`).

### Build package artifacts

```bash
python -m build --wheel
```

### Build native targets with CMake

```bash
cmake -S . -B build -DMLX_ONNX_BUILD_PYTHON_BINDINGS=ON
cmake --build build
```
