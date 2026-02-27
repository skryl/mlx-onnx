# mlx-onnx

[![Tests](https://github.com/skryl/mlx-onnx/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/skryl/mlx-onnx/actions/workflows/tests.yml?query=branch%3Amain)
![Version](https://img.shields.io/badge/version-0.30.7.3-blue)

`mlx-onnx` is a standalone IR/ONNX export library for MLX. It provides a Python package (`mlx_onnx`) and a native C++ library (`mlx_onnx`) for:

- exporting MLX callables to IR
- exporting MLX callables directly to ONNX
- converting IR payloads to ONNX

## Docs Index

- [Python Interface](docs/python-interface.md)
- [C++ Interface](docs/cpp-interface.md)
- [Native Architecture](docs/native-architecture.md)
- [Supported MLX Ops](docs/supported-mlx-ops.md)
- [ONNX WebGPU Demo](https://skryl.github.io/mlx-ruby/demo)

## Installation (pip)

Install from PyPI (pre-built wheel currently targets macOS + Python 3.14):

```bash
pip install mlx-onnx
```

If you are on a different platform or Python version, force a source build via `pip`:

```bash
python -m pip install --upgrade pip build
python -m pip install --no-binary mlx-onnx mlx-onnx
```

Install from source:

```bash
pip install .
```

The installed Python package includes the required bundled `mlx` build/source files under `mlx_onnx/_vendor/mlx`.

Install in editable mode for local development:

```bash
pip install -e .
```

Install from a local wheel:

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

W1 = mx.array([
    [0.2, -0.1, 0.4, 0.0, 0.3, -0.2],
    [-0.3, 0.5, 0.1, -0.4, 0.2, 0.1],
    [0.6, 0.2, -0.5, 0.3, -0.1, 0.2],
    [0.1, -0.2, 0.2, 0.5, 0.4, -0.3],
], dtype=mx.float32)
b1 = mx.array([0.1, -0.1, 0.05, 0.0, 0.2, -0.05], dtype=mx.float32)

W2 = mx.array([
    [0.3, -0.4],
    [0.1, 0.2],
    [-0.2, 0.5],
    [0.4, -0.1],
    [0.2, 0.3],
    [-0.5, 0.2],
], dtype=mx.float32)
b2 = mx.array([0.05, -0.02], dtype=mx.float32)

def tiny_mlp(x):
    h = mx.maximum(x @ W1 + b1, 0.0)
    return h @ W2 + b2

def forward(x):
    return tiny_mlp(x)

x = mx.array([[1.0, -2.0, 0.5, 3.0]], dtype=mx.float32)
mxonnx.export_onnx("tiny_mlp.onnx", forward, x, model_name="tiny_mlp", opset=18)
```

You can also run a compatibility pre-check before writing the ONNX file:

```python
report = mxonnx.export_onnx_compatibility_report(forward, x)
```


## C++ Quickstart

Example for consuming `mlx-onnx` from C++ and exporting a model directly to ONNX:

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

  auto onnx_path = ir::export_onnx(
      "model.onnx", forward, args, kwargs, /*shapeless=*/false, 18, "mlx_cpp_model", options);

  std::cout << "Wrote: " << onnx_path << std::endl;
  return 0;
}
```

### Consuming from CMake

Concrete example (same flow used by `mlx-ruby` in
`../mlx-ruby/graph-ir-onnx-webgpu-red-green/ext/mlx/extconf.rb`):

```bash
# 1) Build/install MLX as shared libs
cmake -S /path/to/mlx -B build/mlx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/build/install \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_PYTHON_STUBS=OFF \
  -DMLX_BUILD_GGUF=OFF \
  -DMLX_BUILD_SAFETENSORS=OFF \
  -DBUILD_SHARED_LIBS=ON
cmake --build build/mlx --target install --config Release -j8

# 2) Build/install mlx-onnx against that MLX install
cmake -S /path/to/mlx-onnx -B build/mlx-onnx \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/build/install \
  -DMLX_ONNX_USE_EXTERNAL_MLX=ON \
  -DMLX_ONNX_EXTERNAL_MLX_INCLUDE_DIR=/path/to/mlx \
  -DMLX_ONNX_EXTERNAL_MLX_LIB_DIR=$PWD/build/install/lib \
  -DMLX_ONNX_BUILD_PYTHON_BINDINGS=OFF
cmake --build build/mlx-onnx --target install --config Release -j8
```

Then link your C++ target against the installed `mlx_onnx` and `mlx` libraries:

```cmake
set(MLX_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/build/install")
add_executable(onnx_exporter main.cpp)
target_include_directories(
  onnx_exporter
  PRIVATE
    /path/to/mlx
    /path/to/mlx-onnx/include
    /path/to/mlx-onnx/src)
target_link_directories(onnx_exporter PRIVATE "${MLX_INSTALL_PREFIX}/lib")
target_link_libraries(onnx_exporter PRIVATE mlx_onnx)
target_link_libraries(onnx_exporter PRIVATE mlx)
set_target_properties(
  onnx_exporter
  PROPERTIES
    BUILD_RPATH "${MLX_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${MLX_INSTALL_PREFIX}/lib")
```

`mlx-ruby` also forces downstream compilation to use the same compiler pair used
for the CMake configure/build to avoid C++ ABI/link mismatches.


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
