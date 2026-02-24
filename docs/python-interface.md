# Python Interface

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

## Python API Interface

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
