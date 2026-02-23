import hashlib
import importlib.util
import json
import os
import tempfile
import unittest

import mlx.core as mx
import mlx_onnx as ir

try:
    import mlx_tests
except ModuleNotFoundError:  # Standalone package mode
    BaseIrTestCase = unittest.TestCase
else:
    BaseIrTestCase = mlx_tests.MLXTestCase


class TestOnnx(BaseIrTestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir_handle = tempfile.TemporaryDirectory()
        cls.tmp_dir = cls.tmp_dir_handle.name

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir_handle.cleanup()

    def _sample_inputs(self):
        x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        y = mx.array([0.5, 0.25, 0.125], dtype=mx.float32)
        return x, y

    def _sample_fun(self):
        def fun(x, y=None):
            out = mx.exp(x)
            if y is not None:
                out = out + y
            return out

        return fun

    def _sample_payload(self):
        fun = self._sample_fun()
        x, y = self._sample_inputs()
        return ir.export_ir(fun, x, y=y)

    def _constant_fun_and_input(self, size=2):
        values = [float(i + 2) for i in range(size)]
        constant = mx.array(values, dtype=mx.float32)
        mx.eval(constant)

        def fun(x):
            return x + constant

        x = mx.array([1.0] * size, dtype=mx.float32)
        return fun, x, values

    def _constant_payload(self, size=2):
        values = [float(i + 10) for i in range(size)]
        return {
            "ir_version": 1,
            "shapeless": False,
            "inputs": [{"name": "x", "shape": [size], "dtype": "float32"}],
            "keyword_inputs": [],
            "outputs": [{"name": "z", "shape": [size], "dtype": "float32"}],
            "constants": [
                {
                    "name": "c",
                    "shape": [size],
                    "dtype": "float32",
                    "values": values,
                }
            ],
            "nodes": [{"op": "Add", "inputs": ["x", "c"], "outputs": ["z"]}],
        }

    def _path(self, suffix):
        return os.path.join(self.tmp_dir, f"{self._testMethodName}_{suffix}")

    def _read_bytes(self, path):
        with open(path, "rb") as handle:
            return handle.read()

    def _write_json_file(self, path, payload):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def _require_module(self, module_name):
        if importlib.util.find_spec(module_name) is None:
            self.skipTest(f"requires Python module '{module_name}'")

    def _assert_nested_close(self, expected, actual, atol=1e-5):
        if isinstance(expected, list):
            self.assertEqual(len(expected), len(actual))
            for index, value in enumerate(expected):
                self._assert_nested_close(value, actual[index], atol=atol)
            return
        self.assertAlmostEqual(float(expected), float(actual), delta=atol)

    def _run_onnxruntime(self, model_path, feeds):
        self._require_module("onnxruntime")
        self._require_module("numpy")
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        runtime_inputs = {}
        for inp in session.get_inputs():
            runtime_inputs[inp.name] = np.array(feeds[inp.name], dtype=np.float32)
        outputs = session.run(None, runtime_inputs)
        return [out.tolist() for out in outputs]

    def _python_build_onnx_from_stub(
        self,
        stub,
        output_path,
        use_external_data=False,
        external_data_location="weights.bin",
        external_data_threshold=1024,
    ):
        self._require_module("onnx")
        self._require_module("numpy")
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        dtype_map = {
            "bool": TensorProto.BOOL,
            "bool_": TensorProto.BOOL,
            "uint8": TensorProto.UINT8,
            "uint16": TensorProto.UINT16,
            "uint32": TensorProto.UINT32,
            "uint64": TensorProto.UINT64,
            "int8": TensorProto.INT8,
            "int16": TensorProto.INT16,
            "int32": TensorProto.INT32,
            "int64": TensorProto.INT64,
            "float16": TensorProto.FLOAT16,
            "float32": TensorProto.FLOAT,
            "float64": TensorProto.DOUBLE,
            "bfloat16": TensorProto.BFLOAT16,
            "complex64": TensorProto.COMPLEX64,
        }
        bool_dtypes = {"bool", "bool_"}
        int_dtypes = {
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
        }
        float_dtypes = {"float16", "float32", "float64", "bfloat16"}

        def flatten_values(value):
            if isinstance(value, list):
                out = []
                for item in value:
                    out.extend(flatten_values(item))
                return out
            return [value]

        def expected_value_count(dims):
            if not dims:
                return 1
            total = 1
            for dim in dims:
                total *= int(dim)
            return total

        def cast_initializer_values(values, dtype_name):
            if dtype_name in bool_dtypes:
                return [bool(v) for v in values]
            if dtype_name in int_dtypes:
                return [int(v) for v in values]
            if dtype_name in float_dtypes:
                return [float(v) for v in values]
            if dtype_name == "complex64":
                out = []
                for value in values:
                    if isinstance(value, dict) and "__mlx_complex__" in value:
                        pair = value["__mlx_complex__"]
                        out.append(complex(float(pair[0]), float(pair[1])))
                    elif isinstance(value, (bool, int, float)):
                        out.append(complex(float(value), 0.0))
                    else:
                        raise RuntimeError(
                            f"unsupported complex64 initializer value: {value!r}"
                        )
                return out
            raise RuntimeError(f"unsupported initializer dtype: {dtype_name!r}")

        def tensor_value_info(spec):
            return helper.make_tensor_value_info(
                spec["name"],
                dtype_map[spec["dtype"]],
                [int(dim) for dim in spec["shape"]],
            )

        def initializer_tensor(spec):
            dtype_name = spec["dtype"]
            elem_type = dtype_map[dtype_name]
            dims = [int(dim) for dim in spec["shape"]]
            values = flatten_values(spec["values"])
            expected = expected_value_count(dims)
            if len(values) != expected:
                raise RuntimeError(
                    f"initializer {spec['name']!r} has wrong value count"
                )
            cast_values = cast_initializer_values(values, dtype_name)
            return helper.make_tensor(spec["name"], elem_type, dims, cast_values)

        graph_spec = stub["graph"]
        nodes = []
        for node in graph_spec["nodes"]:
            attrs = dict(node.get("attributes", {}))
            if node["op_type"] == "Cast" and isinstance(attrs.get("to"), str):
                attrs["to"] = getattr(TensorProto, attrs["to"])
            nodes.append(
                helper.make_node(
                    node["op_type"],
                    list(node["inputs"]),
                    list(node["outputs"]),
                    name=node.get("name", ""),
                    **attrs,
                )
            )

        graph = helper.make_graph(
            nodes=nodes,
            name=graph_spec["name"],
            inputs=[tensor_value_info(spec) for spec in graph_spec["inputs"]],
            outputs=[tensor_value_info(spec) for spec in graph_spec["outputs"]],
            initializer=[
                initializer_tensor(spec)
                for spec in graph_spec.get("initializers", [])
            ],
        )

        model = helper.make_model(
            graph,
            producer_name=stub.get("producer_name", "mlx"),
            opset_imports=[helper.make_operatorsetid("", int(stub["opset"]))],
        )
        if use_external_data:
            for index, initializer in enumerate(model.graph.initializer):
                array = numpy_helper.to_array(initializer)
                model.graph.initializer[index].CopyFrom(
                    numpy_helper.from_array(array, initializer.name)
                )

        onnx.checker.check_model(model)
        if use_external_data:
            onnx.save_model(
                model,
                output_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data_location,
                size_threshold=external_data_threshold,
            )
        else:
            onnx.save(model, output_path)

    def _onnx_model_summary(self, model_path):
        self._require_module("onnx")
        import onnx
        from onnx import helper

        model = onnx.load(model_path, load_external_data=True)

        def norm(value):
            if isinstance(value, bytes):
                return {"bytes_hex": value.hex()}
            if isinstance(value, list):
                return [norm(item) for item in value]
            if isinstance(value, tuple):
                return [norm(item) for item in value]
            if isinstance(value, dict):
                return {key: norm(val) for key, val in value.items()}
            return value

        nodes = []
        for node in model.graph.node:
            attrs = []
            for attr in node.attribute:
                attrs.append([attr.name, norm(helper.get_attribute_value(attr))])
            nodes.append(
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "attributes": sorted(attrs, key=lambda item: item[0]),
                }
            )

        initializers = []
        for tensor in model.graph.initializer:
            initializers.append(
                {
                    "name": tensor.name,
                    "data_type": int(tensor.data_type),
                    "dims": list(tensor.dims),
                    "raw_sha256": hashlib.sha256(tensor.raw_data).hexdigest(),
                    "data_location": int(tensor.data_location),
                    "external_data": sorted(
                        [[entry.key, entry.value] for entry in tensor.external_data]
                    ),
                }
            )

        opset = int(model.opset_import[0].version)
        return {
            "opset": opset,
            "producer_name": model.producer_name,
            "inputs": [
                {
                    "name": info.name,
                    "elem_type": int(info.type.tensor_type.elem_type),
                    "shape": [
                        int(dim.dim_value) for dim in info.type.tensor_type.shape.dim
                    ],
                }
                for info in model.graph.input
            ],
            "outputs": [
                {
                    "name": info.name,
                    "elem_type": int(info.type.tensor_type.elem_type),
                    "shape": [
                        int(dim.dim_value) for dim in info.type.tensor_type.shape.dim
                    ],
                }
                for info in model.graph.output
            ],
            "nodes": nodes,
            "initializers": sorted(initializers, key=lambda item: item["name"]),
        }

    def _compare_native_and_python_oracle(
        self, payload, external_data=False, threshold=1024
    ):
        self._require_module("onnx")
        self._require_module("numpy")
        stub = json.loads(
            ir.ir_to_onnx_json(payload, opset=18, model_name="py_oracle_case")
        )

        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as root:
            native_dir = os.path.join(root, "native")
            python_dir = os.path.join(root, "python")
            os.makedirs(native_dir, exist_ok=True)
            os.makedirs(python_dir, exist_ok=True)

            native_path = os.path.join(native_dir, "model.onnx")
            python_path = os.path.join(python_dir, "model.onnx")

            if external_data:
                ir.ir_to_onnx(
                    native_path,
                    payload,
                    opset=18,
                    model_name="py_oracle_case",
                    external_data=True,
                    external_data_file="model.data",
                    external_data_size_threshold=threshold,
                )
                self._python_build_onnx_from_stub(
                    stub,
                    python_path,
                    use_external_data=True,
                    external_data_location="model.data",
                    external_data_threshold=threshold,
                )
            else:
                ir.ir_to_onnx(
                    native_path,
                    payload,
                    opset=18,
                    model_name="py_oracle_case",
                )
                self._python_build_onnx_from_stub(stub, python_path)

            native_summary = self._onnx_model_summary(native_path)
            python_summary = self._onnx_model_summary(python_path)
            self.assertEqual(python_summary, native_summary)

    def test_export_ir_json_matches_export_ir(self):
        fun = self._sample_fun()
        x, y = self._sample_inputs()
        payload = ir.export_ir(fun, x, y=y)
        from_json = json.loads(ir.export_ir_json(fun, x, y=y))
        self.assertEqual(payload, from_json)

    def test_export_ir_is_deterministic(self):
        fun = self._sample_fun()
        x, y = self._sample_inputs()
        first = ir.export_ir_json(fun, x, y=y)
        second = ir.export_ir_json(fun, x, y=y)
        self.assertEqual(first, second)

    def test_export_ir_includes_keyword_inputs(self):
        fun = self._sample_fun()
        x, y = self._sample_inputs()
        payload = ir.export_ir(fun, x, y=y)
        self.assertEqual(1, len(payload["keyword_inputs"]))
        keyword = payload["keyword_inputs"][0]
        self.assertEqual("y", keyword["name"])
        self.assertIn(keyword["tensor"], [item["name"] for item in payload["inputs"]])

    def test_export_ir_includes_constant_values(self):
        fun, x, expected_values = self._constant_fun_and_input(size=2)
        payload = ir.export_ir(fun, x)
        self.assertEqual(1, len(payload["constants"]))
        constant = payload["constants"][0]
        self.assertEqual("float32", constant["dtype"])
        self.assertEqual([2], constant["shape"])
        self.assertEqual(expected_values, constant["values"])
        self.assertTrue(
            any(constant["name"] in node["inputs"] for node in payload["nodes"])
        )

    def test_ir_to_onnx_json_accepts_json_string_source(self):
        payload = self._sample_payload()
        source = json.dumps(payload)
        onnx_json = ir.ir_to_onnx_json(source, opset=18, model_name="string_source")
        stub = json.loads(onnx_json)
        self.assertEqual("onnx_stub_v1", stub["format"])
        self.assertEqual("string_source", stub["graph"]["name"])

    def test_ir_to_onnx_json_accepts_path_source(self):
        payload = self._sample_payload()
        source_path = self._path("source_ir.json")
        self._write_json_file(source_path, payload)
        onnx_json = ir.ir_to_onnx_json(source_path, opset=18, model_name="path_source")
        stub = json.loads(onnx_json)
        self.assertEqual("onnx_stub_v1", stub["format"])
        self.assertEqual("path_source", stub["graph"]["name"])

    def test_ir_to_onnx_json_rejects_malformed_json(self):
        with self.assertRaises(ValueError) as error:
            ir.ir_to_onnx_json("{not-json")
        self.assertIn("[ir.api] failed to parse graph ir string", str(error.exception))

    def test_ir_to_onnx_json_rejects_malformed_payload(self):
        payload = self._sample_payload()
        payload.pop("nodes")
        with self.assertRaises(RuntimeError) as error:
            ir.ir_to_onnx_json(payload)
        self.assertRegex(str(error.exception), r"(nodes|out_of_range|missing)")

    def test_ir_to_onnx_json_rejects_malformed_constants(self):
        payload = self._sample_payload()
        payload["constants"] = [{"name": "c", "shape": [1], "dtype": "float32"}]
        with self.assertRaises(RuntimeError) as error:
            ir.ir_to_onnx_json(payload)
        self.assertRegex(str(error.exception), r"(values|constants)")

    def test_ir_to_onnx_json_rejects_unknown_op_as_not_implemented(self):
        payload = self._sample_payload()
        payload["nodes"][0]["op"] = "FutureCustomFusionOp"
        with self.assertRaises(NotImplementedError) as error:
            ir.ir_to_onnx_json(payload, opset=18, model_name="unsupported")
        self.assertRegex(str(error.exception), r"(unsupported|FutureCustomFusionOp)")

    def test_ir_to_onnx_json_lowers_random_bits(self):
        payload = {
            "ir_version": 1,
            "shapeless": False,
            "inputs": [{"name": "key", "shape": [2], "dtype": "uint32"}],
            "keyword_inputs": [],
            "outputs": [{"name": "bits", "shape": [2, 3], "dtype": "uint32"}],
            "constants": [],
            "nodes": [
                {
                    "op": "RandomBits",
                    "inputs": ["key"],
                    "outputs": ["bits"],
                    "arguments": [[2, 3], 4],
                }
            ],
        }
        stub = json.loads(ir.ir_to_onnx_json(payload, opset=18, model_name="randbits"))
        self.assertEqual(1, len(stub["graph"]["nodes"]))
        node = stub["graph"]["nodes"][0]
        self.assertEqual("RandomUniform", node["op_type"])
        self.assertEqual([], node["inputs"])
        self.assertEqual(["bits"], node["outputs"])
        self.assertEqual([2, 3], node["attributes"]["shape"])
        self.assertEqual(12, node["attributes"]["dtype"])  # UINT32
        self.assertEqual(0.0, node["attributes"]["low"])
        self.assertEqual(float(2**32), node["attributes"]["high"])

    def test_ir_to_onnx_json_rejects_random_bits_invalid_width(self):
        payload = {
            "ir_version": 1,
            "shapeless": False,
            "inputs": [{"name": "key", "shape": [2], "dtype": "uint32"}],
            "keyword_inputs": [],
            "outputs": [{"name": "bits", "shape": [2, 3], "dtype": "uint32"}],
            "constants": [],
            "nodes": [
                {
                    "op": "RandomBits",
                    "inputs": ["key"],
                    "outputs": ["bits"],
                    "arguments": [[2, 3], 8],
                }
            ],
        }
        with self.assertRaises(RuntimeError) as error:
            ir.ir_to_onnx_json(payload, opset=18, model_name="randbits_bad_width")
        self.assertIn("RandomBits width", str(error.exception))

    def test_ir_to_onnx_json_lowers_erf_inv(self):
        payload = {
            "ir_version": 1,
            "shapeless": False,
            "inputs": [{"name": "x", "shape": [2], "dtype": "float32"}],
            "keyword_inputs": [],
            "outputs": [{"name": "y", "shape": [2], "dtype": "float32"}],
            "constants": [],
            "nodes": [
                {
                    "op": "ErfInv",
                    "inputs": ["x"],
                    "outputs": ["y"],
                    "arguments": [],
                }
            ],
        }
        stub = json.loads(ir.ir_to_onnx_json(payload, opset=18, model_name="erfinv"))
        op_types = [node["op_type"] for node in stub["graph"]["nodes"]]
        self.assertIn("Where", op_types)
        self.assertIn("Sqrt", op_types)
        self.assertIn("Log", op_types)
        self.assertEqual("y", stub["graph"]["nodes"][-1]["outputs"][0])

    def test_ir_to_onnx_binary_rejects_constant_value_count_mismatch(self):
        payload = self._constant_payload(size=2)
        payload["constants"][0]["values"] = [1.0]
        onnx_path = self._path("value_count_mismatch.onnx")
        with self.assertRaises(RuntimeError) as error:
            ir.ir_to_onnx(onnx_path, payload, opset=18, model_name="bad_values")
        self.assertIn("expected 2", str(error.exception))

    def test_export_onnx_rejects_non_path_target(self):
        fun = self._sample_fun()
        x, y = self._sample_inputs()
        with self.assertRaises(ValueError) as error:
            ir.export_onnx(object(), fun, x, y=y, model_name="bad_target")
        self.assertIn("path-like value", str(error.exception))

    def test_ir_to_onnx_rejects_non_path_target(self):
        payload = self._sample_payload()
        with self.assertRaises(ValueError) as error:
            ir.ir_to_onnx(object(), payload, model_name="bad_target")
        self.assertIn("path-like value", str(error.exception))

    def test_export_onnx_direct_matches_ir_to_onnx_binary(self):
        fun = self._sample_fun()
        x, y = self._sample_inputs()
        payload = ir.export_ir(fun, x, y=y)

        direct_path = self._path("direct.onnx")
        from_ir_path = self._path("from_ir.onnx")
        ir.export_onnx(direct_path, fun, x, y=y, model_name="binary_parity", opset=18)
        ir.ir_to_onnx(from_ir_path, payload, model_name="binary_parity", opset=18)
        self.assertEqual(self._read_bytes(direct_path), self._read_bytes(from_ir_path))

    def test_export_onnx_from_mlx_initialized_model(self):
        class TinyMlxModel:
            def __init__(self):
                self.scale = mx.array([1.5, -2.0, 0.25], dtype=mx.float32)
                self.bias = mx.array([0.1, 0.2, -0.3], dtype=mx.float32)
                mx.eval(self.scale, self.bias)

            def __call__(self, x):
                return (x * self.scale) + self.bias

        model = TinyMlxModel()
        x = mx.array(
            [[0.25, -0.5, 1.5], [1.0, 0.0, -1.0]],
            dtype=mx.float32,
        )

        payload = ir.export_ir(model, x)
        self.assertGreaterEqual(len(payload["constants"]), 2)

        direct_path = self._path("mlx_model_direct.onnx")
        from_ir_path = self._path("mlx_model_from_ir.onnx")
        ir.export_onnx(direct_path, model, x, model_name="mlx_model", opset=18)
        ir.ir_to_onnx(from_ir_path, payload, model_name="mlx_model", opset=18)

        self.assertTrue(os.path.exists(direct_path))
        self.assertTrue(os.path.exists(from_ir_path))
        self.assertGreater(os.path.getsize(direct_path), 0)
        self.assertEqual(self._read_bytes(direct_path), self._read_bytes(from_ir_path))

    def test_export_onnx_external_data_writes_sidecar(self):
        fun, x, _ = self._constant_fun_and_input(size=4096)
        onnx_path = self._path("external_default.onnx")
        ir.export_onnx(
            onnx_path,
            fun,
            x,
            model_name="external_default",
            external_data=True,
            external_data_size_threshold=0,
        )
        sidecar = self._path("external_default.data")
        self.assertTrue(os.path.exists(onnx_path))
        self.assertTrue(os.path.exists(sidecar))
        self.assertGreater(os.path.getsize(sidecar), 0)

    def test_export_onnx_external_data_uses_custom_filename(self):
        fun, x, _ = self._constant_fun_and_input(size=4096)
        onnx_path = self._path("external_custom.onnx")
        custom_data_name = "weights.bin"
        custom_data_path = os.path.join(os.path.dirname(onnx_path), custom_data_name)
        ir.export_onnx(
            onnx_path,
            fun,
            x,
            model_name="external_custom",
            external_data=True,
            external_data_file=custom_data_name,
            external_data_size_threshold=0,
        )
        self.assertTrue(os.path.exists(onnx_path))
        self.assertTrue(os.path.exists(custom_data_path))
        self.assertGreater(os.path.getsize(custom_data_path), 0)

    def test_ir_to_onnx_external_data_writes_sidecar(self):
        payload = self._constant_payload(size=4096)
        onnx_path = self._path("ir_external.onnx")
        ir.ir_to_onnx(
            onnx_path,
            payload,
            model_name="ir_external",
            external_data=True,
            external_data_size_threshold=0,
        )
        sidecar = self._path("ir_external.data")
        self.assertTrue(os.path.exists(onnx_path))
        self.assertTrue(os.path.exists(sidecar))
        self.assertGreater(os.path.getsize(sidecar), 0)

    def test_export_onnx_compatibility_report_lists_unsupported_ops(self):
        fun = self._sample_fun()
        x, y = self._sample_inputs()
        supported = ir.export_onnx_compatibility_report(fun, x, y=y)
        self.assertEqual(0, supported["unsupported_nodes"])
        self.assertEqual([], supported["unsupported_ops"])

        payload = ir.export_ir(fun, x, y=y)
        payload["nodes"][0]["op"] = "FutureCustomFusionOp"
        unsupported = json.loads(ir.ir_compatibility_report_json(payload))
        self.assertEqual(1, unsupported["unsupported_nodes"])
        self.assertEqual(["FutureCustomFusionOp"], unsupported["unsupported_ops"])
        self.assertFalse(unsupported["ready_for_stub_conversion"])

    def test_ir_compatibility_report_json_matches_dict_report(self):
        fun = self._sample_fun()
        x, y = self._sample_inputs()
        payload = ir.export_ir(fun, x, y=y)
        dict_report = ir.export_onnx_compatibility_report(fun, x, y=y)
        from_payload = json.loads(ir.ir_compatibility_report_json(payload))
        from_json_string = json.loads(
            ir.ir_compatibility_report_json(json.dumps(payload))
        )
        self.assertEqual(dict_report, from_payload)
        self.assertEqual(dict_report, from_json_string)

    def test_onnx_runtime_parity_exp_add(self):
        fun = self._sample_fun()
        x, y = self._sample_inputs()
        payload = ir.export_ir(fun, x, y=y)
        onnx_path = self._path("runtime_exp_add.onnx")
        ir.ir_to_onnx(onnx_path, payload, model_name="runtime_exp_add")

        expected = (mx.exp(x) + y).tolist()
        feeds = {
            payload["inputs"][0]["name"]: x.tolist(),
            payload["inputs"][1]["name"]: y.tolist(),
        }
        result = self._run_onnxruntime(onnx_path, feeds)
        self._assert_nested_close(expected, result[0])

    def test_onnx_runtime_parity_with_initializer_constants(self):
        payload = self._constant_payload(size=2)
        onnx_path = self._path("runtime_initializer.onnx")
        ir.ir_to_onnx(onnx_path, payload, model_name="runtime_initializer")

        feeds = {"x": [1.0, 2.0]}
        result = self._run_onnxruntime(onnx_path, feeds)
        self._assert_nested_close([11.0, 13.0], result[0])

    def test_native_binary_matches_python_onnx_oracle_no_external_data(self):
        payload = self._sample_payload()
        self._compare_native_and_python_oracle(payload, external_data=False)

    def test_native_binary_matches_python_onnx_oracle_with_external_data(self):
        payload = self._constant_payload(size=4096)
        self._compare_native_and_python_oracle(
            payload, external_data=True, threshold=0
        )

    def test_shapeless_export_ir_and_onnx_json(self):
        def fun(x):
            return mx.reshape(mx.exp(x), (-1,))

        x = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        payload = ir.export_ir(fun, x, shapeless=True)
        self.assertTrue(payload["shapeless"])
        onnx_stub = json.loads(
            ir.ir_to_onnx_json(payload, opset=18, model_name="shapeless_case")
        )
        self.assertEqual("onnx_stub_v1", onnx_stub["format"])
        self.assertEqual("shapeless_case", onnx_stub["graph"]["name"])
        self.assertGreater(len(onnx_stub["graph"]["nodes"]), 0)


if __name__ == "__main__":
    unittest.main()
