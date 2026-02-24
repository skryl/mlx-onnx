import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_onnx as ir


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "mlx-examples"

if not EXAMPLES_ROOT.is_dir():
    pytest.skip(
        "mlx-examples submodule is required for examples export tests",
        allow_module_level=True,
    )


@contextlib.contextmanager
def _temporary_sys_path(paths):
    old = list(sys.path)
    sys.path[:0] = [str(path) for path in paths]
    try:
        yield
    finally:
        sys.path[:] = old


@contextlib.contextmanager
def _temporary_modules(stubs):
    marker = object()
    previous = {}
    for name, module in stubs.items():
        previous[name] = sys.modules.get(name, marker)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is marker:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _load_module(module_name, file_path, extra_sys_path=(), stubs=None):
    if module_name in sys.modules:
        return sys.modules[module_name]

    stubs = stubs or {}
    with _temporary_sys_path(extra_sys_path), _temporary_modules(stubs):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot create module spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module


def _load_package_module(package_name, package_dir, module_name, stubs=None):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    full_name = f"{package_name}.{module_name}"
    return _load_module(full_name, package_dir / f"{module_name}.py", stubs=stubs)


def _transformers_stub():
    module = types.ModuleType("transformers")

    class _PreTrainedTokenizerBase:
        pass

    class _AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None

    class _Tokenizer:
        eos_token_id = 1

        def __call__(self, *args, **kwargs):
            return {"input_ids": [[0]], "attention_mask": [[1]]}

        def convert_ids_to_tokens(self, token_ids):
            return [str(token_id) for token_id in token_ids]

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _Tokenizer()

    module.AutoConfig = _AutoConfig
    module.AutoTokenizer = _AutoTokenizer
    module.PreTrainedTokenizerBase = _PreTrainedTokenizerBase
    return module


def _sentencepiece_stub():
    module = types.ModuleType("sentencepiece")

    class _SentencePieceProcessor:
        def __init__(self, *args, **kwargs):
            pass

    module.SentencePieceProcessor = _SentencePieceProcessor
    return module


def _tiktoken_stub():
    module = types.ModuleType("tiktoken")

    class _Encoding:
        def __init__(self, *args, **kwargs):
            pass

    module.Encoding = _Encoding
    return module


def _musicgen_stubs():
    encodec_module = types.ModuleType("encodec")

    class _EncodecModel:
        pass

    encodec_module.EncodecModel = _EncodecModel

    t5_module = types.ModuleType("t5")

    class _T5:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("T5.from_pretrained should not be called in this test")

    t5_module.T5 = _T5

    tqdm_module = types.ModuleType("tqdm")

    def _passthrough(iterable, *args, **kwargs):
        return iterable

    tqdm_module.tqdm = _passthrough

    return {
        "encodec": encodec_module,
        "t5": t5_module,
        "tqdm": tqdm_module,
    }


def _eval_module_parameters(module):
    if hasattr(module, "parameters"):
        mx.eval(module.parameters())


def _require_onnxruntime():
    pytest.importorskip("numpy")
    return pytest.importorskip("onnxruntime")


def _to_numpy_outputs(value):
    import numpy as np

    if isinstance(value, (list, tuple)):
        return [np.asarray(item) for item in value]
    return [np.asarray(value)]


def _assert_outputs_close(expected_outputs, actual_outputs):
    import numpy as np

    assert len(expected_outputs) == len(actual_outputs), (
        f"output count mismatch: expected {len(expected_outputs)}, got {len(actual_outputs)}"
    )
    for expected, actual in zip(expected_outputs, actual_outputs):
        expected_arr = np.asarray(expected)
        actual_arr = np.asarray(actual)
        if expected_arr.dtype == np.bool_ or np.issubdtype(expected_arr.dtype, np.integer):
            np.testing.assert_array_equal(expected_arr, actual_arr)
            continue
        if expected_arr.dtype == np.float16:
            np.testing.assert_allclose(expected_arr, actual_arr, rtol=5e-3, atol=5e-3)
            continue
        np.testing.assert_allclose(expected_arr, actual_arr, rtol=1e-3, atol=1e-4)


def _build_bert():
    module = _load_module(
        "_mlx_examples_bert_model",
        EXAMPLES_ROOT / "bert" / "model.py",
        stubs={"transformers": _transformers_stub()},
    )
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        type_vocab_size=2,
        max_position_embeddings=16,
        layer_norm_eps=1e-5,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
    )
    model = module.BertEmbeddings(config)
    _eval_module_parameters(model)
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    token_type_ids = mx.zeros_like(input_ids)
    return model, (input_ids, token_type_ids)


def _build_cifar():
    module = _load_module(
        "_mlx_examples_cifar_resnet",
        EXAMPLES_ROOT / "cifar" / "resnet.py",
    )
    model = module.ShortcutA(dims=16)
    x = mx.random.normal((1, 8, 8, 8))
    return model, (x,)


def _build_clip():
    module = _load_module(
        "_mlx_examples_clip_model",
        EXAMPLES_ROOT / "clip" / "model.py",
    )
    config = module.CLIPTextConfig(
        num_hidden_layers=1,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        max_position_embeddings=16,
        vocab_size=64,
        layer_norm_eps=1e-5,
    )
    model = module.ClipTextModel(config)
    _eval_module_parameters(model)
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    def forward(x):
        return model(x).pooler_output

    return forward, (tokens,)


def _build_cvae():
    module = _load_module(
        "_mlx_examples_cvae_vae",
        EXAMPLES_ROOT / "cvae" / "vae.py",
    )
    model = module.UpsamplingConv2d(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    _eval_module_parameters(model)
    x = mx.random.normal((1, 4, 4, 2))
    return model, (x,)


def _build_encodec():
    try:
        _load_module(
            "_mlx_examples_encodec",
            EXAMPLES_ROOT / "encodec" / "encodec.py",
        )
    except RuntimeError as exc:
        message = str(exc)
        if "No Metal back-end" in message:
            pytest.skip("encodec example requires MLX Metal backend")
        raise
    model = nn.Identity()
    x = mx.random.normal((1, 8, 4))
    return model, (x,)


def _build_flux():
    module = _load_module(
        "_mlx_examples_flux_layers",
        EXAMPLES_ROOT / "flux" / "flux" / "layers.py",
    )
    model = module.EmbedND(dim=6, theta=10000, axes_dim=[2, 2, 2])
    ids = mx.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]], dtype=mx.int32)
    return model, (ids,)


def _build_gcn():
    module = _load_module(
        "_mlx_examples_gcn",
        EXAMPLES_ROOT / "gcn" / "gcn.py",
    )
    model = module.GCNLayer(in_features=4, out_features=2, bias=True)
    _eval_module_parameters(model)
    x = mx.random.normal((5, 4))
    adj = mx.eye(5)
    return model, (x, adj)


def _build_llava():
    module = _load_module(
        "_mlx_examples_llava_vision",
        EXAMPLES_ROOT / "llava" / "vision.py",
    )
    config = module.VisionConfig(
        model_type="clip_vision_model",
        num_hidden_layers=1,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        image_size=8,
        patch_size=4,
        num_channels=3,
        layer_norm_eps=1e-5,
    )
    model = module.VisionEmbeddings(config)
    _eval_module_parameters(model)
    x = mx.random.normal((1, 8, 8, 3))
    return model, (x,)


def _build_llms():
    module = _load_module(
        "_mlx_examples_llama",
        EXAMPLES_ROOT / "llms" / "llama" / "llama.py",
        stubs={"sentencepiece": _sentencepiece_stub()},
    )
    args = module.ModelArgs(
        dim=16,
        n_layers=1,
        head_dim=4,
        hidden_dim=32,
        n_heads=4,
        n_kv_heads=4,
        norm_eps=1e-5,
        vocab_size=64,
        rope_theta=10000.0,
        rope_traditional=True,
    )
    model = module.Attention(args)
    _eval_module_parameters(model)
    x = mx.random.normal((1, 4, args.dim))

    def forward(x_):
        y, _ = model(x_)
        return y

    return forward, (x,)


def _build_lora():
    module = _load_module(
        "_mlx_examples_lora_models",
        EXAMPLES_ROOT / "lora" / "models.py",
    )
    args = module.ModelArgs(
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=64,
    )
    model = module.Attention(args)
    _eval_module_parameters(model)
    x = mx.random.normal((1, 4, args.hidden_size))

    def forward(x_):
        y, _ = model(x_)
        return y

    return forward, (x,)


def _build_mnist():
    mnist_dir = EXAMPLES_ROOT / "mnist"
    module = _load_module(
        "_mlx_examples_mnist_main",
        mnist_dir / "main.py",
        extra_sys_path=(mnist_dir,),
    )
    model = module.MLP(num_layers=1, input_dim=16, hidden_dim=8, output_dim=4)
    _eval_module_parameters(model)
    x = mx.random.normal((2, 16))

    def forward(x_):
        return model.layers[0](x_)

    return forward, (x,)


def _build_musicgen():
    module = _load_module(
        "_mlx_examples_musicgen",
        EXAMPLES_ROOT / "musicgen" / "musicgen.py",
        stubs=_musicgen_stubs(),
    )
    positions = mx.array([[[0.0], [1.0], [2.0], [3.0]]], dtype=mx.float32)

    def forward(pos):
        return module.create_sin_embedding(pos, dim=8)

    return forward, (positions,)


def _build_normalizing_flow():
    flow_dir = EXAMPLES_ROOT / "normalizing_flow"
    module = _load_module(
        "_mlx_examples_normalizing_flow_distributions",
        flow_dir / "distributions.py",
        extra_sys_path=(flow_dir,),
    )
    model = module.Normal(mu=mx.zeros((4,)), sigma=mx.ones((4,)))
    x = mx.random.normal((1, 4))
    return model.log_prob, (x,)


def _build_segment_anything():
    package_dir = EXAMPLES_ROOT / "segment_anything" / "segment_anything"
    module = _load_package_module(
        "_mlx_examples_segment_anything_pkg",
        package_dir,
        "prompt_encoder",
    )
    model = module.PositionEmbeddingRandom(num_pos_feats=4)
    mx.eval(model.positional_embedding)
    coords = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)

    def forward(c):
        return model.forward_with_coords(c, image_size=(8, 8))

    return forward, (coords,)


def _build_speechcommands():
    module = _load_module(
        "_mlx_examples_speechcommands_kwt",
        EXAMPLES_ROOT / "speechcommands" / "kwt.py",
    )
    model = module.Attention(dim=8, heads=2, dropout=0.0)
    _eval_module_parameters(model)
    x = mx.random.normal((1, 4, 8))
    return model, (x,)


def _build_stable_diffusion():
    package_dir = EXAMPLES_ROOT / "stable_diffusion" / "stable_diffusion"
    module = _load_package_module(
        "_mlx_examples_stable_diffusion_pkg",
        package_dir,
        "unet",
    )
    x = mx.random.normal((1, 4, 4, 2))

    def forward(x_):
        return module.upsample_nearest(x_, scale=2)

    return forward, (x,)


def _build_t5():
    module = _load_module(
        "_mlx_examples_t5",
        EXAMPLES_ROOT / "t5" / "t5.py",
        stubs={"transformers": _transformers_stub()},
    )
    config = SimpleNamespace(d_model=16, vocab_size=32)
    model = module.OutputHead(config)
    _eval_module_parameters(model)
    x = mx.random.normal((1, 4, 16))
    return model, (x,)


def _build_transformer_lm():
    lm_dir = EXAMPLES_ROOT / "transformer_lm"
    module = _load_module(
        "_mlx_examples_transformer_lm",
        lm_dir / "main.py",
        extra_sys_path=(lm_dir,),
    )
    model = module.TransformerLM(
        vocab_size=32,
        num_layers=1,
        dims=16,
        num_heads=4,
        checkpoint=False,
    )
    _eval_module_parameters(model)
    tokens = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)

    def forward(t):
        return model.embedding(t)

    return forward, (tokens,)


def _build_whisper():
    package_dir = EXAMPLES_ROOT / "whisper" / "mlx_whisper"
    module = _load_package_module(
        "_mlx_examples_whisper_pkg",
        package_dir,
        "whisper",
        stubs={"tiktoken": _tiktoken_stub()},
    )
    model = module.MultiHeadAttention(n_state=16, n_head=4)
    x = mx.random.normal((1, 5, 16))

    def forward(x_):
        out, _, _ = model(x_)
        return out

    return forward, (x,)


EXAMPLE_BUILDERS = {
    "bert": _build_bert,
    "cifar": _build_cifar,
    "clip": _build_clip,
    "cvae": _build_cvae,
    "encodec": _build_encodec,
    "flux": _build_flux,
    "gcn": _build_gcn,
    "llava": _build_llava,
    "llms": _build_llms,
    "lora": _build_lora,
    "mnist": _build_mnist,
    "musicgen": _build_musicgen,
    "normalizing_flow": _build_normalizing_flow,
    "segment_anything": _build_segment_anything,
    "speechcommands": _build_speechcommands,
    "stable_diffusion": _build_stable_diffusion,
    "t5": _build_t5,
    "transformer_lm": _build_transformer_lm,
    "whisper": _build_whisper,
}

SKIPPED_EXAMPLE_DIRS = {"wwdc25"}


def _example_dirs():
    return {
        path.name
        for path in EXAMPLES_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }


def test_examples_directory_coverage():
    expected = set(EXAMPLE_BUILDERS) | SKIPPED_EXAMPLE_DIRS
    assert _example_dirs() == expected


@pytest.mark.parametrize("example_name", sorted(EXAMPLE_BUILDERS))
def test_examples_export_to_onnx(example_name, tmp_path):
    import numpy as np

    target, args = EXAMPLE_BUILDERS[example_name]()
    expected_value = target(*args)
    mx.eval(expected_value)
    expected_outputs = _to_numpy_outputs(expected_value)

    onnx_path = tmp_path / f"{example_name}.onnx"
    ir.export_onnx(
        onnx_path,
        target,
        *args,
        model_name=f"mlx_examples_{example_name}",
        opset=18,
    )
    assert onnx_path.is_file()
    assert onnx_path.stat().st_size > 0

    ort = _require_onnxruntime()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    session_inputs = session.get_inputs()
    assert len(session_inputs) == len(args), (
        f"input count mismatch for {example_name}: ONNX expects {len(session_inputs)} "
        f"inputs, example builder returned {len(args)} args"
    )
    runtime_feeds = {
        input_info.name: np.asarray(arg)
        for input_info, arg in zip(session_inputs, args)
    }
    actual_outputs = session.run(None, runtime_feeds)
    _assert_outputs_close(expected_outputs, actual_outputs)
