import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import fake_triton

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model_repository"
    / "ensemble"
    / "1"
    / "model.py"
)

BATCH = 1
MAX_SEQ = 128
EMBED_DIM = 384


def _make_fake_module():
    mod = types.ModuleType("triton_python_backend_utils")
    for name in (
        "Tensor",
        "InferenceRequest",
        "InferenceResponse",
        "get_input_tensor_by_name",
        "get_output_tensor_by_name",
    ):
        setattr(mod, name, getattr(fake_triton, name))
    return mod


def _register_defaults(sim):
    sim.register(
        "text_tokenizer",
        lambda inp: {
            "input_ids": np.zeros((BATCH, MAX_SEQ), dtype=np.int64),
            "attention_mask": np.ones((BATCH, MAX_SEQ), dtype=np.int64),
            "token_type_ids": np.zeros((BATCH, MAX_SEQ), dtype=np.int64),
        },
    )
    sim.register(
        "embedding",
        lambda inp: {
            "last_hidden_state": np.random.randn(
                BATCH, MAX_SEQ, EMBED_DIM
            ).astype(np.float32),
        },
    )
    sim.register(
        "classification",
        lambda inp: {"logits": np.array([[0.8, 0.2]], dtype=np.float32)},
    )
    sim.register(
        "generation",
        lambda inp: {
            "GENERATED_TEXT": np.array([[b"mock generated answer"]], dtype=np.object_)
        },
    )


@pytest.fixture
def simulator(monkeypatch):
    fake = _make_fake_module()
    monkeypatch.setitem(sys.modules, "triton_python_backend_utils", fake)
    sim = fake_triton._ModelServer()
    fake_triton.set_simulator(sim)
    _register_defaults(sim)
    return sim


def _load_model_class():
    spec = importlib.util.spec_from_file_location("ensemble_model", MODEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.TritonPythonModel


@pytest.fixture
def model_cls(simulator):
    return _load_model_class()


@pytest.fixture
def model(model_cls):
    config = {
        "parameters": {
            "tokenizer_model": {"string_value": "text_tokenizer"},
            "embedding_model": {"string_value": "embedding"},
            "classification_model": {"string_value": "classification"},
            "generation_model": {"string_value": "generation"},
        }
    }
    import json

    instance = model_cls()
    instance.initialize({"model_config": json.dumps(config)})
    return instance


def build_request(text):
    arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).reshape(1, -1)
    return fake_triton.InferenceRequest(
        model_name="ensemble", inputs=[fake_triton.Tensor("TEXT", arr)]
    )


def decode_response(response):
    tensor = fake_triton.get_output_tensor_by_name(response, "RESPONSE")
    raw = tensor.as_numpy()[0][0]
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw