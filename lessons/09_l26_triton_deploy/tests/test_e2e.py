"""End-to-end tests for Part 3: Docker deployment + BLS pipeline on a live server.

Requires a reachable Triton server. Two ways to provide one:
  - an already running Triton at TRITON_HOST:TRITON_GRPC_PORT (default localhost:8001)
  - set TRITON_START_DOCKER=1 to bring the stack up via docker-compose (and tear it down after)

If neither is available the tests skip gracefully.
"""

import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HOST = os.environ.get("TRITON_HOST", "localhost")
GRPC_PORT = int(os.environ.get("TRITON_GRPC_PORT", "8001"))
HTTP_PORT = int(os.environ.get("TRITON_HTTP_PORT", "8000"))
URL = f"{HOST}:{GRPC_PORT}"

READY_TIMEOUT = int(os.environ.get("TRITON_READY_TIMEOUT", "300"))
POLL_INTERVAL = 2

EXPECTED_MODELS = [
    "text_tokenizer",
    "classification",
    "embedding",
    "generation",
    "ensemble",
]

STATEMENT_TEXT = "I love this product."
QUESTION_TEXT = "What is machine learning?"


def port_open(host, port, timeout=2):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_server(client, timeout=READY_TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if client.is_server_live():
                return True
        except InferenceServerException:
            pass
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)
    return False


def wait_model_ready(client, model, timeout=READY_TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if client.is_model_ready(model):
                return True
        except (InferenceServerException, Exception):
            pass
        time.sleep(POLL_INTERVAL)
    return False


def make_text_input(name, text):
    arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).reshape(1, -1)
    infer_input = grpcclient.InferInput(name, list(arr.shape), "UINT8")
    infer_input.set_data_from_numpy(arr)
    return infer_input


def decode_cell(cell):
    return cell.decode("utf-8") if isinstance(cell, bytes) else cell


class _DockerStack:
    def __init__(self):
        self.started = False

    def up(self):
        try:
            subprocess.run(
                ["docker", "compose", "-f", "docker-compose.yaml", "up", "-d", "--wait"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                timeout=600,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
        self.started = True
        return True

    def down(self):
        if not self.started:
            return
        subprocess.run(
            ["docker", "compose", "-f", "docker-compose.yaml", "down", "-v"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            timeout=120,
        )
        self.started = False


@pytest.fixture(scope="module")
def triton_client():
    stack = _DockerStack()

    if port_open(HOST, GRPC_PORT):
        client = grpcclient.InferenceServerClient(url=URL, verbose=False)
        if wait_server(client):
            yield client
            return

    if os.environ.get("TRITON_START_DOCKER") == "1" and stack.up():
        time.sleep(POLL_INTERVAL)
        client = grpcclient.InferenceServerClient(url=URL, verbose=False)
        if wait_server(client):
            try:
                yield client
            finally:
                stack.down()
            return

    pytest.skip(
        f"Triton server not reachable at {URL}. Run it (docker compose up -d) "
        "or set TRITON_START_DOCKER=1 to start automatically."
    )


@pytest.mark.e2e
def test_server_is_live(triton_client):
    assert triton_client.is_server_live()


@pytest.mark.e2e
def test_all_models_reach_ready(triton_client):
    for model in EXPECTED_MODELS:
        assert wait_model_ready(triton_client, model), f"{model} not READY in time"


@pytest.mark.e2e
def test_ensemble_single_request_bls_pipeline(triton_client):
    """One request through the whole chain: tokenizer -> embedding -> classifier -> (generation)."""
    inputs = [make_text_input("TEXT", QUESTION_TEXT)]
    outputs = [grpcclient.InferRequestedOutput("RESPONSE")]
    result = triton_client.infer("ensemble", inputs, outputs=outputs, timeout=120)

    raw = result.as_numpy("RESPONSE")[0][0]
    response = decode_cell(raw)

    assert response, "Empty response from BLS pipeline"
    assert response.startswith(("[Generated]", "[Classification]")), (
        f"Unexpected response format: {response!r}"
    )


@pytest.mark.e2e
def test_generation_model_direct(triton_client):
    inputs = [make_text_input("TEXT", "The capital of France is")]
    outputs = [grpcclient.InferRequestedOutput("GENERATED_TEXT")]
    result = triton_client.infer("generation", inputs, outputs=outputs, timeout=120)

    raw = result.as_numpy("GENERATED_TEXT")[0][0]
    generated = decode_cell(raw)
    assert generated, "Empty generation output"


@pytest.mark.e2e
def test_classification_and_embedding_chain(triton_client):
    """tokenizer -> classification and tokenizer -> embedding via direct model calls."""
    model_bits = {
        "classification": ("logits", None),
        "embedding": ("last_hidden_state", None),
    }

    inputs = [make_text_input("TEXT", STATEMENT_TEXT)]
    tok = triton_client.infer(
        "text_tokenizer",
        inputs,
        outputs=[
            grpcclient.InferRequestedOutput("input_ids"),
            grpcclient.InferRequestedOutput("attention_mask"),
            grpcclient.InferRequestedOutput("token_type_ids"),
        ],
        timeout=60,
    )

    token_inputs = [
        grpcclient.InferInput("input_ids", [1, 128], "INT64"),
        grpcclient.InferInput("attention_mask", [1, 128], "INT64"),
        grpcclient.InferInput("token_type_ids", [1, 128], "INT64"),
    ]
    for t_input, name in zip(token_inputs, ["input_ids", "attention_mask", "token_type_ids"]):
        t_input.set_data_from_numpy(tok.as_numpy(name).astype(np.int64))

    for model, (out_name, _) in model_bits.items():
        result = triton_client.infer(
            model, token_inputs, outputs=[grpcclient.InferRequestedOutput(out_name)], timeout=60
        )
        data = result.as_numpy(out_name)
        assert data is not None and data.size > 0, f"{model} produced no output"