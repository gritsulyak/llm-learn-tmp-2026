import numpy as np
import pytest

from conftest import build_request, decode_response


def test_statement_routes_to_classification(model, simulator):
    sim_calls = []

    original_run = simulator.run

    def spy(model_name, input_dict):
        sim_calls.append(model_name)
        return original_run(model_name, input_dict)

    simulator.run = spy

    response = model.execute([build_request("This is a plain statement.")])[0]
    text = decode_response(response)

    assert text.startswith("[Classification]")
    assert "label=0" in text
    assert "generation" not in sim_calls


def test_question_routes_to_generation(model, simulator):
    simulator.register(
        "classification",
        lambda inp: {"logits": np.array([[0.05, 0.95]], dtype=np.float32)},
    )

    response = model.execute([build_request("What is deep learning?")])[0]
    text = decode_response(response)

    assert text.startswith("[Generated]")
    assert "mock generated answer" in text
    assert "generation" in simulator.call_log


def test_embedding_is_called(model, simulator):
    simulator.register(
        "classification",
        lambda inp: {"logits": np.array([[0.05, 0.95]], dtype=np.float32)},
    )

    model.execute([build_request("How does Triton work?")])

    assert "embedding" in simulator.call_log
    assert "text_tokenizer" in simulator.call_log


def test_generation_receives_text(model, simulator):
    simulator.register(
        "classification",
        lambda inp: {"logits": np.array([[0.05, 0.95]], dtype=np.float32)},
    )
    received = {}

    def gen_handler(inp):
        received["text"] = inp["TEXT"]
        return {"GENERATED_TEXT": np.array([[b"answer"]], dtype=np.object_)}

    simulator.register("generation", gen_handler)

    model.execute([build_request("What is the capital of France?")])

    assert received["text"].shape[0] == 1
    assert received["text"][0].tobytes() == b"What is the capital of France?"


def test_multiple_requests_batch(model, simulator):
    simulator.register(
        "classification",
        lambda inp: {"logits": np.array([[0.05, 0.95]], dtype=np.float32)},
    )

    reqs = [
        build_request("What is AI?"),
        build_request("How does an LLM work?"),
    ]
    responses = model.execute(reqs)

    assert len(responses) == 2
    for resp in responses:
        assert decode_response(resp).startswith("[Generated]")


def test_missing_dependency_model_raises(model, simulator):
    simulator.unregister("embedding")

    with pytest.raises(RuntimeError, match="Error calling embedding"):
        model.execute([build_request("This should fail.")])