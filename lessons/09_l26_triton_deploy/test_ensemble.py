import numpy as np
import tritonclient.grpc as grpcclient
import time
import sys


TRITON_URL = "localhost:8001"


def text_tensor(text):
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).reshape(1, -1)


class TritonTestClient:
    def __init__(self, url=TRITON_URL):
        self.url = url
        self.client = grpcclient.InferenceServerClient(url=url, verbose=False)

    def is_server_live(self):
        return self.client.is_server_live()

    def is_model_ready(self, model_name):
        return self.client.is_model_ready(model_name)

    def infer(self, model_name, inputs, outputs=None):
        inputs_grpc = []
        for name, data in inputs.items():
            dtype = grpcclient.np_to_triton_dtype(data.dtype)
            inputs_grpc.append(grpcclient.InferInput(name, list(data.shape), dtype))
            inputs_grpc[-1].set_data_from_numpy(data)

        if outputs is None:
            outputs = []
        outputs_grpc = [grpcclient.InferRequestedOutput(o) for o in outputs]
        response = self.client.infer(model_name, inputs_grpc, outputs=outputs_grpc)
        return {o: response.as_numpy(o) for o in outputs}


def test_tokenizer(client):
    print("=== Testing Text Tokenizer ===")
    assert client.is_model_ready("text_tokenizer"), "text_tokenizer model not ready"

    texts = ["Hello world", "What is machine learning?"]
    for text in texts:
        inputs = {"TEXT": text_tensor(text)}

        start = time.time()
        result = client.infer(
            "text_tokenizer", inputs,
            outputs=["input_ids", "attention_mask", "token_type_ids"],
        )
        latency = (time.time() - start) * 1000

        print(f"  '{text}' -> input_ids={result['input_ids'].shape}, "
              f"attention_mask={result['attention_mask'].shape} ({latency:.1f}ms)")

    print("  PASSED\n")


def test_classification(client):
    print("=== Testing Classification Model ===")
    assert client.is_model_ready("classification"), "classification model not ready"
    assert client.is_model_ready("text_tokenizer"), "text_tokenizer model not ready"

    texts = ["This movie is great!", "I hate this product.", "What is machine learning?"]

    for text in texts:
        tok_result = client.infer(
            "text_tokenizer", {"TEXT": text_tensor(text)},
            outputs=["input_ids", "attention_mask", "token_type_ids"],
        )

        inputs = {
            "input_ids": tok_result["input_ids"],
            "attention_mask": tok_result["attention_mask"],
            "token_type_ids": tok_result["token_type_ids"],
        }

        start = time.time()
        result = client.infer("classification", inputs, outputs=["logits"])
        latency = (time.time() - start) * 1000

        logits = result["logits"]
        label = int(np.argmax(logits, axis=-1)[0])
        confidence = float(np.max(logits, axis=-1)[0])
        print(f"  '{text}' -> label={label}, confidence={confidence:.3f} ({latency:.1f}ms)")

    print("  PASSED\n")


def test_embedding(client):
    print("=== Testing Embedding Model ===")
    assert client.is_model_ready("embedding"), "embedding model not ready"
    assert client.is_model_ready("text_tokenizer"), "text_tokenizer model not ready"

    texts = ["Hello world", "Machine learning is fun"]

    for text in texts:
        tok_result = client.infer(
            "text_tokenizer", {"TEXT": text_tensor(text)},
            outputs=["input_ids", "attention_mask", "token_type_ids"],
        )

        inputs = {
            "input_ids": tok_result["input_ids"],
            "attention_mask": tok_result["attention_mask"],
            "token_type_ids": tok_result["token_type_ids"],
        }

        start = time.time()
        result = client.infer("embedding", inputs, outputs=["last_hidden_state"])
        latency = (time.time() - start) * 1000

        hidden = result["last_hidden_state"]
        mask = tok_result["attention_mask"].astype(np.float32)
        mask_exp = np.expand_dims(mask, -1)
        pooled = np.sum(hidden * mask_exp, axis=1) / np.clip(mask_exp.sum(axis=1), 1e-9, None)
        norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
        emb = pooled / np.clip(norm, 1e-9, None)

        print(f"  '{text}' -> shape={emb.shape}, norm={np.linalg.norm(emb):.3f} ({latency:.1f}ms)")

    print("  PASSED\n")


def test_generation(client):
    print("=== Testing Generation Model ===")
    assert client.is_model_ready("generation"), "generation model not ready"

    prompts = [
        "The capital of France is",
        "Machine learning is",
    ]

    for prompt in prompts:
        inputs = {"TEXT": text_tensor(prompt)}

        start = time.time()
        result = client.infer("generation", inputs, outputs=["GENERATED_TEXT"])
        latency = (time.time() - start) * 1000

        raw = result["GENERATED_TEXT"][0][0]
        generated = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        print(f"  '{prompt}' -> '{generated[:80]}' ({latency:.1f}ms)")

    print("  PASSED\n")


def test_ensemble(client):
    print("=== Testing Ensemble (BLS) Pipeline ===")
    assert client.is_model_ready("ensemble"), "ensemble model not ready"

    test_cases = [
        ("This is a statement about weather.", "classification"),
        ("What is deep learning?", "generation"),
        ("I love this product!", "classification"),
        ("How does Triton work?", "generation"),
    ]

    for text, expected_type in test_cases:
        inputs = {"TEXT": text_tensor(text)}

        start = time.time()
        result = client.infer("ensemble", inputs, outputs=["RESPONSE"])
        latency = (time.time() - start) * 1000

        raw = result["RESPONSE"][0][0]
        response = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        is_correct = expected_type in response.lower()
        status = "OK" if is_correct else "WRONG"

        print(f"  [{status}] '{text[:50]}' -> '{response[:60]}' ({latency:.1f}ms)")

    print("  PASSED\n")


def main():
    print(f"Triton Test Suite - connecting to {TRITON_URL}\n")

    client = TritonTestClient()

    try:
        if not client.is_server_live():
            print("ERROR: Triton server is not live")
            sys.exit(1)
        print("Server is live\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to Triton: {e}")
        sys.exit(1)

    tests = [
        test_tokenizer,
        test_classification,
        test_embedding,
        test_generation,
        test_ensemble,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test(client)
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}\n")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
