#!/usr/bin/env python3
"""Sustained-load tester for a live Triton server.

Alternative to NVIDIA Perf Analyzer that only needs `tritonclient` (+ `transformers`
for the tokenized models). Measures throughput (RPS) and latency percentiles at a
given concurrency for every model in the repository.

Usage:
    uv run python perf/load_test.py --model ensemble --concurrency 4 --duration 20
    uv run python perf/load_test.py --all          # test all 5 models sequentially
    uv run python perf/load_test.py --model generation --requests 30  # fixed count
"""

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = PROJECT_ROOT / "model_repository" / "classification" / "1"

STRING_MODELS = {
    "ensemble": {"input": "TEXT", "outputs": ["RESPONSE"]},
    "generation": {"input": "TEXT", "outputs": ["GENERATED_TEXT"]},
    "text_tokenizer": {"input": "TEXT", "outputs": ["input_ids"]},
    "classification": None,
    "embedding": None,
}

TOKENIZED_MODELS = {
    "classification": {"outputs": ["logits"]},
    "embedding": {"outputs": ["last_hidden_state"]},
}

DEFAULT_TEXT = "What is machine learning and how does it work?"


def make_infer_input(name, data, dtype):
    infer_input = grpcclient.InferInput(name, list(data.shape), dtype)
    infer_input.set_data_from_numpy(data)
    return infer_input


def build_requests(client, model, text, tokenizer=None):
    """Return (model_name, inputs, outputs) for one inference call."""
    if model in STRING_MODELS and STRING_MODELS[model] is not None:
        spec = STRING_MODELS[model]
        arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).reshape(1, -1)
        inputs = [make_infer_input(spec["input"], arr, "UINT8")]
        return model, inputs, [grpcclient.InferRequestedOutput(o) for o in spec["outputs"]]

    if tokenizer is None:
        raise SystemExit(f"Model {model!r} needs a tokenizer (import transformers).")

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",
    )
    names = ["input_ids", "attention_mask", "token_type_ids"]
    inputs = [
        make_infer_input(n, encoded[n].astype(np.int64), "INT64") for n in names
    ]
    outputs = [
        grpcclient.InferRequestedOutput(o) for o in TOKENIZED_MODELS[model]["outputs"]
    ]
    return model, inputs, outputs


class LoadTest:
    def __init__(self, client, model, text, tokenizer=None):
        self.client = client
        self.model = model
        self.text = text
        self.tokenizer = tokenizer
        self._latencies = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _record(self, seconds):
        with self._lock:
            self._latencies.append(seconds)

    def _send(self):
        model, inputs, outputs = build_requests(
            self.client, self.model, self.text, self.tokenizer
        )
        start = time.perf_counter()
        self.client.infer(model, inputs, outputs=outputs, timeout=300)
        self._record(time.perf_counter() - start)

    def _worker(self):
        while not self._stop.is_set():
            self._send()

    def _worker_fixed(self, total):
        for _ in range(total):
            if self._stop.is_set():
                break
            self._send()

    def run(self, concurrency=1, duration=10, requests=0, warmup=2):
        time.sleep(warmup)

        if requests > 0:
            step = max(1, requests // concurrency)
            threads = [
                threading.Thread(target=self._worker_fixed, args=(step,))
                for _ in range(concurrency)
            ]
        else:
            threads = [
                threading.Thread(target=self._worker) for _ in range(concurrency)
            ]

        started = time.time()
        for t in threads:
            t.start()

        if requests <= 0:
            deadline = started + duration
            while time.time() < deadline:
                time.sleep(0.05)
            self._stop.set()

        for t in threads:
            t.join()
        elapsed = time.time() - started
        self._stop.clear()

        with self._lock:
            lat = sorted(self._latencies)

        return {
            "model": self.model,
            "concurrency": concurrency,
            "elapsed_s": round(elapsed, 2),
            "requests": len(lat),
            "rps": round(len(lat) / elapsed, 2) if elapsed > 0 else 0.0,
            "mean_ms": round(statistics.mean(lat) * 1000, 2) if lat else 0.0,
            "p50_ms": round(lat[len(lat) // 2] * 1000, 2) if lat else 0.0,
            "p95_ms": round(lat[int(len(lat) * 0.95)] * 1000, 2) if lat else 0.0,
            "p99_ms": round(lat[int(len(lat) * 0.99)] * 1000, 2) if lat else 0.0,
        }


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    if TOKENIZER_DIR.exists():
        return AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    return None


def print_result(result):
    print(
        f"  {result['model']:<15} concurrency={result['concurrency']:<2} "
        f"reqs={result['requests']:<6} rps={result['rps']:<8} "
        f"mean={result['mean_ms']:<8} p50={result['p50_ms']:<8} "
        f"p95={result['p95_ms']:<8} p99={result['p99_ms']} ms"
    )


def main():
    parser = argparse.ArgumentParser(description="Triton load test")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", help="model name to test")
    parser.add_argument("--all", action="store_true", help="test every model")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--requests", type=int, default=0, help="fixed request count")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text used as input")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", help="JSON path to save results")
    args = parser.parse_args()

    if not (args.model or args.all):
        parser.error("provide --model or --all")

    url = f"{args.host}:{args.port}"
    client = grpcclient.InferenceServerClient(url=url, verbose=False)
    if not client.is_server_live():
        raise SystemExit(f"Triton server not reachable at {url}")

    tokenizer = load_tokenizer()
    if args.model in TOKENIZED_MODELS and tokenizer is None:
        raise SystemExit("Tokenized models require `transformers` (uv sync).")

    models = []
    if args.all:
        models = [
            "text_tokenizer",
            "classification",
            "embedding",
            "generation",
            "ensemble",
        ]
    else:
        models = [args.model]

    results = []
    for model in models:
        test = LoadTest(client, model, args.text, tokenizer)
        result = test.run(
            concurrency=args.concurrency,
            duration=args.duration,
            requests=args.requests,
            warmup=args.warmup,
        )
        results.append(result)
        print_result(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()