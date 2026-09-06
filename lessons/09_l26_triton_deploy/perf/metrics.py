#!/usr/bin/env python3
"""Fetch and summarize Triton Prometheus metrics (HTTP port 8002).

Reports per-model: inference count, request success count, queue / compute-infer
latency sums (µs). Useful to find where latency is spent (Queue vs Compute Infer).

Usage:
    python perf/metrics.py
    python perf/metrics.py --host localhost --port 8002 --model ensemble
"""

import argparse
import collections
import re
import urllib.request


def fetch_metrics(host, port):
    url = f"http://{host}:{port}/metrics"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.read().decode()


def summarize(text, model_filter=None):
    patterns = {
        "inference_count": re.compile(
            r'nv_inference_count\{model="([^"]+)"[^}]*\}\s+([0-9.]+)'
        ),
        "request_success": re.compile(
            r'nv_inference_request_success\{model="([^"]+)"[^}]*\}\s+([0-9.]+)'
        ),
        "queue_us": re.compile(
            r'nv_inference_queue_duration_us\{model="([^"]+)"[^}]*\}\s+([0-9.]+)'
        ),
        "compute_infer_us": re.compile(
            r'nv_inference_compute_infer_duration_us\{model="([^"]+)"[^}]*\}\s+([0-9.]+)'
        ),
        "exec_count": re.compile(
            r'nv_inference_exec_count\{model="([^"]+)"[^}]*\}\s+([0-9.]+)'
        ),
    }
    rows = collections.defaultdict(dict)
    for label, pattern in patterns.items():
        for model, value in pattern.findall(text):
            if model_filter and model != model_filter:
                continue
            rows[model][label] = float(value)

    header = f"{'model':<16} {'count':>8} {'success':>10} {'queue_ms':>9} {'infer_ms':>9}"
    print(header)
    print("-" * len(header))
    for model in sorted(rows):
        r = rows[model]
        count = int(r.get("inference_count", 0))
        success = int(r.get("request_success", 0))
        queue = r.get("queue_us", 0) / 1e3
        infer = r.get("compute_infer_us", 0) / 1e3
        print(
            f"{model:<16} {count:>8} {success:>10} {queue:>9.1f} {infer:>9.1f}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Triton metrics summary")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model", help="restrict to a model")
    args = parser.parse_args()

    metrics = fetch_metrics(args.host, args.port)
    rows = summarize(metrics, args.model)
    if not rows:
        print("No metrics found (is your Triton metric endpoint on 8002?).")


if __name__ == "__main__":
    main()