#!/usr/bin/env bash
# Run NVIDIA Perf Analyzer (from the Triton SDK image) against all models.
#
# Requires a reachable Triton server on localhost:8001. The SDK image is pulled
# from NGC; set SDK_IMAGE to an alternative if you do not have access.
set -euo pipefail

SDK_IMAGE="${SDK_IMAGE:-nvcr.io/nvidia/tritonserver:24.01-py3-sdk}"
URL="${PERF_URL:-localhost:8001}"
CONCURRENCY="${CONCURRENCY:-1:4}"
MEASUREMENT_INTERVAL="${MEASUREMENT_INTERVAL:-5000}"
PROMPT="${PERF_PROMPT:-What is machine learning and how does it work?}"
OUT_DIR="$(dirname "$0")/results"

mkdir -p "$OUT_DIR"

docker_run() {
  docker run --rm --network host \
    -v "$PWD/$OUT_DIR:/results" \
    "$SDK_IMAGE" perf_analyzer \
      --url "$URL" \
      --concurrency-range "$CONCURRENCY" \
      --measurement-mode time_windows \
      --measurement-interval "$MEASUREMENT_INTERVAL" \
      --percentile 95 \
      "$@"
}

echo "== text_tokenizer =="
docker_run -m text_tokenizer --string-data "$PROMPT"

echo "== classification (needs pre-tokenized input; uses --input-data) =="
echo "  run directly with INFER via SDK client, see perf/load_test.py --model classification"

echo "== embedding =="
docker_run -m embedding --string-data "$PROMPT"

echo "== generation =="
docker_run -m generation --string-data "$PROMPT"

echo "== ensemble (BLS pipeline) =="
docker_run -m ensemble --string-data "$PROMPT"

echo "Done. Results are printed above / saved via --export to $OUT_DIR."