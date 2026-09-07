#!/usr/bin/env bash
# Starts llama.cpp `llama-server` (CPU build or Vulkan on Radeon 780M iGPU)
# for the Triton `generation` model.
#
# Usage:
#   bash perf/start_llama_server.sh              # Vulkan build, -ngl 99 (default)
#   LLAMA_BIN=.../build-cpu/bin/llama-server N_GPU_LAYERS=off \
#       bash perf/start_llama_server.sh          # CPU-only build, no Vulkan
#
# Env overrides: LLAMA_DIR (llama.cpp build dir), LLAMA_BIN (binary path),
#                MODEL (GGUF path), PORT, N_GPU_LAYERS, N_PARALLEL, N_CTX.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-/home/rtg/github/llm_eng/llama.cpp}"
BIN="${LLAMA_BIN:-$LLAMA_DIR/build/bin/llama-server}"
MODEL="${MODEL:-$REPO_DIR/model_repository/generation/1/model.gguf}"
PORT="${PORT:-8080}"
N_GPU_LAYERS="${N_GPU_LAYERS:--ngl 99}"
case "$N_GPU_LAYERS" in
  off|none) GPU_ARGS="" ;;
  *) GPU_ARGS="$N_GPU_LAYERS" ;;
esac
N_PARALLEL="${N_PARALLEL:-4}"
N_CTX="${N_CTX:-512}"

if [ ! -x "$BIN" ]; then
    echo "llama-server not found: $BIN" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "GGUF model not found: $MODEL" >&2
    exit 1
fi

echo "Starting llama-server: model=$MODEL port=$PORT gpu_args='$GPU_ARGS'"
exec "$BIN" \
    -m "$MODEL" \
    $GPU_ARGS \
    -c "$N_CTX" \
    -np "$N_PARALLEL" \
    -t 4 \
    --host 0.0.0.0 \
    --port "$PORT"