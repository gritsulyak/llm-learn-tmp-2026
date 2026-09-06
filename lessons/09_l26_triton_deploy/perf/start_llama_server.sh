#!/usr/bin/env bash
# Starts llama.cpp `llama-server` using the Vulkan backend (Radeon 780M iGPU)
# for the Triton `generation` model.
#
# Usage:
#   bash perf/start_llama_server.sh
#
# Env overrides: LLAMA_DIR (llama.cpp build dir), MODEL (GGUF path),
#                PORT, N_GPU_LAYERS, N_PARALLEL, N_CTX.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-/home/rtg/github/llm_eng/llama.cpp}"
MODEL="${MODEL:-$REPO_DIR/model_repository/generation/1/model.gguf}"
PORT="${PORT:-8080}"
N_GPU_LAYERS="${N_GPU_LAYERS:--ngl 99}"
N_PARALLEL="${N_PARALLEL:-4}"
N_CTX="${N_CTX:-512}"

if [ ! -x "$LLAMA_DIR/build/bin/llama-server" ]; then
    echo "llama-server not found in $LLAMA_DIR" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "GGUF model not found: $MODEL" >&2
    exit 1
fi

echo "Starting llama-server: model=$MODEL port=$PORT"
exec "$LLAMA_DIR/build/bin/llama-server" \
    -m "$MODEL" \
    $N_GPU_LAYERS \
    -c "$N_CTX" \
    -np "$N_PARALLEL" \
    -t 4 \
    --host 0.0.0.0 \
    --port "$PORT"