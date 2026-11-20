#!/usr/bin/env bash
# The decider, a small decision model (vllm-d3-mini, Apache 2.0, 4B) in llama.cpp's llama-server (b11719 or newer: it
# has /v1/systemone and the vllm-d3 format). Text only (no vision projector): ~5 GB of VRAM. It needs the whole request
# in one batch: -b/-ub = context.
#   LLAMA_SERVER=/path/to/llama-server   DECIDER_GGUF=/path/to/vllm-d3-mini-Q8_0.gguf   (CLEF_GGUF: old name)
# Download the model once:
#   hf download ggml-org/vllm-d3-mini-GGUF vllm-d3-mini-Q8_0.gguf --local-dir models
set -e
cd "$(dirname "$0")/.."
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
DECIDER_GGUF="${DECIDER_GGUF:-${CLEF_GGUF:-models/vllm-d3-mini-Q8_0.gguf}}"
exec "$LLAMA_SERVER" -m "$DECIDER_GGUF" --host 127.0.0.1 --port "${DECIDER_PORT:-${CLEF_PORT:-8088}}" -ngl 99 \
     -c 4096 -b 4096 -ub 4096 --parallel 1
