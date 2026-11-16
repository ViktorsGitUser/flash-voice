#!/usr/bin/env bash
# Clef, the decision model (vllm-d3-mini, Apache 2.0; until 11/2026 Cloudflare clef-flash) in llama.cpp's llama-server
# (b11719 or newer: it has /v1/systemone and the vllm-d3 format). Text only (no vision projector): ~5 GB of VRAM. Clef needs the whole request in one batch: -b/-ub = context.
#   LLAMA_SERVER=/path/to/llama-server   CLEF_GGUF=/path/to/vllm-d3-mini-Q8_0.gguf
# Download the model once:
#   hf download ggml-org/vllm-d3-mini-GGUF vllm-d3-mini-Q8_0.gguf --local-dir models
set -e
cd "$(dirname "$0")/.."
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
CLEF_GGUF="${CLEF_GGUF:-models/vllm-d3-mini-Q8_0.gguf}"
exec "$LLAMA_SERVER" -m "$CLEF_GGUF" --host 127.0.0.1 --port "${CLEF_PORT:-8088}" -ngl 99 \
     -c 4096 -b 4096 -ub 4096 --parallel 1
