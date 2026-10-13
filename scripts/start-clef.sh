#!/usr/bin/env bash
# Clef (Cloudflare clef-flash, Apache 2.0) in llama.cpp's llama-server (b11556 or newer: it has /v1/systemone).
# Text only (no vision projector): ~6 GB of VRAM. Clef needs the whole request in one batch: -b/-ub = context.
#   LLAMA_SERVER=/path/to/llama-server   CLEF_GGUF=/path/to/Cloudflare_clef-flash-Q4_K_M.gguf
# Download the model once:
#   hf download bartowski/Cloudflare_clef-flash-GGUF Cloudflare_clef-flash-Q4_K_M.gguf --local-dir models
set -e
cd "$(dirname "$0")/.."
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
CLEF_GGUF="${CLEF_GGUF:-models/Cloudflare_clef-flash-Q4_K_M.gguf}"
exec "$LLAMA_SERVER" -m "$CLEF_GGUF" --host 127.0.0.1 --port "${CLEF_PORT:-8088}" -ngl 99 \
     -c 4096 -b 4096 -ub 4096 --parallel 1
