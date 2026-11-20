#!/usr/bin/env bash
# Starts the parts in the background and the page in front. Logs: logs/*.log
# Pick the parts (see "How small can it get" in the README):
#   TTS=lite  English (Phonon) + German (apptor-tiny-tts), CPU   | qwen (default, GPU ~4.6 GB) | phonon | tiny (= lite)
#   STT=fluid (Fluid Speech Pro: GPU ~1.2 GB, or CPU with FV_FLUID_BACKEND=onnx)  | whisper (default, GPU ~1.2 GB)
#   the decider: the mini model (models/flash-turn, CPU) when it is there; else vllm-d3-mini in llama-server if
#   LLAMA_SERVER is set (~5 GB GPU; with FV_DECIDER=llm also when the mini model is there); NO_DECIDER=1: rules only
#   e.g. the smallest:  TTS=lite STT=fluid scripts/start-all.sh
# Python of each part: TTS_PY / STT_PY / FV_PY (default: the environment from the README if it exists, else python3)
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
TTS="${TTS:-qwen}"; STT="${STT:-whisper}"; NO_DECIDER="${NO_DECIDER:-$NO_CLEF}"
py() { [ -x "$1/bin/python" ] && echo "$1/bin/python" || echo python3; }
case "$TTS" in
  qwen)   TTS_PY="${TTS_PY:-$(py .venv-tts)}";     TTS_CMD=(services/tts_server.py) ;;
  lite|tiny) TTS_PY="${TTS_PY:-$(py .venv-lite)}"; TTS_CMD=(services/tinytts_server.py); export FV_TINYTTS_PORT=8086
          [ -f models/apptor-tiny-tts-at.onnx ] && export FV_TINYTTS_MODEL="${FV_TINYTTS_MODEL:-models/apptor-tiny-tts-at.onnx}"
          export FV_TINYTTS_EN_URL="${FV_TINYTTS_EN_URL:-http://127.0.0.1:8085}" FV_TINYTTS_ACCENT="${FV_TINYTTS_ACCENT:-0}"
          curl -sf -o /dev/null http://127.0.0.1:8085/health || { echo "starting the English voice (Phonon) ..."
            FV_PHONON_LANGS=en "$TTS_PY" services/phonon_server.py > logs/phonon.log 2>&1 & } ;;
  phonon) TTS_PY="${TTS_PY:-$(py .venv-phonon)}";  TTS_CMD=(services/phonon_server.py); export FV_PHONON_PORT=8086 ;;
  *) echo "TTS=$TTS? lite, qwen or phonon"; exit 1 ;;
esac
case "$STT" in
  whisper) STT_PY="${STT_PY:-$(py .venv-stt)}"; STT_CMD=(services/stt_server.py) ;;
  fluid)   [ -x .venv-stt/bin/python ] && STT_PY="${STT_PY:-.venv-stt/bin/python}" || STT_PY="${STT_PY:-$(py .venv-lite)}"
           STT_CMD=(services/fluid_stt_server.py)
           [ -f parakeet.cpp/build/libparakeet.so ] && export FV_FLUID_LIB="${FV_FLUID_LIB:-$PWD/parakeet.cpp/build/libparakeet.so}" || true ;;
  *) echo "STT=$STT? whisper or fluid"; exit 1 ;;
esac
FV_PY="${FV_PY:-$(py .venv)}"
wait_for() { for _ in $(seq 1 180); do curl -sf -o /dev/null "$1" && return 0; sleep 2; done; echo "  $1 does not answer - see logs/"; return 1; }
curl -sf -o /dev/null http://127.0.0.1:8086/health || { echo "starting the voice ($TTS) ..."; "$TTS_PY" "${TTS_CMD[@]}" > logs/tts.log 2>&1 & }
curl -sf -o /dev/null http://127.0.0.1:8087/health || { echo "starting the ears ($STT) ..."; "$STT_PY" "${STT_CMD[@]}" > logs/stt.log 2>&1 & }
BIG=""
if [ -z "$NO_DECIDER" ] && [ -n "$LLAMA_SERVER" ] && { [ "$FV_DECIDER" = llm ] || [ ! -f models/flash-turn/flash-turn.onnx ]; }; then BIG=1
  curl -sf -o /dev/null http://127.0.0.1:8088/health || { echo "starting the big decider ..."; scripts/start-decider.sh > logs/decider.log 2>&1 & }
fi
[ -n "$NO_DECIDER" ] && export FV_DECIDER=rules FV_DECIDER_URL= FV_CLEF_URL=
wait_for http://127.0.0.1:8087/health && wait_for http://127.0.0.1:8086/health
[ -z "$BIG" ] || wait_for http://127.0.0.1:8088/health || true
exec "$FV_PY" -m flashvoice
