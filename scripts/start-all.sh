#!/usr/bin/env bash
# Starts the parts in the background and the page in front. Logs: logs/*.log
# Pick the parts (see "How small can it get" in the README):
#   TTS=qwen (default, GPU ~4.6 GB) | tiny (apptor-tiny-tts, German, CPU) | phonon (CPU, German + English)
#   STT=whisper (default, GPU ~1.2 GB) | fluid (Fluid Speech Pro: GPU ~1.2 GB, or CPU with FV_FLUID_BACKEND=onnx)
#   NO_DECIDER=1  without the decider (~5 GB GPU less; simple rules decide instead)
#   e.g. the smallest:  TTS=tiny STT=fluid NO_DECIDER=1 scripts/start-all.sh
# Python of each part: TTS_PY / STT_PY / FV_PY (default: the environment from the README if it exists, else python3)
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
TTS="${TTS:-qwen}"; STT="${STT:-whisper}"; NO_DECIDER="${NO_DECIDER:-$NO_CLEF}"
py() { [ -x "$1/bin/python" ] && echo "$1/bin/python" || echo python3; }
case "$TTS" in
  qwen)   TTS_PY="${TTS_PY:-$(py .venv-tts)}";     TTS_CMD=(services/tts_server.py) ;;
  tiny)   TTS_PY="${TTS_PY:-$(py .venv-tinytts)}"; TTS_CMD=(services/tinytts_server.py); export FV_TINYTTS_PORT=8086
          [ -f models/apptor-tiny-tts-at.onnx ] && export FV_TINYTTS_MODEL="${FV_TINYTTS_MODEL:-models/apptor-tiny-tts-at.onnx}" || true ;;
  phonon) TTS_PY="${TTS_PY:-$(py .venv-phonon)}";  TTS_CMD=(services/phonon_server.py); export FV_PHONON_PORT=8086 ;;
  *) echo "TTS=$TTS? qwen, tiny or phonon"; exit 1 ;;
esac
case "$STT" in
  whisper) STT_PY="${STT_PY:-$(py .venv-stt)}"; STT_CMD=(services/stt_server.py) ;;
  fluid)   STT_PY="${STT_PY:-$(py .venv-stt)}"; STT_CMD=(services/fluid_stt_server.py)
           [ -f parakeet.cpp/build/libparakeet.so ] && export FV_FLUID_LIB="${FV_FLUID_LIB:-$PWD/parakeet.cpp/build/libparakeet.so}" || true ;;
  *) echo "STT=$STT? whisper or fluid"; exit 1 ;;
esac
FV_PY="${FV_PY:-$(py .venv)}"
wait_for() { for _ in $(seq 1 180); do curl -sf -o /dev/null "$1" && return 0; sleep 2; done; echo "  $1 does not answer - see logs/"; return 1; }
curl -sf -o /dev/null http://127.0.0.1:8086/health || { echo "starting the voice ($TTS) ..."; "$TTS_PY" "${TTS_CMD[@]}" > logs/tts.log 2>&1 & }
curl -sf -o /dev/null http://127.0.0.1:8087/health || { echo "starting the ears ($STT) ..."; "$STT_PY" "${STT_CMD[@]}" > logs/stt.log 2>&1 & }
if [ -n "$NO_DECIDER" ]; then export FV_DECIDER_URL= FV_CLEF_URL=
else curl -sf -o /dev/null http://127.0.0.1:8088/health || { echo "starting the decider ..."; scripts/start-decider.sh > logs/decider.log 2>&1 & }
fi
wait_for http://127.0.0.1:8087/health && wait_for http://127.0.0.1:8086/health
[ -n "$NO_DECIDER" ] || wait_for http://127.0.0.1:8088/health || true
exec "$FV_PY" -m flashvoice
