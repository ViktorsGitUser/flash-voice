#!/usr/bin/env bash
# Starts the four parts in the background and the page in front. Logs: logs/*.log
#   TTS_PY / STT_PY / FV_PY: the Python of each environment (default: python3)
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
TTS_PY="${TTS_PY:-python3}"; STT_PY="${STT_PY:-python3}"; FV_PY="${FV_PY:-python3}"
wait_for() { for _ in $(seq 1 180); do curl -sf -o /dev/null "$1" && return 0; sleep 2; done; echo "  $1 does not answer - see logs/"; return 1; }
curl -sf -o /dev/null http://127.0.0.1:8086/health || { echo "starting the voice (TTS) ..."; "$TTS_PY" services/tts_server.py > logs/tts.log 2>&1 & }
curl -sf -o /dev/null http://127.0.0.1:8087/health || { echo "starting the ears (STT) ..."; "$STT_PY" services/stt_server.py > logs/stt.log 2>&1 & }
curl -sf -o /dev/null http://127.0.0.1:8088/health || { [ -n "$NO_CLEF" ] || { echo "starting Clef ..."; scripts/start-clef.sh > logs/clef.log 2>&1 & }; }
wait_for http://127.0.0.1:8087/health && wait_for http://127.0.0.1:8086/health
[ -n "$NO_CLEF" ] || wait_for http://127.0.0.1:8088/health || true
exec "$FV_PY" -m flashvoice
