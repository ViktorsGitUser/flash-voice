@echo off
setlocal
REM Starts the voice (TTS), the ears (STT), the decider and the page. Each part in its own minimized window.
REM Pick the parts (see "How small can it get" in the README):
REM   TTS=lite (English Phonon + German apptor-tiny-tts, CPU) ^| qwen (default, GPU) ^| phonon
REM   STT=fluid (Fluid Speech Pro; set FV_FLUID_LIB, or FV_FLUID_BACKEND=onnx for the CPU) ^| whisper (default, GPU)
REM   the decider: models\flash-turn (mini, CPU) when it is there, else llama-server (FV_DECIDER=llm forces it)
REM   NO_DECIDER=1  rules only                         e.g.  set TTS=lite ^& set STT=fluid
if "%TTS%"=="" set TTS=qwen
if "%TTS%"=="tiny" set TTS=lite
if "%STT%"=="" set STT=whisper
if "%NO_DECIDER%"=="" set NO_DECIDER=%NO_CLEF%
if "%STT_PY%"=="" set STT_PY=.venv-stt\Scripts\python.exe
if "%FV_PY%"=="" set FV_PY=.venv\Scripts\python.exe
if "%LLAMA_SERVER%"=="" set LLAMA_SERVER=llama-server.exe
if "%DECIDER_GGUF%"=="" set DECIDER_GGUF=%CLEF_GGUF%
if "%DECIDER_GGUF%"=="" set DECIDER_GGUF=models\vllm-d3-mini-Q8_0.gguf
set TTS_SCRIPT=services\tts_server.py
if "%TTS%"=="qwen" if "%TTS_PY%"=="" set TTS_PY=.venv-tts\Scripts\python.exe
if "%TTS%"=="lite" (
  set TTS_SCRIPT=services\tinytts_server.py
  set FV_TINYTTS_PORT=8086
  if "%FV_TINYTTS_EN_URL%"=="" set FV_TINYTTS_EN_URL=http://127.0.0.1:8085
  if "%FV_TINYTTS_ACCENT%"=="" set FV_TINYTTS_ACCENT=0
  if "%TTS_PY%"=="" set TTS_PY=.venv-lite\Scripts\python.exe
  if "%FV_TINYTTS_MODEL%"=="" if exist models\apptor-tiny-tts-at.onnx set FV_TINYTTS_MODEL=models\apptor-tiny-tts-at.onnx
)
if "%TTS%"=="phonon" (
  set TTS_SCRIPT=services\phonon_server.py
  set FV_PHONON_PORT=8086
  if "%TTS_PY%"=="" set TTS_PY=.venv-phonon\Scripts\python.exe
)
set STT_SCRIPT=services\stt_server.py
if "%STT%"=="fluid" set STT_SCRIPT=services\fluid_stt_server.py
cd /d "%~dp0.."
if not exist logs mkdir logs
if "%TTS%"=="lite" curl -s -f -o nul -m 3 http://127.0.0.1:8085/health || start "flash-voice English voice" /min cmd /c "set FV_PHONON_LANGS=en&& "%TTS_PY%" services\phonon_server.py > logs\phonon.log 2>&1"
curl -s -f -o nul -m 3 http://127.0.0.1:8086/health || start "flash-voice TTS" /min cmd /c ""%TTS_PY%" %TTS_SCRIPT% > logs\tts.log 2>&1"
curl -s -f -o nul -m 3 http://127.0.0.1:8087/health || start "flash-voice STT" /min cmd /c ""%STT_PY%" %STT_SCRIPT% > logs\stt.log 2>&1"
set BIG=1
if exist models\flash-turn\flash-turn.onnx if not "%FV_DECIDER%"=="llm" set BIG=
if not "%NO_DECIDER%"=="" (
  set FV_DECIDER=rules
  set FV_DECIDER_URL=
  set FV_CLEF_URL=
) else if "%BIG%"=="1" (
  curl -s -f -o nul -m 3 http://127.0.0.1:8088/health || start "flash-voice decider" /min cmd /c ""%LLAMA_SERVER%" -m "%DECIDER_GGUF%" --host 127.0.0.1 --port 8088 -ngl 99 -c 4096 -b 4096 -ub 4096 --parallel 1 > logs\decider.log 2>&1"
)
echo Waiting for the services (the first start downloads the models) ...
:wait
curl -s -f -o nul -m 3 http://127.0.0.1:8086/health && curl -s -f -o nul -m 3 http://127.0.0.1:8087/health && goto run
timeout /t 3 /nobreak >nul
goto wait
:run
"%FV_PY%" -m flashvoice
