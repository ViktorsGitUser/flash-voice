@echo off
setlocal
REM Starts the voice (TTS), the ears (STT), Clef and the page. Each part in its own minimized window.
REM Set these to your environments (see README):
if "%TTS_PY%"=="" set TTS_PY=.venv-tts\Scripts\python.exe
if "%STT_PY%"=="" set STT_PY=.venv-stt\Scripts\python.exe
if "%FV_PY%"=="" set FV_PY=.venv\Scripts\python.exe
if "%LLAMA_SERVER%"=="" set LLAMA_SERVER=llama-server.exe
if "%CLEF_GGUF%"=="" set CLEF_GGUF=models\Cloudflare_clef-flash-Q4_K_M.gguf
cd /d "%~dp0.."
if not exist logs mkdir logs
curl -s -f -o nul -m 3 http://127.0.0.1:8086/health || start "flash-voice TTS" /min cmd /c ""%TTS_PY%" services\tts_server.py > logs\tts.log 2>&1"
curl -s -f -o nul -m 3 http://127.0.0.1:8087/health || start "flash-voice STT" /min cmd /c ""%STT_PY%" services\stt_server.py > logs\stt.log 2>&1"
if "%NO_CLEF%"=="" curl -s -f -o nul -m 3 http://127.0.0.1:8088/health || start "flash-voice Clef" /min cmd /c ""%LLAMA_SERVER%" -m "%CLEF_GGUF%" --host 127.0.0.1 --port 8088 -ngl 99 -c 4096 -b 4096 -ub 4096 --parallel 1 > logs\clef.log 2>&1"
echo Waiting for the services (the first start downloads the models) ...
:wait
curl -s -f -o nul -m 3 http://127.0.0.1:8086/health && curl -s -f -o nul -m 3 http://127.0.0.1:8087/health && goto run
timeout /t 3 /nobreak >nul
goto wait
:run
"%FV_PY%" -m flashvoice
