"""Settings - from the environment or a .env file next to this project (see .env.example)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env():
    f = Path(os.environ.get("FV_ENV", ROOT / ".env"))
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
E = os.environ.get

# the brain: any OpenAI-compatible chat endpoint (Ollama, llama.cpp, vLLM, LM Studio, OpenAI, OpenRouter ...)
LLM_URL = E("FV_LLM_URL", "http://127.0.0.1:11434/v1").rstrip("/")
LLM_KEY = E("FV_LLM_KEY", "")
LLM_MODEL = E("FV_LLM_MODEL", "")                 # empty: the first model the endpoint lists
LLM_EXTRA = E("FV_LLM_EXTRA", "")                 # JSON merged into each request, e.g. {"temperature": 0.7}
SYSTEM_PROMPT = E("FV_SYSTEM_PROMPT", "")         # empty: the built-in one (short, spoken answers)
# the services
TTS_URL = E("FV_TTS_URL", "http://127.0.0.1:8086").rstrip("/")
STT_URL = E("FV_STT_URL", "http://127.0.0.1:8087").rstrip("/")
CLEF_URL = E("FV_CLEF_URL", "http://127.0.0.1:8088").rstrip("/")   # empty or unreachable: simple rules instead
LANGS = tuple(x.strip() for x in E("FV_LANGS", "en,de").split(",") if x.strip())
VOICE = E("FV_VOICE", "serena")
# the page
HOST = E("FV_HOST", "127.0.0.1")
PORT = int(E("FV_PORT", "8099"))
PASSWORD = E("FV_PASSWORD", "")                   # set it when the page is reachable from other devices
CERT, KEY = E("FV_CERT", ""), E("FV_KEY", "")     # HTTPS (a phone's microphone needs it); "self" = make one
