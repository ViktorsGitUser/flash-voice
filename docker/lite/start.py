"""flash-voice lite: downloads the models on the first start, then runs the four parts and keeps them running.
  Fluid Speech Pro (CPU) :8087   Phonon English :8085   apptor-tiny-tts :8086 (English -> Phonon)   OpenAI API :8090
Settings (environment): FV_THREADS (CPU threads per part, default 4), FV_OAI_KEY (API key, optional),
FV_OAI_VOICE (default voice, m2), FV_TINYTTS_ACCENT (0 = Hochdeutsch ... 1.6 = strong Austrian accent, default 0)."""
import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

M = Path(os.environ.get("FV_MODELS", "/models"))
T = os.environ.get("FV_THREADS", "4")
APP = Path(__file__).resolve().parents[2]


def models():
    tiny = M / "tiny"
    for f in ("apptor-tiny-tts-at.onnx", "apptor-tiny-tts-at.json", "apptor-g2p-de.onnx", "apptor-g2p-de.json",
              "apptor-vocos-de.int8.onnx"):
        if not (tiny / f).exists():
            print("[lite] downloading", f, flush=True)
            hf_hub_download("apptor/apptor-tiny-tts-at", f, local_dir=tiny)
    fluid = M / "fluid-speech-pro-onnx"
    if not (fluid / "encoder-model.int8.onnx.data").exists():
        print("[lite] downloading Fluid Speech Pro int8 (~0.8 GB) ...", flush=True)
        snapshot_download("FluidVoice-Labs/fluid-speech-pro-onnx", local_dir=fluid, allow_patterns=[
            "config.json", "vocab.txt", "LICENSE", "NOTICE", "encoder-model.int8.onnx*", "decoder_joint-model.int8.onnx*"])
    return tiny, fluid


def main():
    tiny, fluid = models()
    env = dict(os.environ)
    parts = {
        "ears": ({"FV_STT_PORT": "8087", "FV_STT_LANGS": "en,de", "FV_FLUID_BACKEND": "onnx",
                  "FV_FLUID_MODEL": str(fluid), "FV_FLUID_THREADS": T}, "services/fluid_stt_server.py"),
        "phonon": ({"FV_PHONON_PORT": "8085", "FV_PHONON_LANGS": "en", "FV_PHONON_THREADS": T,
                    "FV_PHONON_DEFAULT": "henrik"}, "services/phonon_server.py"),
        "voice": ({"FV_TINYTTS_PORT": "8086", "FV_TINYTTS_MODEL": str(tiny / "apptor-tiny-tts-at.onnx"),
                   "FV_TINYTTS_G2P": str(tiny / "apptor-g2p-de.onnx"),
                   "FV_TINYTTS_VOCOS": str(tiny / "apptor-vocos-de.int8.onnx"),
                   "FV_TINYTTS_EN_URL": "http://127.0.0.1:8085", "FV_TINYTTS_THREADS": T,
                   "FV_TINYTTS_ACCENT": env.get("FV_TINYTTS_ACCENT", "0"), "FV_TINYTTS_DEFAULT": "m2"},
                  "services/tinytts_server.py"),
        "api": ({"FV_OAI_PORT": "8090", "FV_OAI_HOST": "0.0.0.0", "FV_TTS_URL": "http://127.0.0.1:8086",
                 "FV_STT_URL": "http://127.0.0.1:8087", "FV_OAI_VOICE": env.get("FV_OAI_VOICE", "m2")},
                "services/openai_audio.py"),
    }
    procs = {}

    def start(name):
        extra, script = parts[name]
        procs[name] = subprocess.Popen([sys.executable, script], cwd=APP, env={**env, **extra})

    for name in parts:
        start(name)
    print("[lite] ready soon on :8090 (OpenAI audio API)", flush=True)
    while True:
        time.sleep(5)
        for name, p in procs.items():
            if p.poll() is not None:
                print(f"[lite] {name} stopped ({p.returncode}) - starting it again", flush=True)
                start(name)


if __name__ == "__main__":
    main()
