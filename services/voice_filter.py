"""Clean a recording before it becomes a voice: the model imitates everything in it - also its noise. Takes off steady
background noise (noisereduce, spectral gating, 85 % so the voice does not sound tinny) and low hum (high-pass at
70 Hz). Used by tts_server.py's /voices/clone; also by hand:

    python services/voice_filter.py recording.m4a name "exactly what is said"   -> voices/name.wav + name.txt

Only with the consent of the person whose voice it is.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000


def load(path: str) -> np.ndarray:
    """Any audio file -> mono float32 at 24 kHz (wav/flac directly, the rest through ffmpeg on the PATH)."""
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(1)
        if sr != SR:
            audio = np.interp(np.arange(0, len(audio), sr / SR), np.arange(len(audio)), audio).astype(np.float32)
        return audio
    except Exception:
        out = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(out, dtype=np.float32)


def noise_db(x: np.ndarray) -> float:
    """Level of the quietest 10 % of 50-ms frames: the noise between the words."""
    n = int(0.05 * SR)
    frames = x[: len(x) // n * n].reshape(-1, n)
    return float(20 * np.log10(np.percentile(np.sqrt((frames ** 2).mean(1)) + 1e-9, 10)))


def clean(audio: np.ndarray) -> np.ndarray:
    import noisereduce as nr
    from scipy.signal import butter, sosfiltfilt
    audio = sosfiltfilt(butter(2, 70, "highpass", fs=SR, output="sos"), audio).astype(np.float32)
    audio = nr.reduce_noise(y=audio, sr=SR, stationary=True, prop_decrease=0.85, n_fft=1024).astype(np.float32)
    return audio / max(1e-6, float(np.abs(audio).max())) * 0.6


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, name, text = sys.argv[1:]
    folder = Path(__file__).resolve().parent.parent / "voices"
    raw = load(src)
    out = clean(raw)
    folder.mkdir(exist_ok=True)
    sf.write(folder / f"{name}.wav", out, SR, subtype="PCM_16")
    (folder / f"{name}.txt").write_text(text.strip(), encoding="utf-8")
    print(f"{name}: {len(out) / SR:.1f} s, noise {noise_db(raw / max(1e-6, float(np.abs(raw).max())) * 0.6):.1f} dB "
          f"-> {noise_db(out):.1f} dB, saved in {folder / (name + '.wav')}")
