# flash-voice lite: a small voice + ears for Open WebUI (or any app using the OpenAI audio API), CPU only.
#   voice: apptor-tiny-tts (German, Apache 2.0) + Phonon English (Apache 2.0)
#   ears:  Fluid Speech Pro int8 (German, English, Croatian; CC BY 4.0)
# One port: 8090 (OpenAI-compatible: /v1/audio/speech, /v1/audio/transcriptions). Models are downloaded on the
# first start into /models (~1 GB, keep it as a volume).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY docker/lite/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY services/ services/
COPY docker/lite/start.py docker/lite/start.py
ENV HF_HOME=/models/hf FV_MODELS=/models PYTHONUNBUFFERED=1 FV_THREADS=4
EXPOSE 8090
VOLUME /models
CMD ["python", "docker/lite/start.py"]
