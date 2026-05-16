# syntax=docker/dockerfile:1
# Targets: cpu (slim + onnxruntime CPU) | gpu (slim + onnxruntime-gpu + wheels nvidia-cu12)
# Evita nvidia/cuda:~2GB en pull (Coolify suele cortar el build); en runtime sigue haciendo falta GPU NVIDIA + toolkit.

FROM python:3.12-slim-bookworm AS cpu

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     libgomp1 \
     tesseract-ocr \
     tesseract-ocr-spa \
     tesseract-ocr-eng \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PORT=8001
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/var/cache/huggingface

EXPOSE 8001

CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port \"${PORT:-8001}\" --timeout-keep-alive 120"]


FROM python:3.12-slim-bookworm AS gpu

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     libgomp1 \
     tesseract-ocr \
     tesseract-ocr-spa \
     tesseract-ocr-eng \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt \
  && pip uninstall -y onnxruntime \
  && pip install --no-cache-dir \
       "onnxruntime-gpu>=1.19.2,<2.0.0" \
       "nvidia-cublas-cu12" \
       "nvidia-cudnn-cu12" \
       "nvidia-cuda-runtime-cu12" \
  && python3 -c "import pathlib, site; \
root = pathlib.Path(site.getsitepackages()[0]) / 'nvidia'; \
paths = sorted({str(p.parent) for p in root.rglob('*.so*') if p.is_file()}); \
open('/etc/ocr-nvidia-ld-path', 'w').write(':'.join(paths))"

COPY docker/entrypoint-gpu.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY src/ ./src/

ENV PORT=8001
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/var/cache/huggingface

EXPOSE 8001

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port \"${PORT:-8001}\" --timeout-keep-alive 120"]
