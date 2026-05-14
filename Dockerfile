# syntax=docker/dockerfile:1
# Imagen OCR (Moondream ONNX). USE_GPU=1 sustituye onnxruntime por onnxruntime-gpu (NVIDIA en el host).
FROM python:3.12-slim-bookworm

ARG USE_GPU=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# moondream arrastra onnxruntime CPU; para CUDA en contenedor hace falta el paquete -gpu.
RUN if [ "$USE_GPU" = "1" ]; then \
      pip uninstall -y onnxruntime \
      && pip install --no-cache-dir "onnxruntime-gpu>=1.19.2,<2.0.0"; \
    fi

COPY src/ ./src/

ENV PORT=8001
ENV PYTHONUNBUFFERED=1
# Caché Hugging Face (montar volumen en compose para no re-descargar en cada recreación)
ENV HF_HOME=/var/cache/huggingface

EXPOSE 8001

CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port \"${PORT:-8001}\" --timeout-keep-alive 120"]
