# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ARG USE_GPU=0

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

RUN if [ "$USE_GPU" = "1" ]; then \
      pip uninstall -y onnxruntime \
      && pip install --no-cache-dir "onnxruntime-gpu>=1.19.2,<2.0.0"; \
    else \
      : "onnxruntime (CPU) viene con moondream; no instalamos onnxruntime-gpu"; \
    fi

COPY src/ ./src/

ENV PORT=8001
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/var/cache/huggingface

EXPOSE 8001

CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port \"${PORT:-8001}\" --timeout-keep-alive 120"]
