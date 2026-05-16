# syntax=docker/dockerfile:1
# Targets: cpu (slim, onnxruntime CPU) | gpu (CUDA 12 + cuDNN, onnxruntime-gpu)
# El wheel onnxruntime-gpu necesita libcublasLt en la imagen; python:slim no las trae.

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


# 12.4.1-*-ubuntu24.04 no está publicado en Docker Hub; 12.6.3 sí (cuBLAS 12 para onnxruntime-gpu)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     python3 \
     python3-pip \
     python3-venv \
     libgomp1 \
     tesseract-ocr \
     tesseract-ocr-spa \
     tesseract-ocr-eng \
  && ln -sf /usr/bin/python3 /usr/bin/python \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip \
  && python3 -m pip install --no-cache-dir -r requirements.txt \
  && python3 -m pip uninstall -y onnxruntime \
  && python3 -m pip install --no-cache-dir "onnxruntime-gpu>=1.19.2,<2.0.0"

COPY src/ ./src/

ENV PORT=8001
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/var/cache/huggingface
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

EXPOSE 8001

CMD ["sh", "-c", "exec python3 -m uvicorn src.main:app --host 0.0.0.0 --port \"${PORT:-8001}\" --timeout-keep-alive 120"]
