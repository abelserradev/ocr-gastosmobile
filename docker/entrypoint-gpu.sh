#!/bin/sh
# Ruta a .so de los wheels nvidia-*-cu12 (libcublasLt, etc.) para onnxruntime-gpu en slim
if [ -f /etc/ocr-nvidia-ld-path ]; then
  _nv_path=$(cat /etc/ocr-nvidia-ld-path)
  if [ -n "$_nv_path" ]; then
    export LD_LIBRARY_PATH="${_nv_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi
exec "$@"
