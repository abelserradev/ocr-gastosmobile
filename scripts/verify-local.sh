#!/usr/bin/env sh
# Verificación rápida del OCR antes de desplegar al servidor.
# Uso:
#   cd ocr && ./scripts/verify-local.sh
#   VERIFY_INVOICE_IMAGE=/ruta/factura.jpg ./scripts/verify-local.sh
#
# Opcional:
#   OCR_BASE_URL=http://127.0.0.1:8001

set -eu

BASE="${OCR_BASE_URL:-http://127.0.0.1:8001}"
BASE="${BASE%/}"

echo "=== OCR Gastos — verificación local ==="
echo "Base URL: $BASE"
echo ""

echo "--- 1) GET /health ---"
if ! code=$(curl -sS -o /tmp/ocr-health.json -w "%{http_code}" "$BASE/health"); then
  echo "ERROR: no se pudo conectar. ¿Está el contenedor arriba?"
  echo "  docker compose -f docker-compose.local.yml up --build"
  exit 1
fi

if [ "$code" != "200" ]; then
  echo "ERROR: HTTP $code"
  cat /tmp/ocr-health.json 2>/dev/null || true
  exit 1
fi

if command -v jq >/dev/null 2>&1; then
  jq . /tmp/ocr-health.json
else
  cat /tmp/ocr-health.json
  echo "(instala \`jq\` para formato JSON legible)"
fi
echo ""

sample="${VERIFY_INVOICE_IMAGE:-}"
if [ -z "$sample" ]; then
  echo "--- 2) POST /parse-invoice (omitido) ---"
  echo "Para probar una factura real:"
  echo "  VERIFY_INVOICE_IMAGE=/ruta/a/tu-factura.jpg $0"
  echo ""
  echo "Listo: health OK. Falta prueba manual de parse-invoice para aprobar pase a servidor."
  exit 0
fi

if [ ! -f "$sample" ]; then
  echo "ERROR: no existe el archivo: $sample"
  exit 1
fi

echo "--- 2) POST /parse-invoice (archivo: $sample) ---"
curl -sS -X POST "$BASE/parse-invoice" \
  -F "file=@${sample}" \
  -H "Accept: application/json" | tee /tmp/ocr-parse.json

echo ""
if command -v jq >/dev/null 2>&1; then
  echo ""
  echo "--- Resumen (jq) ---"
  jq '{amount,date,merchant,confidence,currency,raw_preview:(.raw_text|tostring|.[0:200])}' \
    /tmp/ocr-parse.json 2>/dev/null || true
fi

echo ""
echo "=== Fin ==="
echo "Revisa: amount/merchant coherentes, raw_text con bloque Tesseract, confidence razonable."
echo "Si OK → aprobación para subir misma imagen/compose al servidor."
