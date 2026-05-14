# Servicio OCR - Gastos

Servicio Python (FastAPI) para extracción de datos de facturas usando **Moondream** (vision-language model local).

## ¿Qué hace?

- Recibe imágenes de facturas (JPG, PNG, WebP)
- Extrae: **monto**, **fecha**, **comercio**, **descripción**
- Retorna JSON estructurado con nivel de confianza

## Requisitos

- Python 3.10+
- ~4GB RAM (Moondream corre en CPU)
- GPU opcional (CUDA para velocidad)

## Instalación

```bash
cd ocr/
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

```bash
# Desarrollo (puerto 8001 por defecto)
python src/main.py

# O con uvicorn explícito
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

## Docker (NVIDIA GPU en el host)

El servicio vive en esta carpeta (`ocr/`). El código Nest que llama al OCR está en `backend/src/ocr/` (no es este contenedor).

**Requisitos:** Docker, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) y driver propio del host (p. ej. Quadro P600).

```bash
cd ocr/
docker compose up --build -d
# Health: curl -s http://localhost:8001/health
```

- Caché de Hugging Face en el volumen `huggingface_ocr_cache` (no re-descarga el `.mf.gz` en cada `docker compose down` sin `-v`).
- **`MOONDREAM_ONNX_VARIANT=0.5b`** recomendable si la GPU tiene poca VRAM (p. ej. 2GB): añádelo bajo `environment` en `docker-compose.yml` o en un `.env` junto al compose.
- **Solo CPU:** comenta `gpus: all` en `docker-compose.yml` y ejecuta `docker compose build --build-arg USE_GPU=0` antes del `up`.
- **Cloudflare:** el proxy naranja corta ~100s; inferencias largas pueden necesitar DNS only, async job o API Moondream en nube (`MOONDREAM_API_KEY`).

## Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/parse-invoice` | Recibe imagen multipart, retorna datos extraídos |
| GET | `/health` | Health check, indica si modelo está cargado |

## Ejemplo de uso (curl)

```bash
curl -X POST \
  http://localhost:8001/parse-invoice \
  -F "file=@factura.jpg" \
  -H "Accept: application/json"
```

## Respuesta

```json
{
  "amount": 25.50,
  "date": "2026-05-14",
  "merchant": "Farmatodo",
  "description": "Medicamentos",
  "rawText": "FARMATODO C.A.\nFecha: 14/05/2026\nTotal: 25.50",
  "confidence": 0.85,
  "currency": "USD"
}
```

## Integración con NestJS

El backend NestJS se comunica con este servicio vía HTTP:

```
Frontend (Angular) → NestJS (backend) → OCR Service (Python)
                        ↓
                    POST /ocr/parse-invoice
```

Variable de entorno en backend:
```bash
OCR_SERVICE_URL=http://localhost:8001  # default
```

## Notas

- Primera ejecución: descarga modelo Moondream (~1.6GB) automáticamente
- El modelo se cachea en `~/.cache/moondream/`
- Procesamiento típico: 2-5 segundos por imagen en CPU

## Troubleshooting

**Out of memory:** Reducir batch size o usar GPU si está disponible.

**Modelo no carga:** Verificar conexión a HuggingFace (primera vez requiere internet).

**Timeout:** Configura `OCR_FORWARD_TIMEOUT_MS` en el backend si la inferencia supera el valor por defecto; revisa también timeouts del reverse proxy (Traefik/Cloudflare).
