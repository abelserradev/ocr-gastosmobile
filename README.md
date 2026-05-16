# Servicio OCR - Gastos

Servicio Python (FastAPI) para facturas: **Tesseract** + **Moondream** (híbrido, texto legible en `raw_text`).

## ¿Qué hace?

- Recibe imágenes de facturas (JPG, PNG, WebP)
- **Híbrido:** **Tesseract** (OCR clásico, `spa+eng`) + **Moondream** (VLM); los campos se fusionan con **preferencia por Tesseract** cuando aporta texto sustancial
- Extrae: **monto**, **fecha**, **comercio**, **descripción**
- En la respuesta, `raw_text` combina ambas fuentes con cabeceras `# Tesseract (OCR)` y `# Moondream (VLM)` para depuración

## Requisitos

- Python 3.10+
- ~4GB RAM (Moondream + Tesseract en CPU)
- **Tesseract** en el sistema: en Docker ya se instala (`tesseract-ocr`, `spa`/`eng`); en local: `sudo apt install tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng` (Debian/Ubuntu)

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

## Docker local (`docker-compose.local.yml`)

En el repo hay un compose pensado para desarrollo en tu PC:

```bash
cd ocr/
docker compose -f docker-compose.local.yml up --build
```

Usa **`USE_GPU=0`** (inferencia en **CPU** dentro del contenedor). Las GPUs **AMD** (p. ej. RX 580) **no ejecutan CUDA** (CUDA es de NVIDIA); el paquete `onnxruntime-gpu` de esta imagen está orientado a NVIDIA. ROCm en AMD es otro stack y Polaris/RX 580 suele quedar mal soportado para este pipeline. Los detalles están comentados en `docker-compose.local.yml`.

### Verificación local (antes de desplegar al servidor)

1. Levantá el stack y esperá a que el health sea estable (la primera vez Moondream descarga pesos; puede tardar minutos).
2. Ejecutá el script (health + factura opcional):

```bash
cd ocr/
chmod +x scripts/verify-local.sh   # una vez
./scripts/verify-local.sh
VERIFY_INVOICE_IMAGE=/ruta/a/tu-factura.jpg ./scripts/verify-local.sh
```

Variables opcionales: `OCR_BASE_URL` si el puerto no es 8001.

3. **Criterios de aprobación** (revisión manual del JSON):
   - `amount` / `merchant` / `currency` coherentes con el ticket.
   - `raw_text` incluye `# Tesseract (OCR)` con líneas legibles; Moondream puede ser auxiliar.
   - `confidence` acorde (no forzar si el ticket es ilegible).

4. Misma imagen Docker / mismas variables en producción → despliegue alineado con lo probado localmente.

## Docker (NVIDIA GPU en el host)

El servicio vive en esta carpeta (`ocr/`). El código Nest que llama al OCR está en `backend/src/ocr/` (no es este contenedor).

**Requisitos:** Docker, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) y driver propio del host (p. ej. Quadro P600).

```bash
cd ocr/
docker compose up --build -d
# Health: curl -s http://localhost:8001/health
```

- Caché de Hugging Face en el volumen `huggingface_ocr_cache` (no re-descarga el `.mf.gz` en cada `docker compose down` sin `-v`).
- **`MOONDREAM_ONNX_VARIANT=0.5b`** recomendable si la GPU tiene poca VRAM (p. ej. 2GB): variable de entorno en Coolify o `.env`.
- **GPU (servidor NVIDIA):** `build.target: gpu` usa la misma base **slim** + `onnxruntime-gpu` y wheels `nvidia-cublas-cu12` (evita descargar ~2 GB de `nvidia/cuda` en cada deploy de Coolify). Sigue necesitando `deploy.resources` / `--gpus all` y drivers en el host.
- **Coolify build cortado al 50–90 s:** suele ser timeout o disco al bajar `nvidia/cuda`; con el Dockerfile actual el build es mucho más pequeño. Si falla igual, sube el timeout de build en Coolify o haz `docker compose build` por SSH en el servidor.
- **Solo CPU:** `OCR_DOCKER_TARGET=cpu` en el build o `docker-compose.local.yml` (`target: cpu`) y comenta `deploy.resources.reservations.devices`.
- **Coolify + GPU:** si el validador falla con `deploy`, Custom Docker Options: `--gpus all`. Rebuild sin caché tras cambiar el Dockerfile.
- **Cloudflare:** el proxy naranja corta ~100s; inferencias largas pueden necesitar DNS only, async job o API Moondream en nube (`MOONDREAM_API_KEY`).

## Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/parse-invoice` | Recibe imagen multipart, retorna datos extraídos |
| GET | `/health` | Comprueba el servicio; `model_loaded` true cuando Moondream ya está en memoria; `precache_finished` / `precache_error` ayudan a diagnosticar la precarga en segundo plano |

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
