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

**Timeout:** El backend NestJS tiene timeout de 30s; imágenes grandes pueden tardar más.
