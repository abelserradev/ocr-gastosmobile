"""
Servicio OCR para extracción de datos de facturas.
Usa Moondream para vision-language local y extracción de información.
"""

import io
import os
import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

app = FastAPI(
    title="OCR Service - Gastos",
    description="Extracción de datos de facturas usando Moondream",
    version="0.1.0",
)

# CORS para desarrollo (el backend NestJS llamará a este servicio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod: solo el backend NestJS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo Moondream (lazy loading)
_moondream_model = None
_moondream_tokenizer = None


def get_moondream_model():
    """Inicializa y retorna el modelo Moondream (singleton)."""
    global _moondream_model, _moondream_tokenizer
    if _moondream_model is None:
        try:
            from moondream import Moondream, detect_device

            device = detect_device()
            print(f"[OCR] Cargando Moondream en dispositivo: {device}")

            _moondream_model = Moondream.from_pretrained("vikhyatk/moondream2")
            _moondream_model = _moondream_model.to(device)
            _moondream_tokenizer = _moondream_model.tokenizer
        except Exception as e:
            print(f"[OCR] Error cargando Moondream: {e}")
            raise RuntimeError(f"No se pudo cargar Moondream: {e}")
    return _moondream_model, _moondream_tokenizer


class ParseInvoiceResponse(BaseModel):
    """Respuesta del análisis de factura."""

    amount: Optional[float] = Field(
        None, description="Monto total detectado en la factura"
    )
    date: Optional[str] = Field(
        None, description="Fecha detectada (YYYY-MM-DD)"
    )
    merchant: Optional[str] = Field(
        None, description="Nombre del comercio/establecimiento"
    )
    description: Optional[str] = Field(
        None, description="Descripción de items si está disponible"
    )
    raw_text: str = Field(
        ..., description="Texto crudo extraído de la imagen"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confianza general de la extracción (0-1)"
    )
    currency: str = Field(
        default="USD",
        description="Moneda detectada (USD, BS, etc.)"
    )


class HealthResponse(BaseModel):
    """Respuesta de health check."""

    status: str
    model_loaded: bool
    version: str


def extract_amount_from_text(text: str) -> tuple[Optional[float], str]:
    """
    Extrae el monto más probable del texto.
    Busca patrones como: $25.50, 25.50 USD, Total: 100, etc.
    Retorna (monto, moneda_detectada).
    """
    # Patrones comunes para montos
    patterns = [
        # Total con símbolo $ o USD al final
        r'(?:total|monto|amount|importe)[:\s]*[\$\s]*([\d,]+\.?\d*)\s*(?:usd|\$)?',
        # $XX.XX o XX.XX USD
        r'\$?\s*([\d,]+\.\d{2})\s*(?:usd|\$)?',
        # Patrón venezolano: Bs. XX.XXX,XX o XX.XXX,XX Bs.
        r'(?:bs\.?|bol[ií]vares?)[:\s]*([\d.,]+)\s*(?:bs\.?)?',
        # Número grande al final (probable total)
        r'(?:total|pagar)[:\s]*([\d,]+\.?\d*)',
    ]

    text_clean = text.lower().replace(',', '')

    for pattern in patterns:
        matches = re.findall(pattern, text_clean, re.IGNORECASE)
        if matches:
            # Tomar el último match (generalmente el total)
            try:
                amount_str = matches[-1]
                # Limpiar y convertir
                amount_str = amount_str.replace(',', '').strip()
                # Si tiene punto decimal, asumimos que es decimal
                amount = float(amount_str)
                if amount > 0:
                    # Detectar moneda
                    currency = "BS" if "bs" in text_clean or "bolívar" in text_clean else "USD"
                    return amount, currency
            except ValueError:
                continue

    return None, "USD"


def extract_date_from_text(text: str) -> Optional[str]:
    """
    Extrae fecha del texto en formato YYYY-MM-DD.
    Soporta formatos comunes: DD/MM/YYYY, DD-MM-YYYY, etc.
    """
    # Patrones de fecha
    patterns = [
        # DD/MM/YYYY o DD-MM-YYYY
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
        # YYYY-MM-DD
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                if len(match[2]) == 4:  # DD/MM/YYYY
                    day, month, year = int(match[0]), int(match[1]), int(match[2])
                else:  # YYYY-MM-DD
                    year, month, day = int(match[0]), int(match[1]), int(match[2])

                # Validar fecha
                dt = datetime(year, month, day)
                return dt.strftime("%Y-%m-%d")
            except (ValueError, IndexError):
                continue

    return None


def extract_merchant_from_text(text: str) -> Optional[str]:
    """
    Extrae el nombre del comercio/establecimiento.
    Busca líneas que parezcan nombres de negocio.
    """
    lines = text.strip().split('\n')

    # Palabras que indican que NO es el comercio
    exclude_words = [
        'factura', 'recibo', 'ticket', 'total', 'subtotal', 'iva',
        'fecha', 'hora', 'monto', 'cambio', 'efectivo', 'tarjeta',
        'gracias', 'vuelva', 'pronto', 'cliente', 'cajero',
        'telefono', 'direccion', 'rif', 'nit'
    ]

    # Buscar primera línea que parezca nombre de negocio
    for line in lines[:10]:  # Revisar primeras 10 líneas
        line_clean = line.strip()
        if len(line_clean) < 3 or len(line_clean) > 50:
            continue

        # No debe contener números de documento
        if re.search(r'\d{3,}', line_clean):
            continue

        # No debe contener palabras excluidas
        if any(word in line_clean.lower() for word in exclude_words):
            continue

        # Debe tener al menos 2 palabras o ser un nombre propio
        if len(line_clean.split()) >= 1:
            return line_clean.title()

    return None


def calculate_confidence(
    amount: Optional[float],
    date: Optional[str],
    merchant: Optional[str],
    raw_text: str
) -> float:
    """Calcula una puntuación de confianza basada en qué tanto extrajimos."""
    score = 0.0

    if amount is not None and amount > 0:
        score += 0.4
    if date is not None:
        score += 0.3
    if merchant is not None:
        score += 0.2

    # Bonus si el texto tiene buena longitud (indica OCR decente)
    if len(raw_text) > 50:
        score += 0.1

    return min(score, 1.0)


@app.post("/parse-invoice", response_model=ParseInvoiceResponse)
async def parse_invoice(file: UploadFile = File(..., description="Imagen de la factura")):
    """
    Recibe una imagen de factura y extrae: monto, fecha, comercio, descripción.
    """
    # Validar tipo de archivo
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {content_type}. Solo se aceptan imágenes."
        )

    try:
        # Leer imagen
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB max
            raise HTTPException(status_code=400, detail="Imagen demasiado grande (max 10MB)")

        image = Image.open(io.BytesIO(contents))

        # Convertir a RGB si es necesario
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Obtener modelo Moondream
        model, tokenizer = get_moondream_model()

        # Prompt optimizado para facturas
        prompt = """Analyze this invoice/receipt image and extract the following information:
        1. Total amount (look for "Total", "Monto", "Importe", "Amount")
        2. Date (in any format)
        3. Merchant/Store name (business name)
        4. Brief description of items if visible

        Respond in this exact format:
        TOTAL: [amount with currency]
        DATE: [date]
        MERCHANT: [store name]
        ITEMS: [brief list]

        If any information is not visible, write "NOT VISIBLE".
        """

        # Ejecutar Moondream
        answer = model.query(image, prompt, tokenizer)
        raw_text = answer if isinstance(answer, str) else str(answer)

        # Extraer datos estructurados
        amount, currency = extract_amount_from_text(raw_text)
        date = extract_date_from_text(raw_text)
        merchant = extract_merchant_from_text(raw_text)

        # Descripción: líneas que parezcan items
        description = None
        lines = raw_text.split('\n')
        for line in lines:
            if any(kw in line.lower() for kw in ['items:', 'description:', 'productos:']):
                description = line.split(':', 1)[-1].strip()
                if description == "not visible":
                    description = None
                break

        # Calcular confianza
        confidence = calculate_confidence(amount, date, merchant, raw_text)

        return ParseInvoiceResponse(
            amount=amount,
            date=date,
            merchant=merchant,
            description=description,
            raw_text=raw_text,
            confidence=confidence,
            currency=currency
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando imagen: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check - indica si el modelo está cargado."""
    model_loaded = _moondream_model is not None
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        version="0.1.0"
    )


@app.on_event("startup")
async def startup_event():
    """Precargar el modelo al iniciar (opcional, puede tardar)."""
    try:
        print("[OCR] Iniciando precarga del modelo Moondream...")
        get_moondream_model()
        print("[OCR] Modelo cargado exitosamente")
    except Exception as e:
        print(f"[OCR] Advertencia: No se pudo precargar el modelo: {e}")
        print("[OCR] El modelo se cargará en la primera solicitud")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
