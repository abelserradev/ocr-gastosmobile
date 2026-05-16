"""
Servicio OCR para extracción de datos de facturas.
Usa Moondream para vision-language local y extracción de información.
"""

import asyncio
import io
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import moondream as md
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from moondream.types import VLM

# Revisión fijada: empaquetados .mf.gz usados en PyPI moondream 0.0.5; si HF deja de servirla, definir MOONDREAM_HF_REVISION.
DEFAULT_MOONDREAM_MF_REVISION = "9dddae84d54db4ac56fe37817aeaeb502ed083e2"

_logger = logging.getLogger("ocr")
if not logging.root.handlers:
    logging.basicConfig(level=os.getenv("OCR_LOG_LEVEL", "INFO").upper())

_moondream_vlm: Optional[VLM] = None
_moondream_lock = threading.Lock()
_precache_finished: bool = False
_precache_error: Optional[str] = None


def _default_mf_filename_from_env() -> str:
    """0.5b por defecto en VPS; 2b si el operador pide más calidad y tiene RAM."""
    variant = os.getenv("MOONDREAM_ONNX_VARIANT", "0.5b").strip().lower()
    if variant in ("2b", "2", "large"):
        return "moondream-2b-int8.mf.gz"
    return "moondream-0_5b-int8.mf.gz"


def _create_moondream_vlm() -> VLM:
    """Construye el cliente VLM: nube, ruta local, o descarga HF + ONNX en CPU."""
    api_key = os.getenv("MOONDREAM_API_KEY", "").strip()
    if api_key:
        print("[OCR] Moondream: cliente nube (MOONDREAM_API_KEY)")
        return md.vl(api_key=api_key)

    explicit_path = os.getenv("MOONDREAM_MODEL_PATH", "").strip()
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise RuntimeError(
                f"MOONDREAM_MODEL_PATH no es un archivo válido: {explicit_path}"
            )
        print(f"[OCR] Moondream: ONNX desde {explicit_path}")
        return md.vl(model=explicit_path)

    repo_id = os.getenv("MOONDREAM_HF_REPO", "vikhyatk/moondream2").strip()
    revision = os.getenv(
        "MOONDREAM_HF_REVISION", DEFAULT_MOONDREAM_MF_REVISION
    ).strip()
    filename = os.getenv("MOONDREAM_HF_FILENAME", "").strip()
    if not filename:
        filename = _default_mf_filename_from_env()

    print(
        f"[OCR] Moondream: descarga/verificación HF repo={repo_id} file={filename} …"
    )
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )
    print(f"[OCR] Pesos listos en {weights_path}")
    return md.vl(model=weights_path)


def get_moondream_model() -> VLM:
    """Singleton del modelo; moondream 0.0.5 expone query(image, question) -> {'answer': str}."""
    global _moondream_vlm
    if _moondream_vlm is not None:
        return _moondream_vlm
    with _moondream_lock:
        if _moondream_vlm is not None:
            return _moondream_vlm
        try:
            _moondream_vlm = _create_moondream_vlm()
        except Exception as err:
            print(f"[OCR] Error inicializando Moondream: {err}")
            raise RuntimeError(f"No se pudo inicializar Moondream: {err}") from err
    return _moondream_vlm


async def _precargar_moondream_en_fondo() -> None:
    """HF + ONNX puede tardar minutos; no debe retrasar el accept() de Uvicorn ni el healthcheck del proxy."""
    global _precache_finished, _precache_error
    _precache_finished = False
    _precache_error = None
    try:
        print(
            "[OCR] Precarga Moondream en segundo plano (evita 502 del proxy mientras descarga/carga)…"
        )
        await asyncio.to_thread(get_moondream_model)
        print("[OCR] Modelo Moondream listo para inferencia")
    except Exception as err:
        msg = str(err)
        _precache_error = msg[:800] if len(msg) > 800 else msg
        print(
            f"[OCR] Precarga en fondo falló (reintento al llamar /parse-invoice): {err}"
        )
    finally:
        _precache_finished = True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setattr(
        _app.state,
        "moondream_warm_task",
        asyncio.create_task(_precargar_moondream_en_fondo()),
    )
    yield


app = FastAPI(
    title="OCR Service - Gastos",
    description="Extracción de datos de facturas usando Moondream",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS para desarrollo (el backend NestJS llamará a este servicio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod: solo el backend NestJS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    version: str
    precache_finished: bool = Field(
        ...,
        description="True cuando terminó el intento de precarga en segundo plano (éxito o error)",
    )
    precache_error: Optional[str] = Field(
        None,
        description="Si la precarga falló, mensaje breve; null si aún no terminó o fue OK",
    )


MAX_OCR_IMAGE_SIDE = max(768, min(4096, int(os.getenv("OCR_MAX_IMAGE_SIDE", "1600"))))
_LOG_RAW_CHARS = max(120, min(4000, int(os.getenv("OCR_LOG_RAW_MAX_CHARS", "600"))))


def resize_image_if_too_large(image: Image.Image) -> Image.Image:
    """Algunos VLM mejoran si el lado largo no supera OCR_MAX_IMAGE_SIDE (memoria + texto legible)."""
    width, height = image.size
    longest = max(width, height)
    if longest <= MAX_OCR_IMAGE_SIDE:
        return image
    ratio = MAX_OCR_IMAGE_SIDE / longest
    new_size = max(1, int(width * ratio)), max(1, int(height * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def is_low_signal_vlm_answer(raw: str) -> bool:
    """
    Moondream a veces responde con metalenguaje en inglés tipo "[Note: ...]"
    cuando en realidad no transcribe la foto; no debemos puntear eso como "texto OCR bueno".
    """
    condensed = raw.lower().strip()
    if len(condensed) < 15:
        return True
    bad_markers = (
        "[note:",
        "cannot read",
        "can't read",
        "can not see",
        "exact content is not visible",
        "image is blurry",
        "no visible text",
        "unable to ",
        "i cannot ",
    )
    if any(m in condensed for m in bad_markers):
        return True
    if condensed.startswith("[") and "note" in condensed[:80]:
        return True
    return False


def parse_money_fragment(fragment: str) -> tuple[Optional[float], str]:
    """
    Interpreta números con coma/punto americanos y estilo VE (miles por punto,
    decimal por coma): Bs 60.552,00 ó 60552,00.
    """
    s = fragment.strip()
    currency = _infer_currency_near_amount(s.lower())

    s = re.sub(r"(?i)\b(bs\.?|ves|usd)\b\s*", "", s)
    s = s.replace("$", "").replace("\u2009", "").replace(" ", "").strip()

    numeric = re.findall(
        r"[\d]{1,3}(?:[\.,]\d{3})+(?:[\.,]\d{1,4})?|\d+(?:[\.,]\d+)?",
        s,
        re.IGNORECASE,
    )
    if not numeric:
        return None, currency or "USD"
    candidate = numeric[-1]
    amount = parse_localized_money_token(candidate)
    if amount is None or amount <= 0:
        return None, currency or "USD"
    return amount, currency or "USD"


def parse_localized_money_token(token: str) -> Optional[float]:
    normalized = token.strip()
    # TODO: Soporte EUR con espacio NBSP si aparece en recibos importados

    decimal_comma_ve = bool(
        re.match(r"^[\d]{1,3}(?:\.[\d]{3})*,\d{1,4}$", normalized)
    )

    try:
        if decimal_comma_ve:
            main, frac = normalized.rsplit(",", 1)
            whole = main.replace(".", "")
            result = float(f"{whole}.{frac}")
            return result if result > 0 else None

        decimal_dot_us = bool(re.match(r"^[\d,]+\.\d{1,4}$", normalized))

        only_dots = "." in normalized and "," not in normalized

        last_dot_idx = normalized.rfind(".")
        rest_after_last_dot = (
            normalized[last_dot_idx + 1 :] if last_dot_idx != -1 else ""
        )

        ambiguous_dot_as_decimal = bool(
            only_dots and last_dot_idx != -1 and len(rest_after_last_dot) <= 2
        )

        if decimal_dot_us or ambiguous_dot_as_decimal:
            return float(normalized.replace(",", ""))

        if "," in normalized and "." not in normalized:
            stripped = normalized.replace(",", "")
            return float(stripped)

        if "." in normalized and "," in normalized:
            if normalized.rfind(".") > normalized.rfind(","):
                return float(normalized.replace(",", ""))
            return parse_localized_money_token(normalized.replace(".", "").replace(",", "."))

        stripped_all = normalized.replace(",", "").replace(".", "")
        if stripped_all.isdigit():
            raw_f = float(stripped_all)
            return raw_f if raw_f > 0 else None
    except (ValueError, ArithmeticError):
        return None

    return None


def _infer_currency_near_amount(fragment_lower: str) -> str:
    if re.search(r"(bs\.?|bol[ií]v|ves\b)", fragment_lower, re.I):
        return "BS"
    if re.search(r"(usd|\$)", fragment_lower, re.I):
        return "USD"
    return ""


def extract_structured_fields(text: str) -> tuple[
    Optional[float], str, Optional[str], Optional[str], Optional[str]
]:
    """
    Prefiere etiquetas TOTAL:/DATE:/MERCHANT: que el prompt pide; evita regex frágiles
    sobre párrafos libres cuando el modelo colabora.
    """
    amount: Optional[float] = None
    currency_found = ""
    merchant_f: Optional[str] = None
    date_str: Optional[str] = None
    items_chunk: Optional[str] = None

    for rx, key in (
        (
            r"(?ims)^TOTAL(?:\s+A\s+PAGAR|\s+PAGADO)?\s*[:\.]?\s*(.+)$",
            "total",
        ),
        (r"(?ims)^DATE\s*[:\.]?\s*(.+)$", "date"),
        (r"(?ims)^MERCHANT\s*[:\.]?\s*(.+)$", "merchant"),
        (r"(?ims)^(?:FECHA)\s*[:\.]?\s*(.+)$", "date_es"),
        (
            r"(?ims)^(?:COMERCIO|TIENDA|NEGOCIO|RAZÓN\s+SOCIAL)"
            r"\s*[:\.]?\s*(.+)$",
            "merchant_es",
        ),
        (r"(?ims)^ITEMS\s*[:\.]?\s*(.+)$", "items_en"),
        (r"(?ims)^ART[IÍ]CULOS\s*[:\.]?\s*(.+)$", "items_es"),
        (r"(?ims)^(?:DESCRIPCIÓN|DESCRIPCION)\s*[:\.]?\s*(.+)$", "desc"),
    ):
        finds = list(re.finditer(rx, text))
        if not finds:
            continue
        last = finds[-1]
        chunk = last.group(1).strip()

        lowered = chunk.lower()
        invisible = lowered in {"not visible", "n/a", "na", "", "nv", "---"}
        visible_no = lowered.startswith(("not visible", "cannot", "unable"))
        if invisible or visible_no:
            continue

        if key in ("merchant", "merchant_es"):
            stripped = strip_trailing_instructions(chunk)
            if len(stripped) > 2:
                merchant_f = stripped.strip()
                if len(merchant_f) > 120:
                    merchant_f = merchant_f[:120]

        elif key in ("date", "date_es"):
            normalized = normalize_structured_date_line(chunk)
            if normalized:
                date_str = normalized

        elif key == "total":
            amt2, curr2 = parse_money_fragment(chunk)
            if amt2 is not None and amt2 > 0:
                amount = amt2
                currency_found = curr2 or currency_found or "USD"

        elif key in ("items_en", "items_es", "desc"):
            stripped_meta = strip_trailing_instructions(chunk)
            low_meta = stripped_meta.lower()
            if low_meta.startswith(("not visible", "cannot")):
                continue
            cand = stripped_meta.strip()
            if len(cand) < 4:
                continue
            pick = cand if cand else None
            if pick and (
                items_chunk is None or len(pick) > len(items_chunk or "")
            ):
                items_chunk = pick[:400]

    return amount, currency_found, merchant_f, date_str, items_chunk


def normalize_structured_date_line(chunk: str) -> Optional[str]:
    trimmed = strip_trailing_instructions(chunk)
    iso = extract_date_from_text(trimmed)
    return iso


def strip_trailing_instructions(fragment: str) -> str:
    """Quita coma final o texto pegado después de la primera fecha en una línea."""
    cleaned = fragment.split("(", 1)[0].strip()
    parts = cleaned.split(",", 1)
    if parts and looks_like_calendar_bit(parts[0]):
        candidate = parts[0].strip()
        return candidate
    return cleaned.strip()


def looks_like_calendar_bit(s: str) -> bool:
    return bool(re.search(r"\d{1,4}[/\-\.]\d{1,4}[/\-\.]\d{2,4}", s))


def extract_amount_from_text(text: str) -> tuple[Optional[float], str]:
    """
    Extrae el monto más probable del texto sin romper formato venezolano (punto=miles).
    Combina TOTAL / palabras clave y captura cercana a símbolo Bs ó $.
    """
    currency_seen = infer_currency_hint_from_context(text.lower())

    for pattern in (
        r"(?i)(?:total\s+a\s+pagar|total\s+pagado|total\s+factura"
        r"|monto\s+total|gran\s+total|importe\s+total)\s*[:\.]?\s*([^\n\r]{1,120})",
        r"(?i)(?:total|importe\s+factura)[:\.]?\s*Bs\.?\s*([\d\s.,]{3,42})",
        r"(?i)Bs\.?\s*([\d][\d\s.,]{2,41})",
        r"(?i)\$\s*([\d][\d\s.,]{1,41})",
        r"(?i)(?:pagar|cambio|cobr(?:ar)?)\s*[:\.]?\s*([^\n\r]{1,96})",
    ):
        chunks = list(re.finditer(pattern, text))
        if not chunks:
            continue

        cand_text = chunks[-1].group(1).strip()
        amt_ok, curr = parse_money_fragment(cand_text)
        if amt_ok is not None:
            cur_out = (curr.strip() if curr else "") or currency_seen or "USD"
            return amt_ok, cur_out or "USD"

    stray_amt, stray_cur = parse_money_fragment(text)
    fallback_cur = infer_currency_hint_from_context(text.lower()) or "USD"
    if stray_amt is None:
        return None, fallback_cur
    return stray_amt, (stray_cur or currency_seen or fallback_cur)


def infer_currency_hint_from_context(low: str) -> str:
    """Heurística laxa cuando el modelo no etiqueta moneda cerca del monto."""
    if re.search(r"\bbs(?:\.|,|\s)?", low):
        return "BS"
    if re.search(r"\bbol[ií]v", low):
        return "BS"
    if re.search(r"\busd\b", low) or "$" in low:
        return "USD"
    return ""


def resolve_invoice_currency(raw_text_lc: str, *candidates: Optional[str]) -> str:
    for picked in candidates:
        if isinstance(picked, str) and picked.upper() == "BS":
            return "BS"
    for cand2 in candidates:
        if cand2:
            trimmed = cand2.strip()
            if trimmed:
                return trimmed.upper()
    hint = infer_currency_hint_from_context(raw_text_lc)
    return hint if hint else "USD"




def extract_date_from_text(text: str) -> Optional[str]:
    """
    Extrae fecha del texto en formato YYYY-MM-DD.
    Soporta formatos comunes: DD/MM/YYYY, DD-MM-YYYY, etc.
    """
    # Patrones de fecha (latinos incluyen punto)
    patterns = [
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})",
        r"(\d{1,2})\.(\d{1,2})\.(\d{4})",
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
        r'(\d{4})\.(\d{1,2})\.(\d{1,2})',
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
    Si el VLM no devolvió MERCHANT: estructurado, tomamos la primera línea “de encabezado”
    que no sea metadato del formulario.
    """
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]

    label_rx = re.compile(
        r"(?ix)^\s*(merchant|fecha|date|total|items|articulos|descripción|descripcion)\s*[:\.]"
    )

    exclude_words = [
        "factura comercial",
        "factura de venta",
        "recibo de pago",
        "subtotal",
        "iva",
        "cambio",
        "gracias",
        "vuelva pronto",
    ]

    for line_clean in lines[:14]:
        lc = line_clean.lower()
        if "[note" in lc or "not visible" in lc or lc.startswith(("unable", "cannot")):
            continue
        if label_rx.match(line_clean):
            continue

        chunk_len = len(line_clean)
        if chunk_len < 4 or chunk_len > 96:
            continue

        lowered_all = "".join(lc.split())
        if any(bad.replace(" ", "") in lowered_all for bad in exclude_words):
            continue

        numeric_density = sum(1 for char in line_clean if char.isdigit())
        # RIF/control fiscal largos: demasiados dígitos para ser solo razón social
        if numeric_density >= max(10, chunk_len // 4):
            continue

        spaced = (
            line_clean.strip()
            if any(tag in lc for tag in ["c.a.", "s.a.", ",", ".,"])
            else line_clean.title()
        )
        return spaced

    return None


def invoice_transcript_bonus_ok(raw_text: str) -> bool:
    """
    Solo damos puntos extra cuando el modelo devolvió un bloque rico típico de facturas,
    pero la heurística de campos quizá no llegó — evita falsos positivos tipo “notes” cortas.
    """
    stripped = raw_text.strip()
    lc = stripped.lower()
    cues = (
        "total",
        "bs.",
        "bs ",
        "bolívar",
        "bolivar",
        "usd",
        "importe",
        "factura",
        "ticket",
        "rif",
        "iva",
        "$",
    )

    numeric_spread_ok = bool(
        re.search(r"\d[\d\s./,-]{10,}", stripped, re.MULTILINE)
    )
    lexical_hit = sum(1 for chunk in cues if chunk in lc) >= 1
    headerish = ("\n" in stripped or stripped.count(":") >= 2)
    length_ok = len(stripped) >= 110
    return length_ok and numeric_spread_ok and lexical_hit and headerish


def calculate_confidence(
    amount: Optional[float],
    date: Optional[str],
    merchant: Optional[str],
    raw_text: str,
    description: Optional[str],
) -> float:
    score = 0.0

    if amount is not None and amount > 0:
        score += 0.35
    if date is not None:
        score += 0.3
    if merchant is not None:
        score += 0.2
    if description:
        score += 0.05

    # Bonus sólo ante transcripciones ricas; antes un párrafo “Note:” alto valía igual que OCR real.
    transcript_signal = invoice_transcript_bonus_ok(raw_text) and (
        not is_low_signal_vlm_answer(raw_text)
    )
    if transcript_signal:
        score += 0.1

    return min(score, 1.0)


INVOICE_VLM_PROMPT = """Lee la foto de una factura, ticket comercial o nota fiscal REAL.

PASO CRÍTICO (español + instrucciones mínimas en inglés):

1. Antes que nada TRANSCRIBE el texto impreso (negocio, dirección corta si ayuda a identificar la tienda,
   número de documento cuando exista y el TOTAL FINAL). No improvises historia ni descripciones
   tipo “hay varias líneas”.

2. NUNCA uses frases tipo “cannot read”, “[Note:]”, “the exact content is not visible” ni disculpas meta.
   Si algo no existe en papel, pon NOT VISIBLE en la línea exacta solicitada más abajo.

3. FINAL OUTPUT (copia estos encabezados tal cual):

TOTAL:
DATE:
MERCHANT:
ITEMS:

Formato después de TOTAL: incluye símbolo o moneda (ej. “Bs 60.552,00”).
DATE acepta el formato DD/MM/AAAA que veas escrito exactamente igual.
MERCHANT debe ser razón social o nombre comercial real (no texto genérico).
ITEMS: máximo tres ítems o “NOT VISIBLE”.

RULES REMINDER:

• English summary: behave like OCR + bookkeeping — copy numbers from the slip, forbid editorial notes.
"""


@app.post("/parse-invoice", response_model=ParseInvoiceResponse)
async def parse_invoice(file: UploadFile = File(..., description="Imagen de la factura")):
    """
    Recibe una imagen de factura y extrae: monto, fecha, comercio, descripción.
    """
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail="Imagen demasiado grande (max 10MB)"
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception:
        declared = file.content_type or "sin Content-Type"
        raise HTTPException(
            status_code=400,
            detail=(
                f"No se pudo abrir la imagen (tipo declarado: {declared}). "
                "Usa JPG, PNG o WebP."
            ),
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    try:
        vlm = get_moondream_model()
        image_for_model = resize_image_if_too_large(image)

        def _invoke_moondream_parsing():
            """Evita trabar FastAPI cuando query() síncrono tarda sobre CPU ONNX."""
            return vlm.query(image_for_model, INVOICE_VLM_PROMPT)

        query_result = await asyncio.to_thread(_invoke_moondream_parsing)
        raw_text = (
            query_result.get("answer", "")
            if isinstance(query_result, dict)
            else str(query_result)
        )

        log_snippet = re.sub(r"\s+", " ", raw_text.strip())[:_LOG_RAW_CHARS]
        _logger.info("Moondream answer (primeros caracteres=%s): %s", len(raw_text), log_snippet)

        (
            structured_amount,
            structured_currency_hint,
            structured_merchant,
            structured_date,
            structured_items,
        ) = extract_structured_fields(raw_text)

        heuristic_amount, heuristic_currency = extract_amount_from_text(raw_text)

        amount_pick: Optional[float] = (
            structured_amount
            if structured_amount is not None and structured_amount > 0
            else heuristic_amount
        )

        raw_lc_compact = raw_text.lower()

        structured_cur_pick = (
            structured_currency_hint.strip()
            if (
                structured_amount is not None
                and structured_amount > 0
                and structured_currency_hint.strip()
            )
            else None
        )
        heuristic_cur_pick = (
            heuristic_currency.strip()
            if (
                heuristic_amount is not None
                and heuristic_currency.strip()
                and heuristic_amount > 0
            )
            else None
        )

        currency_pick = resolve_invoice_currency(
            raw_lc_compact,
            structured_cur_pick,
            heuristic_cur_pick,
        )

        merchant_pick = structured_merchant or extract_merchant_from_text(raw_text)
        date_pick = structured_date or extract_date_from_text(raw_text)

        description_pick: Optional[str] = structured_items
        if description_pick is None:
            keyed_labels = frozenset({"items", "description", "productos", "articulos"})
            for line in raw_text.split("\n"):
                trimmed = line.strip()
                if ":" not in trimmed:
                    continue
                heading, sep, tail = trimmed.partition(":")
                if not sep:
                    continue
                label_lc = heading.strip().lower()
                if label_lc not in keyed_labels:
                    continue

                cleaned_tail = tail.strip()
                lc_tail = cleaned_tail.lower()
                if lc_tail in {"not visible", "nv", "---", ""}:
                    continue

                description_pick = cleaned_tail[:520]
                break

        confidence = calculate_confidence(
            amount_pick,
            date_pick,
            merchant_pick,
            raw_text,
            description_pick,
        )

        return ParseInvoiceResponse(
            amount=amount_pick,
            date=date_pick,
            merchant=merchant_pick,
            description=description_pick,
            raw_text=raw_text,
            confidence=confidence,
            currency=currency_pick,
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
    model_loaded = _moondream_vlm is not None
    err = _precache_error if _precache_finished and not model_loaded else None
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        version="0.1.0",
        precache_finished=_precache_finished,
        precache_error=err,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
