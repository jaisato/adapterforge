"""Síntesis de pares instrucción->respuesta anclados al documento.

Dos modos:

- **Online** (por defecto): un modelo *teacher* (configurable vía litellm o
  endpoint OpenAI-compatible) genera la respuesta a partir de las plantillas
  versionadas en ``prompts/``. Anclar cada ejemplo al documento reduce la
  alucinación del teacher.
- **Offline** (``--offline``): generadores determinísticos que construyen la
  respuesta a partir de la estructura del propio documento (artículos, frases
  con obligaciones, plazos y sanciones). Sin API ni red: sirve para generar el
  dataset de muestra, para los tests y para entornos sin conectividad.

Principios v3 (lecciones de la autopsia de v1/v2/v2b):

- **Ventana estructural, no prefijo**: los detectores corren sobre el
  articulado completo y el input adjunta solo la cabecera + los artículos que
  contienen el dato, dentro de un presupuesto de tokens medido con el
  tokenizer real (:mod:`adapterforge.data.budget`). Nada de
  ``texto[:4500]`` cayendo en el preámbulo.
- **Anclaje garantizado**: ningún ejemplo se emite si su valor objetivo no
  aparece literalmente en la ventana que ve el modelo.
- **Etiquetas validadas**: un "plazo" debe ser una duración o fecha real
  (número + unidad temporal, o fecha), una "multa" debe llevar cuantía; las
  capturas desbordadas y los «(no se detectan)» se descartan.
- **Diversidad**: pools de instrucciones amplios muestreados por ejemplo (no
  por documento), variantes con y sin cita, y variantes JSON estilo
  benchmark («Responde únicamente con un objeto JSON con la clave X»).
- **Mezcla equilibrada**: ``--balance`` re-muestrea por tarea hacia la
  distribución del benchmark (extract/qa/resumen dominan).

Tareas soportadas (una plantilla por tarea en ``prompts/``):

- ``resumen``: prosa de un párrafo (mayoría) o ficha estructurada.
- ``qa``: preguntas/respuestas por artículo con cita [BOE-A-..., art. N].
- ``obligaciones``: extracción de obligaciones/plazos/sanciones (JSON/lista/prosa).
- ``llano``: reescritura en lenguaje llano para no juristas.
- ``extraccion``: dato puntual con respuesta corta (frase o JSON verbatim).

Uso CLI::

    python -m adapterforge.data.synthesize \
        --input data/interim/documents.jsonl \
        --output data/interim/sft_raw.jsonl --offline

TODO(jairo): variante Evol-Instruct (mutar instrucciones) y self-instruct con
semillas humanas para aumentar diversidad del dataset online.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from adapterforge.data.budget import TokenCounter, assert_token_budget
from adapterforge.data.extract import (
    Article,
    RawDocument,
    load_corpus,
    split_articles,
    split_sentences,
)
from adapterforge.jsonl import write_jsonl
from adapterforge.settings import get_settings

SFT_TASKS = ("resumen", "qa", "obligaciones", "llano", "extraccion")

# Mezcla objetivo por tarea, alineada con el domain_bench (40 % extract,
# 32 % qa, 28 % resumen) más las tareas de estilo del producto.
DEFAULT_BALANCE = {
    "extraccion": 0.30,
    "qa": 0.25,
    "resumen": 0.20,
    "llano": 0.15,
    "obligaciones": 0.10,
}

# Presupuesto de tokens de la ventana de documento (cabecera + artículos).
# Con sistema (~60) + instrucción (~40) + respuesta (~250-500) + plantilla,
# deja el render completo holgadamente bajo seq_len 2048.
DEFAULT_WINDOW_TOKENS = 1100


def _variant(doc_id: str, n: int, salt: str = "") -> int:
    """Índice de variante determinista por documento (reproducible)."""
    return zlib.crc32(f"{doc_id}:{salt}".encode()) % n


def _example_rng(seed: int, *keys: str) -> random.Random:
    """RNG determinista por ejemplo (no por documento): rompe el monocultivo."""
    return random.Random(f"{seed}|" + "|".join(keys))


# Marcadores léxicos para detectar obligaciones/plazos/sanciones en texto
# jurídico-administrativo. Deliberadamente simples y auditables.
OBLIGATION_MARKERS = (
    "deberá",
    "deberán",
    "estarán obligad",
    "están obligad",
    "queda prohibid",
    "no podrá",
    "habrán de",
    "tendrán que",
)
SANCTION_MARKERS = ("multa", "sanción", "sancionad", "infracción", "reintegro")
_DEADLINE_PATTERNS = (
    re.compile(r"plazo(?: máximo)? de ([^,.;]+)"),
    re.compile(r"antes del (\d[^,.;]*)"),
    re.compile(r"en el término de ([^,.;]+)"),
)

_MONTHS_ES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)

# Validación de etiquetas de extracción: una duración es número/numeral +
# unidad temporal (con calificativo opcional); una fecha es «D de <mes>
# (de AAAA)?». Todo lo demás es captura desbordada de la regex y se descarta.
_NUMERAL_WORDS = (
    r"(?:un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|"
    r"doce|trece|catorce|quince|dieciséis|dieciseis|diecisiete|dieciocho|"
    r"diecinueve|veinte|veintiún|veintiun|treinta|cuarenta|cincuenta|sesenta|"
    r"setenta|ochenta|noventa|cien|ciento|mil|\d+)"
)
_TIME_UNITS = r"(?:día|días|dia|dias|mes|meses|año|años|semana|semanas|hora|horas)"
_QUALIFIERS = r"(?:\s+(?:hábiles|habiles|naturales|lectivos))?"
_DURATION_RE = re.compile(
    rf"^{_NUMERAL_WORDS}(?:\s+{_NUMERAL_WORDS})?\s+{_TIME_UNITS}{_QUALIFIERS}$",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    rf"^(?:el\s+)?\d{{1,2}}\s+de\s+(?:{'|'.join(_MONTHS_ES)})(?:\s+de\s+\d{{4}})?$",
    re.IGNORECASE,
)
# Cuantías monetarias («7.500 euros», «600.000 €» y la forma BOE de los
# millones: «900.000 de euros», «2.000.000 de euros») para multa_maxima.
_AMOUNT_RE = re.compile(r"\d[\d.,]*\s*(?:de\s+)?(?:euros|€)")

# Conectores anafóricos: una frase que empieza así no funciona como
# respuesta autónoma de extracción («Asimismo, deberán...» ¿quiénes?).
_CONNECTOR_START = re.compile(
    r"^(?:asimismo|además|adicionalmente|también|igualmente|del mismo modo|"
    r"por su parte|no obstante|sin embargo|en su caso)\b",
    re.IGNORECASE,
)
# Marcadores de lista embebidos («a) ... b) ...») dentro de una "frase".
_EMBEDDED_LIST_MARKER = re.compile(r"\n\s*[a-z]\)\s")

# Glosario para la reescritura offline en lenguaje llano (orden importa:
# primero las expresiones más largas).
PLAIN_GLOSSARY: tuple[tuple[str, str], ...] = (
    ("estarán obligadas a", "tienen que"),
    ("estarán obligados a", "tienen que"),
    ("la presente disposición", "esta norma"),
    ("La presente resolución", "Esta resolución"),
    ("La presente norma", "Esta norma"),
    ("La presente orden", "Esta orden"),
    ("La presente ley", "Esta ley"),
    ("tiene por objeto", "sirve para"),
    ("de conformidad con", "según"),
    ("con carácter previo", "antes"),
    ("sin perjuicio de", "además de"),
    ("a tal efecto", "para ello"),
    ("en el plazo de", "en un máximo de"),
    ("No obstante", "Sin embargo"),
    ("no obstante", "sin embargo"),
    ("deberán", "tienen que"),
    ("deberá", "tiene que"),
)

_RANGO_ARTICLE = {
    "ley": "una ley",
    "real decreto": "un real decreto",
    "real decreto-ley": "un real decreto-ley",
    "orden ministerial": "una orden ministerial",
    "orden": "una orden ministerial",
    "resolución": "una resolución",
}

_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")

# Rangos gramaticalmente femeninos ("la Ley", "de la Orden"...).
_FEMININE_RANKS = {"ley", "orden", "orden ministerial", "resolución"}

# Artículo del órgano emisor según su primera palabra ("el Ministerio...").
_ORGAN_ARTICLES = {
    "ministerio": "el ",
    "jefatura": "la ",
    "servicio": "el ",
    "secretaría": "la ",
    "agencia": "la ",
    "instituto": "el ",
}

_TERMINAL_CHARS = (".", "!", "?", "»", "]", ")")


def format_date_es(iso_date: str) -> str:
    """'2026-03-18' -> '18 de marzo de 2026' (para el lenguaje llano)."""
    year, month, day = iso_date.split("-")
    return f"{int(day)} de {_MONTHS_ES[int(month) - 1]} de {int(year)}"


def _short_ref(doc: RawDocument, *, preceded_by_de: bool = False) -> str:
    """Referencia corta con artículo correcto: 'el Real Decreto X' / 'de la Ley Y'."""
    feminine = doc.rango.lower() in _FEMININE_RANKS
    if preceded_by_de:
        return ("de la " if feminine else "del ") + doc.short_title()
    return ("la " if feminine else "el ") + doc.short_title()


def _organ_with_article(organ: str) -> str:
    """Antepone el artículo al órgano emisor cuando se conoce su género."""
    first_word = organ.split()[0].lower() if organ.split() else ""
    return _ORGAN_ARTICLES.get(first_word, "") + organ


def _ends_sentence(text: str) -> bool:
    """Comprueba que un target de prosa termina en frontera de frase."""
    stripped = text.rstrip()
    return bool(stripped) and stripped.endswith(_TERMINAL_CHARS)


class SftExample(BaseModel):
    """Ejemplo instruccional previo al formateo chat."""

    model_config = ConfigDict(extra="forbid")

    id: str
    doc_id: str
    task: str
    instruction: str
    input: str
    response: str
    source: str = Field(description="'offline' o nombre del modelo teacher")


class DpoPair(BaseModel):
    """Par de preferencia (chosen/rejected) previo al formateo chat."""

    model_config = ConfigDict(extra="forbid")

    id: str
    doc_id: str
    task: str
    instruction: str
    input: str
    chosen: str
    rejected: str
    source: str = "offline"


# ---------------------------------------------------------------------------
# Detectores estructurales (sobre el articulado completo, con validación)
# ---------------------------------------------------------------------------


class DeadlineHit(NamedTuple):
    """Plazo validado: valor verbatim, tipo, ÍNDICE del artículo y frase.

    ``article`` es el índice en la lista de artículos, no el número del
    artículo: los BOE que modifican otras normas repiten números («Artículo
    1.» aparece dos veces en 155/872 docs del corpus) y un dict por número
    ensombrecería artículos. ``sentence`` (la frase que contiene el plazo)
    permite derivar el PROPÓSITO del plazo para preguntas desambiguadas
    cuando el documento contiene varios plazos.
    """

    value: str
    kind: str  # "duracion" | "fecha"
    article: int
    sentence: str = ""


class SentenceHit(NamedTuple):
    """Frase detectada (obligación/sanción) e índice del artículo."""

    sentence: str
    article: int


def classify_deadline(value: str) -> str | None:
    """'tres meses' -> 'duracion'; '31 de diciembre de 2026' -> 'fecha'; else None."""
    cleaned = " ".join(value.split())
    if len(cleaned.split()) > 7:
        return None  # captura desbordada de la regex
    if _DURATION_RE.match(cleaned):
        return "duracion"
    if _DATE_RE.match(cleaned):
        return "fecha"
    return None


# Prefijos válidos dentro de una captura desbordada: «tres meses para
# presentar las solicitudes» -> valor «tres meses» + contexto de propósito.
_DURATION_PREFIX = re.compile(
    rf"^({_NUMERAL_WORDS}(?:\s+{_NUMERAL_WORDS})?\s+{_TIME_UNITS}{_QUALIFIERS})(?![\wáéíóúñ])",
    re.IGNORECASE,
)
_DATE_PREFIX = re.compile(
    rf"^((?:el\s+)?\d{{1,2}}\s+de\s+(?:{'|'.join(_MONTHS_ES)})(?:\s+de\s+\d{{4}})?)(?![\wáéíóúñ])",
    re.IGNORECASE,
)


def extract_deadline_value(raw: str) -> tuple[str, str] | None:
    """Recorta una captura al valor de plazo válido: (valor, tipo) o None.

    La regex de plazos captura hasta la siguiente puntuación, así que en
    «plazo de tres meses para presentar las solicitudes» la captura entera
    no valida; su prefijo «tres meses» sí, y el resto es el contexto que
    permite derivar el propósito de la pregunta.
    """
    cleaned = " ".join(raw.split())
    match = _DURATION_PREFIX.match(cleaned)
    if match:
        return match.group(1), "duracion"
    match = _DATE_PREFIX.match(cleaned)
    if match:
        return match.group(1), "fecha"
    return None


def find_deadline_hits(articles: list[Article], limit: int = 6) -> list[DeadlineHit]:
    """Plazos validados con el índice de su artículo y la frase de origen."""
    hits: list[DeadlineHit] = []
    seen: set[str] = set()
    for index, article in enumerate(articles):
        for sentence in split_sentences(article.body):
            for pattern in _DEADLINE_PATTERNS:
                for match in pattern.finditer(sentence):
                    extracted = extract_deadline_value(match.group(1))
                    if extracted is None:
                        continue
                    value, kind = extracted
                    if value.lower() in seen:
                        continue
                    seen.add(value.lower())
                    hits.append(
                        DeadlineHit(value=value, kind=kind, article=index, sentence=sentence)
                    )
    return hits[:limit]


# Rúbricas de artículo demasiado genéricas para servir de propósito.
_GENERIC_HEADINGS = {
    "objeto", "ambito de aplicacion", "ámbito de aplicación",
    "definiciones", "entrada en vigor", "disposiciones generales",
}


# Palabras funcionales que no pueden cerrar un propósito (corte a 8 palabras
# colgante: «para desarrollar dicho plan y»).
_DANGLING_WORDS = frozenset(
    "y o u e ni no de del al la el los las que en a su se para con por sin".split()
)
# Un propósito que contiene OTRO plazo produce ecos autocontradictorios
# («El plazo para su informe en el plazo de un mes es de diez días»).
_PURPOSE_BLOCKLIST = re.compile(
    r"\b(?:plazo|d[ií]as?|mes(?:es)?|años?|semanas?|horas?|dich[oa]s?)\b",
    re.IGNORECASE,
)


def _trim_purpose(text: str) -> str | None:
    """Acota el propósito: corta en «desde/a partir/a contar» y a 8 palabras.

    Rechaza propósitos que embeben otro plazo o anáforas («dicho plan») y
    despoja las palabras funcionales colgantes tras el corte (hallazgos de
    la revisión v5: 3/120 propósitos truncados o autocontradictorios).
    """
    text = re.split(r"\s+(?:desde|a partir|a contar|contados?)\b", text)[0].strip()
    if _PURPOSE_BLOCKLIST.search(text):
        return None
    words = text.split()[:8]
    while words and words[-1].lower() in _DANGLING_WORDS:
        words.pop()
    if len(words) < 2:
        return None
    return " ".join(words)


def deadline_purpose(hit: DeadlineHit, articles: list[Article]) -> str | None:
    """Frase de propósito de un plazo («para presentar las solicitudes»).

    Necesaria cuando el documento contiene VARIOS plazos: una pregunta
    genérica («¿qué plazo se establece?») sería ambigua y enseñaría al
    modelo a responder un plazo cualquiera (fallo real de v3 en eb-001).
    Se deriva del complemento «para ...» de la propia frase o, en su
    defecto, de la rúbrica del artículo; None si no hay forma fiable.
    """
    escaped = re.escape(hit.value)
    match = re.search(
        rf"(?:plazo(?: máximo)? de|en el término de)\s+{escaped},?\s+para\s+([^,.;]{{4,90}})",
        hit.sentence,
        re.IGNORECASE,
    )
    if match:
        trimmed = _trim_purpose(match.group(1))
        if trimmed:
            return f"para {trimmed}"
    match = re.search(
        rf"plazo\s+para\s+([^,.;]{{4,90}}?)\s+(?:será|es)\s+de\s+{escaped}",
        hit.sentence,
        re.IGNORECASE,
    )
    if match:
        trimmed = _trim_purpose(match.group(1))
        if trimmed:
            return f"para {trimmed}"
    if 0 <= hit.article < len(articles):
        heading = articles[hit.article].heading.rstrip(".").strip()
        if (
            heading
            and heading.lower() not in _GENERIC_HEADINGS
            and len(heading.split()) <= 6
            and not _PURPOSE_BLOCKLIST.search(heading)
        ):
            return f"en relación con {heading.lower()}"
    return None


def find_deadlines(text: str, limit: int = 5) -> list[str]:
    """Expresiones de plazo VALIDADAS («tres meses», «31 de diciembre de 2026»)."""
    deadlines: list[str] = []
    for pattern in _DEADLINE_PATTERNS:
        for match in pattern.finditer(text):
            extracted = extract_deadline_value(match.group(1))
            if extracted and extracted[0] not in deadlines:
                deadlines.append(extracted[0])
    return deadlines[:limit]


def _sentence_hits(
    articles: list[Article], markers: tuple[str, ...], limit: int, max_words: int = 45
) -> list[SentenceHit]:
    """Frases con marcador léxico, acotadas y con índice de artículo."""
    hits: list[SentenceHit] = []
    for index, article in enumerate(articles):
        for sentence in split_sentences(article.body):
            words = len(sentence.split())
            if words < 4 or words > max_words:
                continue
            if not _ends_sentence(sentence):
                continue
            if _CONNECTOR_START.match(sentence):
                continue  # anafórica: no se sostiene fuera de contexto
            if any(marker in sentence.lower() for marker in markers):
                hits.append(SentenceHit(sentence=sentence, article=index))
                if len(hits) >= limit:
                    return hits
    return hits


def find_obligation_hits(articles: list[Article], limit: int = 5) -> list[SentenceHit]:
    return _sentence_hits(articles, OBLIGATION_MARKERS, limit)


def find_sanction_hits(articles: list[Article], limit: int = 3) -> list[SentenceHit]:
    return _sentence_hits(articles, SANCTION_MARKERS, limit)


def find_amount_hits(articles: list[Article], limit: int = 2) -> list[SentenceHit]:
    """Frases sancionadoras CON cuantía monetaria (para multa_maxima)."""
    hits = []
    for hit in _sentence_hits(articles, SANCTION_MARKERS, limit * 3):
        if _AMOUNT_RE.search(hit.sentence):
            hits.append(hit)
            if len(hits) >= limit:
                break
    return hits


def _amount_value(amount: str) -> float:
    """«30.000 euros» -> 30000.0; «601,04 euros» -> 601.04; «2.000.000 de euros» ok."""
    number = amount.replace("euros", "").replace("€", "").strip()
    if number.endswith(" de") or number.endswith("\tde"):
        number = number[:-3].strip()
    number = number.replace(".", "").replace(",", ".")
    try:
        return float(number)
    except ValueError:
        return 0.0


def max_amount_in(text: str) -> str | None:
    """Cuantía máxima verbatim de un texto («multa de 3.000 a 7.500 euros»)."""
    amounts = _AMOUNT_RE.findall(text)
    if not amounts:
        return None
    return max(amounts, key=_amount_value).strip()


# Tramos del régimen sancionador («infracciones leves/graves/muy graves»).
# El orden importa: «muy graves» antes que «graves» para no ensombrecerlo.
_TIER_RE = re.compile(r"\binfracci(?:ón|ones)\s+(muy\s+graves?|graves?|leves?)\b", re.IGNORECASE)
_TIER_PLURAL = {"leve": "leves", "grave": "graves", "muy grave": "muy graves"}


class TierHit(NamedTuple):
    """Tramo sancionador: nombre, cuantía máxima del tramo y frase de origen."""

    tier: str  # "leves" | "graves" | "muy graves"
    amount: str
    article: int
    sentence: str


# Marcadores de lista dentro de una «frase» sancionadora: «a) Las leves, con
# multa de hasta X. b) Las graves...» — las letras minúsculas no abren frase
# para el separador, así que el escalado entero llega como una sola frase.
_TIER_LIST_MARKER = re.compile(r"(?:^|\s)[a-z]\)\s")


def _mentioned_tiers(text: str) -> set[str]:
    return {
        _TIER_PLURAL.get(" ".join(m.lower().split()), " ".join(m.lower().split()))
        for m in _TIER_RE.findall(text)
    }


def find_sanction_tiers(articles: list[Article]) -> list[TierHit]:
    """Tramos sancionadores con cuantía atribuible sin ambigüedad.

    v4 etiquetaba multa_maxima como el máximo de la ventana SIEMPRE, y el
    modelo aprendió a responder el tope de las «muy graves» aunque se le
    preguntara por las «graves» (fallo real en eb-128). Aquí cada tramo se
    ancla a SU cuantía.

    Escanea las frases de los artículos DIRECTAMENTE (no vía
    ``find_amount_hits``: su tope de 45 palabras descartaba los escalados
    reales de 49-800 palabras antes de poder trocearlos) y sub-divide SIEMPRE
    por marcadores de lista («a) ... b) ...», el patrón de los escalados
    BOE) antes de atribuir. Un segmento que mezcla tramos se descarta.
    """
    tiers: list[TierHit] = []
    seen: set[str] = set()
    for index, article in enumerate(articles):
        for sentence in split_sentences(article.body):
            if not any(marker in sentence.lower() for marker in SANCTION_MARKERS):
                continue
            if not _AMOUNT_RE.search(sentence):
                continue
            for segment in _TIER_LIST_MARKER.split(sentence):
                mentioned = _mentioned_tiers(segment)
                if len(mentioned) != 1:
                    continue  # sin tramo, o tramos mezclados: inatribuible
                tier = next(iter(mentioned))
                amount = max_amount_in(segment)
                if amount is None or tier in seen:
                    continue
                seen.add(tier)
                tiers.append(
                    TierHit(
                        tier=tier, amount=amount, article=index,
                        sentence=segment.strip(),
                    )
                )
    return tiers


def find_obligation_sentences(text: str, limit: int = 5) -> list[str]:
    """Frases que expresan obligaciones (compatibilidad: sobre texto plano)."""
    found = [
        sentence
        for sentence in split_sentences(text)
        if 4 <= len(sentence.split()) <= 45
        and any(marker in sentence.lower() for marker in OBLIGATION_MARKERS)
    ]
    return found[:limit]


def find_sanction_sentences(text: str, limit: int = 3) -> list[str]:
    """Frases que mencionan sanciones, multas o reintegros (compatibilidad)."""
    found = [
        sentence
        for sentence in split_sentences(text)
        if 4 <= len(sentence.split()) <= 45
        and any(marker in sentence.lower() for marker in SANCTION_MARKERS)
    ]
    return found[:limit]


def simplify_text(text: str) -> str:
    """Aplica el glosario de lenguaje llano (sustitución determinística)."""
    result = text
    for formal, plain in PLAIN_GLOSSARY:
        result = result.replace(formal, plain)
    return result


# ---------------------------------------------------------------------------
# Ventana estructural de contexto
# ---------------------------------------------------------------------------


def _fit_sentences(text: str, counter: TokenCounter, token_budget: int) -> str:
    """Prefijo del texto a nivel de frase que cabe en el presupuesto."""
    if counter.count(text) <= token_budget:
        return text
    kept: list[str] = []
    used = 0
    for sentence in split_sentences(text):
        cost = counter.count(sentence) + 1
        if used + cost > token_budget:
            break
        kept.append(sentence)
        used += cost
    return " ".join(kept)


def _article_block(article: Article) -> str:
    head = f"Artículo {article.number}." + (f" {article.heading}" if article.heading else "")
    return f"{head}\n{article.body}".strip()


def _window_header(doc: RawDocument) -> str:
    """Cabecera de la ventana: incluye materias y URL para poder anclarlos."""
    return (
        f"{doc.header()}\n"
        f"Materias: {', '.join(doc.materia)} · Texto oficial: {doc.url}"
    )


def compose_window(
    doc: RawDocument,
    articles: list[Article],
    needed: list[int],
    counter: TokenCounter,
    token_budget: int = DEFAULT_WINDOW_TOKENS,
) -> str:
    """Cabecera + artículos indicados (por ÍNDICE), dentro del presupuesto.

    ``needed`` contiene índices en ``articles`` (no números de artículo: los
    BOE que modifican otras normas repiten números). Los artículos se añaden
    ÍNTEGROS en el orden pedido mientras quepan; si ni el primero cabe, se
    recorta a nivel de frase. El llamador es responsable de verificar que su
    target aparece en la ventana devuelta (anclaje garantizado).
    """
    header = _window_header(doc)
    budget = token_budget - counter.count(header)
    valid = [i for i in dict.fromkeys(needed) if 0 <= i < len(articles)]
    blocks: list[str] = []
    for index in valid:
        block = _article_block(articles[index])
        cost = counter.count(block) + 2
        if cost <= budget:
            blocks.append(block)
            budget -= cost
    if not blocks:
        # Ni el primer artículo necesario cabe entero: recorte por frases.
        first = articles[valid[0]] if valid else (articles[0] if articles else None)
        if first is not None:
            head = f"Artículo {first.number}." + (
                f" {first.heading}" if first.heading else ""
            )
            body = _fit_sentences(
                first.body, counter, max(budget - counter.count(head) - 2, 0)
            )
            if body:
                blocks.append(f"{head}\n{body}")
    if not blocks:
        return header
    return header + "\n\n" + "\n\n".join(blocks)


def _first_sentence(text: str) -> str:
    """Primera frase CON CONTENIDO del texto.

    Los cuerpos de artículo suelen empezar por apartados numerados («1. El
    presente real decreto...»); el separador de frases parte «1.» como
    "frase" propia, así que se salta cualquier fragmento de menos de 4
    palabras hasta dar con una frase real.
    """
    stripped = re.sub(r"^\s*\d+\.\s+", "", text.strip())
    for sentence in split_sentences(stripped):
        if len(sentence.split()) >= 4:
            return sentence
    return stripped.strip()


def _valid_objeto(text: str) -> bool:
    """Un objeto usable: frase real y sin cortes colgantes («...redacción: «»)."""
    stripped = text.rstrip()
    return len(stripped.split()) >= 4 and not stripped.endswith(("«", ":", ";", ","))


# ---------------------------------------------------------------------------
# Pools de instrucciones (muestreados POR EJEMPLO, con y sin cita)
# ---------------------------------------------------------------------------

_RESUMEN_INSTRUCTIONS_ESTRUCTURADO = (
    "Elabora un resumen estructurado de la siguiente disposición con los "
    "campos: norma, rango, órgano emisor, publicación, materias, objeto, "
    "puntos clave por artículo, obligaciones y plazos. Cita la fuente.",
    "Resume la disposición siguiente en formato de ficha con estos campos: "
    "norma, rango, órgano emisor, publicación, materias, objeto, puntos "
    "clave, obligaciones y plazos. Incluye la cita de la fuente.",
    "Prepara una ficha-resumen de esta norma (norma, rango, órgano emisor, "
    "publicación, materias, objeto, puntos clave por artículo, obligaciones "
    "y plazos), citando la fuente al final.",
)

_RESUMEN_INSTRUCTIONS_PROSA = (
    # Espejo del estilo de instrucción del benchmark (sin cita).
    ("Resume en un párrafo la disposición, indicando objeto, obligaciones y "
     "plazos principales.", False),
    ("Resume en un párrafo la disposición, indicando objeto y plazos.", False),
    ("Resume en un párrafo la disposición, indicando obligaciones, plazos y "
     "sanciones si las hay.", False),
    ("Resume esta disposición en un solo párrafo: qué regula, a quién "
     "obliga y qué plazos marca.", False),
    # Variantes con cita (minoritarias: el estilo del producto).
    ("Escribe un resumen breve, en prosa y sin listas, de esta disposición "
     "indicando objeto, obligaciones y los plazos más relevantes. Termina "
     "citando la fuente.", True),
    ("¿De qué trata la siguiente disposición? Contesta en un único párrafo "
     "mencionando su objeto y sus plazos clave, y cita la fuente.", True),
)

# «El presente real decreto tiene por objeto ...» -> apertura en voz activa
# al estilo de las referencias del bench («El Real Decreto X tiene por
# objeto regular ...» en lugar de «X es una disposición de rango ...»).
_OBJETO_BOILERPLATE = re.compile(
    r"^(?:el|la|este|esta)\s+presente\s+[\wáéíóúñ \-]+?\s+tiene\s+por\s+objeto\s+",
    re.IGNORECASE,
)


def _resumen_opening(doc: RawDocument, objeto: str) -> str:
    """Primera frase del resumen en prosa, en voz activa y sin metadatos."""
    short = _short_ref(doc)
    short_cap = short[0].upper() + short[1:]
    stripped = _OBJETO_BOILERPLATE.sub("", objeto).strip()
    if stripped and stripped != objeto:
        return f"{short_cap} tiene por objeto {stripped}"
    if objeto:
        return f"{short_cap} regula lo siguiente: {objeto}"
    return (
        f"{short_cap} es una disposición publicada por "
        f"{_organ_with_article(doc.organo_emisor)}."
    )

_QA_INSTRUCTIONS = (
    "Según el texto de la disposición, ¿qué establece el artículo {numero} "
    "{referencia}? Cita la fuente entre corchetes.",
    "¿Qué dispone el artículo {numero} {referencia}? Responde citando la "
    "fuente entre corchetes.",
    "Explica el contenido del artículo {numero} {referencia} y cita la "
    "fuente entre corchetes al final.",
    "Resume qué regula el artículo {numero} {referencia}, con la cita de la "
    "fuente entre corchetes.",
    "¿Qué dice exactamente el artículo {numero} {referencia}? Apoya tu "
    "respuesta con la cita de la fuente.",
)

_LLANO_INSTRUCTIONS = (
    "Explica en lenguaje llano, para una persona sin formación jurídica, qué "
    "dice {referencia} y qué obligaciones y plazos le afectan.",
    "Explícame como si no supiera nada de derecho qué establece {referencia} "
    "y qué tengo que hacer o tener en cuenta.",
    "Traduce a lenguaje sencillo el contenido de {referencia}: qué regula, a "
    "quién obliga y con qué plazos.",
    "¿Me afecta {referencia}? Cuéntamelo en lenguaje claro: qué regula, qué "
    "obligaciones impone y qué fechas conviene apuntar.",
)

_EXTRACCION_PLAZO_VARIANTS = (
    ("Según la disposición siguiente, ¿qué plazo se establece? Responde en "
     "una sola frase y cita la fuente entre corchetes.", True),
    ("Indica de forma breve el plazo que fija esta disposición, citando la "
     "fuente entre corchetes.", True),
    ("¿De cuánto tiempo se dispone según esta norma? Contesta con una única "
     "frase que incluya la cita de la fuente.", True),
    ("¿Qué plazo establece esta disposición? Responde en una sola frase, "
     "sin añadir nada más.", False),
    ("Dime únicamente el plazo que fija la norma siguiente, en una frase.", False),
)

_PLAZO_STEMS = (
    "El plazo establecido es de {v}",
    "El plazo fijado es de {v}",
    "La norma establece un plazo de {v}",
)


def _disambiguated_plazo_variants(purpose: str) -> tuple[tuple[str, bool], ...]:
    """Variantes de pregunta que especifican QUÉ plazo se pide."""
    return (
        (f"Según la disposición siguiente, ¿qué plazo se establece {purpose}? "
         "Responde en una sola frase y cita la fuente entre corchetes.", True),
        (f"Indica brevemente el plazo fijado {purpose}, citando la fuente "
         "entre corchetes.", True),
        (f"¿Qué plazo fija esta norma {purpose}? Responde en una sola frase, "
         "sin añadir nada más.", False),
        (f"Extrae el plazo previsto {purpose}. Responde en una sola frase.", False),
        (f"¿De cuánto tiempo se dispone {purpose} según esta norma? Contesta "
         "con una única frase que incluya la cita de la fuente.", True),
    )


def _disambiguated_plazo_stems(purpose: str, kind: str) -> tuple[str, ...]:
    """Stems de respuesta que ECO el propósito preguntado.

    v4 respondía «El plazo establecido es de X» sin repetir el propósito: la
    loss nunca ligaba pregunta y valor, y el modelo podía aprender «responde
    un plazo cualquiera». Repetir el propósito en la respuesta hace que la
    elección correcta forme parte de la señal de entrenamiento.
    """
    if kind == "fecha":
        return (
            f"La fecha límite {purpose} es el {{v}}",
            f"El plazo {purpose} vence el {{v}}",
        )
    return (
        f"El plazo {purpose} es de {{v}}",
        f"El plazo establecido {purpose} es de {{v}}",
    )


def _tier_multa_variants(tier: str) -> tuple[tuple[str, bool], ...]:
    """Preguntas de multa que nombran el TRAMO cuyo tope se pide."""
    return (
        (f"¿Qué multa máxima se prevé para las infracciones {tier}? Responde "
         "de forma breve citando la fuente entre corchetes.", True),
        (f"Indica la multa máxima aplicable a las infracciones {tier}, "
         "citando la fuente entre corchetes.", True),
        (f"¿Con qué multa máxima se sancionan las infracciones {tier} según "
         "esta disposición? Contesta en una frase.", False),
    )


def _tier_multa_stems(tier: str) -> tuple[str, ...]:
    return (
        f"La multa máxima para las infracciones {tier} es de {{v}}",
        f"Las infracciones {tier} se sancionan con multa de hasta {{v}}",
    )


_MULTA_GLOBAL_VARIANTS = (
    ("¿Cuál es la multa más alta que puede llegar a imponerse según esta "
     "disposición, considerando todos los tipos de infracción? Responde de "
     "forma breve citando la fuente entre corchetes.", True),
    ("Indica el importe máximo absoluto de multa que contempla esta norma, "
     "sea cual sea el tipo de infracción, citando la fuente.", True),
)

_MULTA_GLOBAL_STEMS = (
    "La multa más alta que contempla la norma es de {v}",
    "El importe máximo absoluto de multa asciende a {v}",
)

_FECHA_STEMS = (
    "La fecha límite es el {v}",
    "El plazo vence el {v}",
)

_EXTRACCION_SANCION_VARIANTS = (
    ("¿Qué sanción o multa prevé la siguiente disposición? Responde de forma "
     "breve citando la fuente entre corchetes.", True),
    ("Indica brevemente el régimen sancionador que establece esta norma, con "
     "cita de la fuente entre corchetes.", True),
    ("¿Qué multa puede imponerse según esta disposición? Contesta en una "
     "frase.", False),
)

_EXTRACCION_OBLIGACION_VARIANTS = (
    ("¿Qué obligación principal impone la siguiente disposición y a quién? "
     "Responde de forma breve citando la fuente.", True),
    ("Señala en una o dos frases la obligación más relevante que establece "
     "esta norma, con la cita de la fuente.", True),
    ("Resume en una frase la obligación central de esta norma.", False),
)

_OBLIGACIONES_JSON_INSTRUCTION = (
    "Extrae las obligaciones, los plazos y las sanciones de la siguiente "
    'disposición. Responde únicamente con un objeto JSON con las claves '
    '"obligaciones", "plazos", "sanciones" y "fuente".'
)

# Variantes estilo benchmark: JSON con clave pedida y valor VERBATIM.
_EXTRACCION_JSON_INSTR = (
    "Extrae {desc} de la siguiente disposición. Responde únicamente con un "
    'objeto JSON con la clave "{campo}".',
    "Lee la disposición y devuelve solo un objeto JSON con la clave "
    '"{campo}" y el valor exacto tal como aparece en el texto.',
    "Devuelve un objeto JSON, sin texto adicional, con la clave "
    '"{campo}" cuyo valor sea {desc} según esta norma.',
)


# ---------------------------------------------------------------------------
# Generadores offline (determinísticos, sin LLM)
# ---------------------------------------------------------------------------


class SynthContext(NamedTuple):
    """Parámetros compartidos de la síntesis offline."""

    counter: TokenCounter
    token_budget: int
    seed: int


def _doc_input(doc: RawDocument) -> str:
    """Bloque de documento completo (solo para compatibilidad/tests)."""
    return f"{doc.header()}\n\n{doc.texto}"


def offline_resumen(
    doc: RawDocument, articles: list[Article], ctx: SynthContext
) -> list[SftExample]:
    """Resumen: prosa de un párrafo (mayoría) o ficha estructurada."""
    rng = _example_rng(ctx.seed, doc.id, "resumen")
    obligation_hits = find_obligation_hits(articles, limit=3)
    deadline_hits = find_deadline_hits(articles, limit=4)
    sanction_hits = find_sanction_hits(articles, limit=1)
    needed = sorted(
        {0}
        | {h.article for h in obligation_hits}
        | {h.article for h in deadline_hits}
        | {h.article for h in sanction_hits}
    )
    window = compose_window(doc, articles, needed, ctx.counter, ctx.token_budget)
    # Anclaje: solo se citan obligaciones/plazos/sanciones de la ventana.
    obligations = [h.sentence for h in obligation_hits if h.sentence in window]
    deadlines = [h.value for h in deadline_hits if h.value in window]
    sanctions = [h.sentence for h in sanction_hits if h.sentence in window]
    objeto = _first_sentence(articles[0].body) if articles else _first_sentence(doc.texto)
    if not _valid_objeto(objeto) or objeto not in window:
        objeto = ""

    use_prosa = rng.random() < 0.65
    if use_prosa:
        instruction, cite = _RESUMEN_INSTRUCTIONS_PROSA[
            rng.randrange(len(_RESUMEN_INSTRUCTIONS_PROSA))
        ]
        # Estilo de las referencias del bench: párrafo DENSO en voz activa
        # (apertura con la norma como sujeto, obligaciones verbatim como
        # frases propias, sanción y plazos entrelazados), sin metadatos.
        # v6: las predicciones de v4/v5 quedaban un ~20% más cortas que las
        # referencias y ROUGE-L penalizaba el recall — el target incluye
        # ahora hasta 3 obligaciones y la frase sancionadora si existe.
        opening = _resumen_opening(doc, objeto)
        sentences = [opening if _ends_sentence(opening) else f"{opening}."]
        for obligation in obligations[:3]:
            sentences.append(obligation if _ends_sentence(obligation) else f"{obligation}.")
        if sanctions and sanctions[0] not in sentences:
            sanction = sanctions[0]
            sentences.append(sanction if _ends_sentence(sanction) else f"{sanction}.")
        if deadlines:
            sentences.append(f"Los plazos principales son {'; '.join(deadlines)}.")
        if cite:
            sentences.append(f"Fuente: [{doc.id}].")
        response = " ".join(sentences)
    else:
        instruction = _RESUMEN_INSTRUCTIONS_ESTRUCTURADO[
            rng.randrange(len(_RESUMEN_INSTRUCTIONS_ESTRUCTURADO))
        ]
        included_articles = [a for a in articles if _article_block(a) in window]
        key_points = []
        for a in included_articles[1:5]:
            point = _first_sentence(a.body)
            if not _valid_objeto(point):
                continue  # apartados numerales pelados y cortes colgantes fuera
            label = a.heading.rstrip(".") if a.heading else f"Artículo {a.number}"
            key_points.append(f"- Artículo {a.number} ({label}): {point}")
        lines = [
            f"### Resumen estructurado — {doc.short_title()}",
            "",
            f"- **Norma**: {doc.titulo}",
            f"- **Rango**: {doc.rango}",
            f"- **Órgano emisor**: {doc.organo_emisor}",
            f"- **Publicación**: {doc.fecha_publicacion}",
            f"- **Materias**: {', '.join(doc.materia)}",
        ]
        if objeto:
            lines.append(f"- **Objeto**: {objeto}")
        if key_points:
            lines += ["", "**Puntos clave por artículo:**", *key_points]
        if obligations:
            lines += ["", "**Obligaciones destacadas:**"]
            lines += [f"- {sentence}" for sentence in obligations]
        if deadlines:
            lines += ["", f"**Plazos relevantes:** {'; '.join(deadlines)}."]
        lines += ["", f"Fuente: [{doc.id}] {doc.url}"]
        response = "\n".join(lines)

    if not _ends_sentence(response.splitlines()[-1]) and not response.rstrip().endswith(doc.url):
        return []
    return [
        SftExample(
            id=f"{doc.id}-resumen",
            doc_id=doc.id,
            task="resumen",
            instruction=instruction,
            input=window,
            response=response,
            source="offline",
        )
    ]


def offline_qa(
    doc: RawDocument, articles: list[Article], ctx: SynthContext, max_questions: int = 3
) -> list[SftExample]:
    """Q&A por artículo con cita en el formato del portfolio."""
    examples: list[SftExample] = []
    candidates = [(i, a) for i, a in enumerate(articles) if a.body][:max_questions]
    for index, article in candidates:
        rng = _example_rng(ctx.seed, doc.id, "qa", str(article.number))
        window = compose_window(doc, articles, [index], ctx.counter, ctx.token_budget)
        sentences = split_sentences(article.body)
        quoted = " ".join(sentences[:2])
        if quoted not in window:  # anclaje: la cita debe estar en la ventana
            quoted = _first_sentence(article.body)
            if quoted not in window:
                continue
        heading = f" ({article.heading.rstrip('.')})" if article.heading else ""
        reference = _short_ref(doc, preceded_by_de=True)
        response = (
            f"El artículo {article.number}{heading} {reference} establece lo "
            f"siguiente: «{quoted}» [{doc.id}, art. {article.number}]."
        )
        instruction = _QA_INSTRUCTIONS[rng.randrange(len(_QA_INSTRUCTIONS))].format(
            numero=article.number, referencia=reference
        )
        examples.append(
            SftExample(
                id=f"{doc.id}-qa-{article.number}",
                doc_id=doc.id,
                task="qa",
                instruction=instruction,
                input=window,
                response=response,
                source="offline",
            )
        )
    return examples


def offline_obligaciones(
    doc: RawDocument, articles: list[Article], ctx: SynthContext
) -> list[SftExample]:
    """Extracción de obligaciones/plazos/sanciones con formato variable."""
    rng = _example_rng(ctx.seed, doc.id, "obligaciones")
    obligation_hits = find_obligation_hits(articles, limit=5)
    deadline_hits = find_deadline_hits(articles, limit=5)
    sanction_hits = find_sanction_hits(articles, limit=3)
    needed = sorted(
        {h.article for h in obligation_hits}
        | {h.article for h in deadline_hits}
        | {h.article for h in sanction_hits}
    )
    if not needed:
        return []  # sin señal estructural no hay ejemplo (nunca «no se detectan»)
    window = compose_window(doc, articles, needed, ctx.counter, ctx.token_budget)
    obligations = [h.sentence for h in obligation_hits if h.sentence in window]
    deadlines = [h.value for h in deadline_hits if h.value in window]
    sanctions = [h.sentence for h in sanction_hits if h.sentence in window]
    if not (obligations or deadlines or sanctions):
        return []

    fmt = rng.randrange(3)
    if fmt == 0:  # JSON
        instruction = _OBLIGACIONES_JSON_INSTRUCTION
        payload = {
            "obligaciones": obligations,
            "plazos": deadlines,
            "sanciones": sanctions,
            "fuente": doc.id,
        }
        response = json.dumps(payload, ensure_ascii=False, indent=2)
    elif fmt == 1:  # Lista en texto plano (solo secciones con contenido)
        instruction = (
            "Enumera las obligaciones, los plazos y las sanciones que establece "
            "la siguiente disposición, en apartados con viñetas. Si algún "
            "apartado no aplica, omítelo. Cita la fuente al final."
        )
        lines: list[str] = []
        if obligations:
            lines += ["Obligaciones:"] + [f"- {s}" for s in obligations]
        if deadlines:
            lines += ["", "Plazos:"] + [f"- {d}" for d in deadlines]
        if sanctions:
            lines += ["", "Sanciones:"] + [f"- {s}" for s in sanctions]
        lines += ["", f"Fuente: [{doc.id}]"]
        response = "\n".join(line for line in lines).strip()
    else:  # Prosa breve
        instruction = (
            "Describe brevemente, en prosa, qué obligaciones impone esta "
            "disposición, con qué plazos y qué sanciones prevé. Cita la fuente."
        )
        parts = []
        if obligations:
            parts.append(f"La norma obliga, entre otras cosas, a lo siguiente: {obligations[0]}")
        if deadlines:
            parts.append(f"Fija plazos de {', '.join(deadlines[:3])}.")
        if sanctions:
            parts.append(f"En materia sancionadora: {sanctions[0]}")
        parts.append(f"[{doc.id}]")
        response = " ".join(parts)

    return [
        SftExample(
            id=f"{doc.id}-obligaciones",
            doc_id=doc.id,
            task="obligaciones",
            instruction=instruction,
            input=window,
            response=response,
            source="offline",
        )
    ]


def offline_llano(
    doc: RawDocument, articles: list[Article], ctx: SynthContext
) -> list[SftExample]:
    """Explicación en lenguaje llano, en ficha o en prosa corrida (50/50)."""
    rng = _example_rng(ctx.seed, doc.id, "llano")
    obligation_hits = find_obligation_hits(articles, limit=3)
    deadline_hits = find_deadline_hits(articles, limit=3)
    needed = sorted(
        {0} | {h.article for h in obligation_hits} | {h.article for h in deadline_hits}
    )
    window = compose_window(doc, articles, needed, ctx.counter, ctx.token_budget)
    obligations = [simplify_text(h.sentence) for h in obligation_hits if h.sentence in window]
    deadlines = [h.value for h in deadline_hits if h.value in window]
    objeto = _first_sentence(articles[0].body) if articles else ""
    if objeto and (not _valid_objeto(objeto) or objeto not in window):
        objeto = ""

    rango_phrase = _RANGO_ARTICLE.get(
        doc.rango.lower(), f"una norma de rango {doc.rango.lower()}"
    )
    instruction = _LLANO_INSTRUCTIONS[rng.randrange(len(_LLANO_INSTRUCTIONS))].format(
        referencia=_short_ref(doc)
    )

    if rng.random() < 0.5:  # ficha con encabezados
        lines = [
            f"**¿Qué es?** {doc.short_title()} es {rango_phrase} que publicó "
            f"{_organ_with_article(doc.organo_emisor)} el "
            f"{format_date_es(doc.fecha_publicacion)}.",
        ]
        if objeto:
            lines += ["", f"**¿De qué trata?** {simplify_text(objeto)}"]
        if obligations:
            lines += ["", "**¿Qué obligaciones impone?**"]
            lines += [f"- {sentence}" for sentence in obligations]
        if deadlines:
            lines += ["", "**Plazos que conviene apuntar:**"]
            lines += [f"- {deadline}" for deadline in deadlines]
        lines += [
            "",
            "**En resumen:** revisa si estás dentro del ámbito de aplicación y "
            f"anota los plazos. El texto oficial está en {doc.url} [{doc.id}].",
        ]
        response = "\n".join(lines)
    else:  # prosa corrida sin encabezados
        sentences = [
            f"{doc.short_title()} es {rango_phrase} que publicó "
            f"{_organ_with_article(doc.organo_emisor)} el "
            f"{format_date_es(doc.fecha_publicacion)}.",
        ]
        if objeto:
            sentences.append(f"En pocas palabras, {simplify_text(objeto).rstrip('.')}." )
        if obligations:
            sentences.append(
                f"Lo más importante que exige: {obligations[0].rstrip('.').lower()}."
            )
        if deadlines:
            sentences.append(
                f"Apunta estas fechas o plazos: {'; '.join(deadlines)}."
            )
        sentences.append(f"El texto oficial está en {doc.url} [{doc.id}].")
        response = " ".join(sentences)

    return [
        SftExample(
            id=f"{doc.id}-llano",
            doc_id=doc.id,
            task="llano",
            instruction=instruction,
            input=window,
            response=response,
            source="offline",
        )
    ]


def _extraccion_example(
    doc: RawDocument,
    ctx: SynthContext,
    *,
    suffix: str,
    window: str,
    value: str,
    json_field: str,
    json_desc: str,
    prose_variants: tuple[tuple[str, bool], ...],
    prose_stems: tuple[str, ...],
    require_desc: bool = False,
) -> SftExample | None:
    """Ejemplo de extracción (frase corta o JSON verbatim) con anclaje.

    ``require_desc``: con pregunta desambiguada, solo valen las plantillas
    JSON que transportan la descripción (una pregunta genérica con etiqueta
    desambiguada re-crearía la ambigüedad que se quiere eliminar).
    """
    rng = _example_rng(ctx.seed, doc.id, "extraccion", suffix)
    if value not in window:
        return None  # anclaje garantizado: el dato debe estar en la ventana
    if rng.random() < 0.5:  # variante JSON estilo benchmark
        json_pool = (
            tuple(t for t in _EXTRACCION_JSON_INSTR if "{desc}" in t)
            if require_desc
            else _EXTRACCION_JSON_INSTR
        )
        instruction = json_pool[rng.randrange(len(json_pool))].format(
            campo=json_field, desc=json_desc
        )
        response = json.dumps({json_field: value}, ensure_ascii=False)
    else:
        instruction, cite = prose_variants[rng.randrange(len(prose_variants))]
        stem = prose_stems[rng.randrange(len(prose_stems))].format(v=value)
        response = f"{stem} [{doc.id}]." if cite else f"{stem}."
    return SftExample(
        id=f"{doc.id}-extraccion-{suffix}",
        doc_id=doc.id,
        task="extraccion",
        instruction=instruction,
        input=window,
        response=response,
        source="offline",
    )


def _future_date(value: str, doc: RawDocument) -> bool:
    """Una fecha límite creíble no es anterior a la publicación de la norma.

    Filtra fechas de referencia/elegibilidad («antes del 31 de diciembre de
    2024» en una norma de 2026) que no son el plazo que la norma establece.
    Sin año explícito no hay forma de saberlo: se descarta.
    """
    year_match = re.search(r"\b(\d{4})\b", value)
    if not year_match:
        return False
    return int(year_match.group(1)) >= int(doc.fecha_publicacion[:4])


def offline_extraccion(
    doc: RawDocument, articles: list[Article], ctx: SynthContext
) -> list[SftExample]:
    """Datos puntuales con respuesta corta: SOLO el dato pedido, verbatim.

    Es la tarea que ejercita el domain_bench (extract): plazos (duración y
    fecha futura), multa máxima con cuantía y obligación principal, en
    variantes de frase única (con/sin cita) y de JSON con la clave pedida.
    """
    examples: list[SftExample] = []
    all_deadline_hits = find_deadline_hits(articles, limit=6)
    for i, hit in enumerate(all_deadline_hits[:3]):
        if hit.kind == "fecha" and not _future_date(hit.value, doc):
            continue  # fecha de referencia pre-publicación: no es un plazo
        # Con varios plazos en el documento, la ventana INCLUYE los artículos
        # de los plazos hermanos (el artículo del plazo pedido va primero y
        # sobrevive seguro al presupuesto): el modelo debe aprender a elegir
        # el plazo preguntado con el distractor delante (fallo real de v3 en
        # eb-001: respondió el plazo de justificación cuando se preguntaba
        # por el de solicitud).
        needed = [hit.article] + [
            h.article for h in all_deadline_hits if h.article != hit.article
        ]
        window = compose_window(doc, articles, needed, ctx.counter, ctx.token_budget)
        visible = [h for h in all_deadline_hits if h.value in window]
        purpose = None
        if len(visible) > 1:
            purpose = deadline_purpose(hit, articles)
            if purpose is None:
                continue  # varios plazos visibles y sin propósito: no se emite
            # El propósito debe DISCRIMINAR: si otro plazo visible deriva el
            # mismo propósito (p. ej. dos plazos del mismo artículo que caen
            # al fallback de rúbrica), la pareja instrucción->etiqueta sería
            # contradictoria entre ejemplos. No se emite.
            sibling_purposes = {
                deadline_purpose(h, articles)
                for h in visible
                if h.value != hit.value
            }
            if purpose in sibling_purposes:
                continue
        # v6: un stem VERBATIM — la frase del documento que contiene el plazo
        # («Las solicitudes se presentarán en el plazo de veinte días
        # hábiles») — enseña a responder con la fórmula normativa exacta.
        # v5 parafraseaba («en un plazo de...») y el containment castigaba
        # hasta el cambio de un artículo gramatical (fallo real en eb-142).
        verbatim_stem = None
        if _ends_sentence(hit.sentence) and len(hit.sentence.split()) <= 45:
            verbatim_stem = (
                hit.sentence.rstrip(".").replace("{", "{{").replace("}", "}}")
            )
        if hit.kind == "duracion":
            json_desc = f"el plazo {purpose}" if purpose else "el plazo"
            prose_variants = (
                _disambiguated_plazo_variants(purpose) if purpose
                else _EXTRACCION_PLAZO_VARIANTS
            )
            stems = _disambiguated_plazo_stems(purpose, "duracion") if purpose else _PLAZO_STEMS
            if verbatim_stem is not None:
                stems = stems + (verbatim_stem,)
            example = _extraccion_example(
                doc, ctx,
                suffix=f"plazo-{i}", window=window, value=hit.value,
                json_field="plazo", json_desc=json_desc,
                prose_variants=prose_variants, prose_stems=stems,
                require_desc=purpose is not None,
            )
        else:
            json_desc = f"la fecha límite {purpose}" if purpose else "la fecha límite"
            prose_variants = (
                _disambiguated_plazo_variants(purpose) if purpose
                else _EXTRACCION_PLAZO_VARIANTS
            )
            stems = _disambiguated_plazo_stems(purpose, "fecha") if purpose else _FECHA_STEMS
            if verbatim_stem is not None:
                stems = stems + (verbatim_stem,)
            example = _extraccion_example(
                doc, ctx,
                suffix=f"fecha-{i}", window=window, value=hit.value,
                json_field="fecha_limite", json_desc=json_desc,
                prose_variants=prose_variants, prose_stems=stems,
                require_desc=purpose is not None,
            )
        if example is not None:
            examples.append(example)

    # Multa máxima. Con régimen de tramos (leves/graves/muy graves) las
    # preguntas nombran el TRAMO y la etiqueta es el tope de ESE tramo, con
    # los demás tramos visibles como distractores (v4 etiquetaba siempre el
    # máximo de ventana y aprendió a responder el tope de las «muy graves»
    # aunque se preguntara por las «graves»: fallo real en eb-128). Se añade
    # además una pregunta global con fraseo explícitamente absoluto.
    tiers = find_sanction_tiers(articles)
    tier_examples_emitted = False
    if len(tiers) >= 2:
        needed = sorted({t.article for t in tiers})
        window = compose_window(doc, articles, needed, ctx.counter, ctx.token_budget)
        visible_tiers = [
            t for t in tiers if t.sentence in window and t.amount in window
        ]
        if len(visible_tiers) >= 2:
            for t in visible_tiers:
                slug = t.tier.replace(" ", "-")
                example = _extraccion_example(
                    doc, ctx,
                    suffix=f"multa-{slug}", window=window, value=t.amount,
                    json_field="multa_maxima",
                    json_desc=f"la multa máxima para las infracciones {t.tier}",
                    prose_variants=_tier_multa_variants(t.tier),
                    prose_stems=_tier_multa_stems(t.tier),
                    require_desc=True,
                )
                if example is not None:
                    examples.append(example)
                    tier_examples_emitted = True
            global_amount = max(
                (t.amount for t in visible_tiers), key=_amount_value
            )
            # Guard: si la ventana muestra CUALQUIER importe mayor que el
            # candidato (un tramo inatribuible, un rango cuyo tope no
            # capturamos), el ejemplo «máximo absoluto» mentiría a la vista
            # del propio texto. No se emite (2/15 globales del primer build
            # contradecían su ventana: hallazgo de la revisión).
            window_max = max_amount_in(window)
            if window_max is None or _amount_value(window_max) > _amount_value(global_amount):
                pass
            else:
                example = _extraccion_example(
                    doc, ctx,
                    suffix="multa-global", window=window, value=global_amount,
                    json_field="multa_maxima",
                    json_desc="la multa más alta que puede imponerse, "
                              "considerando todos los tipos de infracción",
                    prose_variants=_MULTA_GLOBAL_VARIANTS,
                    prose_stems=_MULTA_GLOBAL_STEMS,
                    require_desc=True,
                )
                if example is not None:
                    examples.append(example)

    # Sin tramos discriminables: máximo real de la ventana con pregunta
    # genérica (único importe o régimen no tramificado).
    if not tier_examples_emitted:
        amount_hits = find_amount_hits(articles, limit=6)
        if amount_hits:
            needed = sorted({h.article for h in amount_hits})
            window = compose_window(doc, articles, needed, ctx.counter, ctx.token_budget)
            visible = [h.sentence for h in amount_hits if h.sentence in window]
            amount = max_amount_in(" ".join(visible)) if visible else None
            # Guard de consistencia: la etiqueta solo vale si es el importe
            # más alto VISIBLE en la ventana. Los escalados largos que las
            # frases sancionadoras cortas no capturan (>45 palabras) siguen
            # en la ventana y desmentirían la etiqueta (4 casos del primer
            # build enseñaban un sub-máximo: hallazgo de la revisión).
            if amount is not None:
                window_max = max_amount_in(window)
                if window_max is not None and _amount_value(window_max) > _amount_value(amount):
                    amount = None
            if amount is not None:
                example = _extraccion_example(
                    doc, ctx,
                    suffix="multa-0", window=window, value=amount,
                    json_field="multa_maxima", json_desc="la multa máxima",
                    prose_variants=_EXTRACCION_SANCION_VARIANTS,
                    prose_stems=("La multa máxima prevista es de {v}",
                                 "La sanción puede alcanzar los {v}"),
                )
                if example is not None:
                    examples.append(example)

    for i, hit in enumerate(find_obligation_hits(articles, limit=2)):
        if _EMBEDDED_LIST_MARKER.search(hit.sentence):
            continue  # «a) ... b) ...» embebido: no es una respuesta autónoma
        rng = _example_rng(ctx.seed, doc.id, "extraccion", f"oblig-{i}")
        window = compose_window(doc, articles, [hit.article], ctx.counter, ctx.token_budget)
        if hit.sentence not in window:
            continue
        instruction, cite = _EXTRACCION_OBLIGACION_VARIANTS[
            rng.randrange(len(_EXTRACCION_OBLIGACION_VARIANTS))
        ]
        body = hit.sentence if _ends_sentence(hit.sentence) else f"{hit.sentence}."
        response = f"{body.rstrip('.')} [{doc.id}]." if cite else body
        examples.append(
            SftExample(
                id=f"{doc.id}-extraccion-obligacion-{i}",
                doc_id=doc.id,
                task="extraccion",
                instruction=instruction,
                input=window,
                response=response,
                source="offline",
            )
        )
        break  # una obligación por documento es suficiente

    # Órgano emisor (siempre en la cabecera: anclaje trivial).
    rng = _example_rng(ctx.seed, doc.id, "extraccion", "organo")
    if rng.random() < 0.35:
        window = compose_window(doc, articles, [0], ctx.counter, ctx.token_budget)
        instruction = _EXTRACCION_JSON_INSTR[
            rng.randrange(len(_EXTRACCION_JSON_INSTR))
        ].format(campo="organo", desc="el órgano emisor")
        examples.append(
            SftExample(
                id=f"{doc.id}-extraccion-organo",
                doc_id=doc.id,
                task="extraccion",
                instruction=instruction,
                input=window,
                response=json.dumps({"organo": doc.organo_emisor}, ensure_ascii=False),
                source="offline",
            )
        )
    return examples


_OFFLINE_GENERATORS = {
    "resumen": offline_resumen,
    "qa": offline_qa,
    "obligaciones": offline_obligaciones,
    "llano": offline_llano,
    "extraccion": offline_extraccion,
}


def rebalance(
    examples: list[SftExample],
    weights: dict[str, float],
    seed: int = 42,
) -> list[SftExample]:
    """Re-muestrea por tarea hacia los pesos objetivo (determinista).

    Escala el total al máximo alcanzable sin sobre-representar ninguna tarea:
    ``N* = min_t(count_t / w_t)``; las tareas excedentarias se muestrean.
    """
    by_task: dict[str, list[SftExample]] = {}
    for example in examples:
        by_task.setdefault(example.task, []).append(example)
    applicable = {t: w for t, w in weights.items() if t in by_task and w > 0}
    if not applicable:
        return examples
    n_star = min(len(by_task[t]) / w for t, w in applicable.items())
    rng = random.Random(seed)
    kept: list[SftExample] = []
    for task, task_examples in sorted(by_task.items()):
        weight = applicable.get(task)
        if weight is None:
            kept.extend(task_examples)  # tarea fuera de los pesos: intacta
            continue
        target = min(len(task_examples), max(1, round(weight * n_star)))
        if target >= len(task_examples):
            kept.extend(task_examples)
        else:
            indices = sorted(rng.sample(range(len(task_examples)), target))
            kept.extend(task_examples[i] for i in indices)
    return kept


def build_sft_offline(
    docs: list[RawDocument],
    tasks: tuple[str, ...] = SFT_TASKS,
    seed: int = 42,
    max_per_task: int | None = None,
    counter: TokenCounter | None = None,
    token_budget: int = DEFAULT_WINDOW_TOKENS,
    balance: dict[str, float] | None = None,
) -> list[SftExample]:
    """Genera ejemplos SFT offline con ventana estructural y anclaje."""
    counter = counter or TokenCounter()
    ctx = SynthContext(counter=counter, token_budget=token_budget, seed=seed)
    rng = random.Random(seed)
    examples: list[SftExample] = []
    split_cache: dict[str, list[Article]] = {}
    for task in tasks:
        if task not in _OFFLINE_GENERATORS:
            raise ValueError(f"Tarea desconocida: {task!r} (disponibles: {SFT_TASKS})")
        generated: list[SftExample] = []
        for doc in docs:
            if doc.id not in split_cache:
                _, split_cache[doc.id] = split_articles(doc.texto)
            generated.extend(_OFFLINE_GENERATORS[task](doc, split_cache[doc.id], ctx))
        if max_per_task is not None and len(generated) > max_per_task:
            indices = sorted(rng.sample(range(len(generated)), max_per_task))
            generated = [generated[i] for i in indices]
        examples.extend(generated)
    if balance:
        examples = rebalance(examples, balance, seed=seed)
    return examples


# ---------------------------------------------------------------------------
# Pares de preferencia (DPO) offline
# ---------------------------------------------------------------------------


def _degrade_vague(doc: RawDocument) -> str:
    return (
        "La disposición analizada regula diversas cuestiones de interés general. "
        "En términos generales, establece una serie de obligaciones y plazos para "
        "los sujetos afectados, así como un posible régimen sancionador. Para más "
        "detalle se recomienda consultar el texto completo publicado en el boletín "
        "oficial correspondiente."
    )


def _degrade_copy(doc: RawDocument) -> str:
    words = doc.texto.split()
    return " ".join(words[:60]) + " (...)"


def _degrade_unstructured(doc: RawDocument) -> str:
    obligations = find_obligation_sentences(doc.texto, limit=2)
    base = " ".join(obligations) if obligations else _first_sentence(doc.texto)
    return (
        f"Básicamente la norma dice que {base.lower()} Y también hay plazos y "
        "puede haber multas si no se cumple, aunque depende de cada caso."
    )


_DEGRADERS = (_degrade_vague, _degrade_copy, _degrade_unstructured)


def build_dpo_offline(
    docs: list[RawDocument],
    seed: int = 42,
    n_pairs: int = 15,
    counter: TokenCounter | None = None,
    token_budget: int = DEFAULT_WINDOW_TOKENS,
) -> list[DpoPair]:
    """Construye pares chosen/rejected degradando la respuesta buena."""
    counter = counter or TokenCounter()
    ctx = SynthContext(counter=counter, token_budget=token_budget, seed=seed)
    rng = random.Random(seed)
    candidates: list[SftExample] = []
    for doc in docs:
        _, articles = split_articles(doc.texto)
        candidates.extend(offline_resumen(doc, articles, ctx))
        candidates.extend(offline_obligaciones(doc, articles, ctx))
    rng.shuffle(candidates)
    pairs: list[DpoPair] = []
    docs_by_id = {doc.id: doc for doc in docs}
    for index, example in enumerate(candidates[:n_pairs]):
        degrader = _DEGRADERS[index % len(_DEGRADERS)]
        pairs.append(
            DpoPair(
                id=f"{example.id}-dpo",
                doc_id=example.doc_id,
                task=example.task,
                instruction=example.instruction,
                input=example.input,
                chosen=example.response,
                rejected=degrader(docs_by_id[example.doc_id]),
            )
        )
    return pairs


# ---------------------------------------------------------------------------
# Síntesis online (teacher vía litellm / endpoint OpenAI-compatible)
# ---------------------------------------------------------------------------

# Tolerante a negritas markdown: «**PREGUNTA:** ...» también cumple contrato.
_TEACHER_QA_BLOCKS = re.compile(
    r"\*{0,2}PREGUNTA\*{0,2}\s*:\s*(.+?)\s*\*{0,2}RESPUESTA\*{0,2}\s*:\s*(.+)",
    re.DOTALL,
)
_TEACHER_BOILERPLATE = re.compile(r"\b(PREGUNTA|RESPUESTA)\s*:")

# Instrucción almacenada para cada tarea online: debe corresponder al formato
# que la plantilla del teacher EXIGE (ficha para resumen, JSON para
# obligaciones). Emparejar una instrucción de prosa con una respuesta en
# ficha enseñaría al modelo a ignorar el formato pedido.
_ONLINE_INSTRUCTION_POOLS: dict[str, tuple[str, ...]] = {
    "resumen": _RESUMEN_INSTRUCTIONS_ESTRUCTURADO,
    "obligaciones": (_OBLIGACIONES_JSON_INSTRUCTION,),
}


def _online_instruction(task: str, doc: RawDocument, rng: random.Random) -> str | None:
    """Instrucción coherente con el contrato de la plantilla, o None."""
    if task == "llano":
        template = _LLANO_INSTRUCTIONS[rng.randrange(len(_LLANO_INSTRUCTIONS))]
        return template.format(referencia=_short_ref(doc))
    pool = _ONLINE_INSTRUCTION_POOLS.get(task)
    if pool is None:
        return None
    return pool[rng.randrange(len(pool))]


def parse_teacher_qa(raw: str) -> tuple[str, str] | None:
    """'PREGUNTA: ... RESPUESTA: ...' -> (pregunta, respuesta) o None."""
    match = _TEACHER_QA_BLOCKS.search(raw)
    if not match:
        return None
    question = " ".join(match.group(1).split()).strip("* ")
    answer = match.group(2).strip().strip("*").strip()
    if not question or not answer or _TEACHER_BOILERPLATE.search(answer):
        return None
    return question, answer


def render_prompt(template_text: str, doc: RawDocument, texto: str | None = None) -> str:
    """Sustituye placeholders ``{{campo}}`` del template con datos del doc.

    ``texto`` permite pasar la ventana estructural en lugar del documento
    completo (obligatorio para que el ejemplo sobreviva al presupuesto).
    """
    mapping = {
        "id": doc.id,
        "rango": doc.rango,
        "titulo": doc.titulo,
        "titulo_corto": doc.short_title(),
        "organo_emisor": doc.organo_emisor,
        "fecha_publicacion": doc.fecha_publicacion,
        "materias": ", ".join(doc.materia),
        "url": doc.url,
        "texto": texto if texto is not None else doc.texto,
    }

    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in mapping:
            missing.append(key)
            return match.group(0)
        return mapping[key]

    rendered = _PLACEHOLDER.sub(_replace, template_text)
    if missing:
        raise ValueError(f"Placeholders desconocidos en la plantilla: {sorted(set(missing))}")
    return rendered


def load_prompt_template(prompts_dir: str | Path, task: str) -> str:
    """Carga la plantilla de una tarea desde ``prompts/<task>.md``."""
    path = Path(prompts_dir) / f"{task}.md"
    if not path.exists():
        raise FileNotFoundError(f"No existe la plantilla {path} para la tarea {task!r}")
    return path.read_text(encoding="utf-8")


def synthesize_online(
    docs: list[RawDocument],
    tasks: tuple[str, ...],
    *,
    model: str,
    backend: str,
    api_base: str | None,
    api_key: str | None,
    prompts_dir: str | Path,
    counter: TokenCounter | None = None,
    token_budget: int = DEFAULT_WINDOW_TOKENS,
    seed: int = 42,
) -> list[SftExample]:
    """Genera ejemplos preguntando al teacher sobre la MISMA ventana anclada.

    El teacher ve la ventana estructural (no el documento completo): así el
    ejemplo resultante respeta el presupuesto de tokens y el input guardado
    coincide exactamente con lo que el teacher leyó. Las salidas de la tarea
    ``qa`` se parsean estructuralmente (PREGUNTA:/RESPUESTA:) y las que no
    cumplen el contrato se descartan (contadas en el log).
    """
    from adapterforge.llm import LlmError, chat_complete  # perezoso

    counter = counter or TokenCounter()
    examples: list[SftExample] = []
    rejected = 0
    errors = 0
    for task in tasks:
        template = load_prompt_template(prompts_dir, task)
        for index, doc in enumerate(docs, start=1):
            _, articles = split_articles(doc.texto)
            needed = [a.number for a in articles]
            window = compose_window(doc, articles, needed, counter, token_budget)
            prompt = render_prompt(template, doc, texto=window)
            try:
                response = chat_complete(
                    [{"role": "user", "content": prompt}],
                    model,
                    backend=backend,
                    api_base=api_base,
                    api_key=api_key,
                    temperature=0.4,
                    max_tokens=1200,
                )
            except LlmError as exc:
                # Un fallo transitorio no debe tirar el lote entero.
                errors += 1
                print(f"[synthesize] error del teacher en {doc.id} ({exc}); se omite")
                continue
            if index % 25 == 0:
                print(f"[synthesize] {task}: {index}/{len(docs)} documentos", flush=True)
            if task == "qa":
                parsed = parse_teacher_qa(response)
                if parsed is None:
                    rejected += 1
                    continue
                instruction, response = parsed
            else:
                if _TEACHER_BOILERPLATE.search(response):
                    rejected += 1
                    continue
                rng = _example_rng(seed, doc.id, "online", task)
                instruction = _online_instruction(task, doc, rng)
                if instruction is None:
                    rejected += 1  # tarea sin instrucción coherente: se omite
                    continue
            examples.append(
                SftExample(
                    id=f"{doc.id}-{task}",
                    doc_id=doc.id,
                    task=task,
                    instruction=instruction,
                    input=window,
                    response=response,
                    source=model,
                )
            )
    if rejected or errors:
        print(
            f"[synthesize] teacher: {rejected} salidas descartadas (contrato), "
            f"{errors} errores de llamada omitidos"
        )
    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_balance(spec: str) -> dict[str, float] | None:
    """'extraccion=0.3,qa=0.25' -> dict; 'none' -> None; 'default' -> DEFAULT."""
    if spec == "none":
        return None
    if spec == "default":
        return dict(DEFAULT_BALANCE)
    weights: dict[str, float] = {}
    for part in spec.split(","):
        task, _, value = part.partition("=")
        weights[task.strip()] = float(value)
    return weights


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Síntesis de dataset instruccional")
    parser.add_argument("--input", required=True, help="Corpus JSONL (salida de extract)")
    parser.add_argument("--output", required=True, help="JSONL de ejemplos generados")
    parser.add_argument(
        "--offline", action="store_true", help="Sin LLM: plantillas determinísticas"
    )
    parser.add_argument("--dpo", action="store_true", help="Genera pares chosen/rejected (offline)")
    parser.add_argument("--tasks", default=",".join(SFT_TASKS), help="Tareas separadas por comas")
    parser.add_argument("--max-per-task", type=int, default=None)
    parser.add_argument("--n-pairs", type=int, default=15, help="Nº de pares DPO")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance",
        default="default",
        help="Pesos por tarea 'tarea=w,...', 'default' o 'none' (solo offline)",
    )
    parser.add_argument(
        "--token-budget", type=int, default=DEFAULT_WINDOW_TOKENS,
        help="Presupuesto de tokens de la ventana de documento",
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="Para el gate de presupuesto")
    parser.add_argument(
        "--max-over-frac", type=float, default=0.02,
        help="Fracción máxima de ejemplos que pueden exceder seq-len",
    )
    parser.add_argument("--tokenizer", default=None, help="Ruta a tokenizer.json (opcional)")
    parser.add_argument("--model", default=None, help="Teacher (default: AF_TEACHER_MODEL)")
    parser.add_argument("--backend", default="litellm", choices=["litellm", "openai"])
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompts-dir", default="prompts")
    args = parser.parse_args(argv)

    settings = get_settings()
    docs = load_corpus(args.input)
    if args.max_docs is not None:
        docs = docs[: args.max_docs]
    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    counter = TokenCounter(args.tokenizer)
    print(f"[synthesize] contador de tokens: {counter.describe()}")

    if args.dpo:
        if not args.offline:
            raise SystemExit(
                "La generación DPO online no está en el alcance base; usa --offline. "
                "TODO(jairo): pares de preferencia online (teacher fuerte vs débil)."
            )
        pairs = build_dpo_offline(
            docs, seed=args.seed, n_pairs=args.n_pairs,
            counter=counter, token_budget=args.token_budget,
        )
        count = write_jsonl(args.output, (pair.model_dump() for pair in pairs))
        print(f"[synthesize] {count} pares DPO offline -> {args.output}")
        return 0

    if args.offline:
        examples = build_sft_offline(
            docs, tasks=tasks, seed=args.seed, max_per_task=args.max_per_task,
            counter=counter, token_budget=args.token_budget,
            balance=_parse_balance(args.balance),
        )
    else:
        examples = synthesize_online(
            docs,
            tasks,
            model=args.model or settings.teacher_model,
            backend=args.backend,
            api_base=args.api_base or settings.api_base,
            api_key=args.api_key or settings.api_key,
            prompts_dir=args.prompts_dir,
            counter=counter,
            token_budget=args.token_budget,
            seed=args.seed,
        )

    # Gate de presupuesto: nunca más un dataset que no cabe en seq_len.
    from adapterforge.data.format import SYSTEM_PROMPT  # import local: sin ciclo

    report = assert_token_budget(
        examples, counter, seq_len=args.seq_len,
        max_over_frac=args.max_over_frac, system_prompt=SYSTEM_PROMPT,
    )
    composition: dict[str, int] = {}
    for example in examples:
        composition[example.task] = composition.get(example.task, 0) + 1
    count = write_jsonl(args.output, (example.model_dump() for example in examples))
    mode = "offline" if args.offline else f"online ({args.model or settings.teacher_model})"
    print(f"[synthesize] {count} ejemplos SFT {mode} -> {args.output}")
    print(f"[synthesize] composición por tarea: {dict(sorted(composition.items()))}")
    print(
        f"[synthesize] presupuesto: p50={report['p50']} p95={report.get('p95')} "
        f"max={report['max']} tokens · {report['over']} exceden seq_len={args.seq_len}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
