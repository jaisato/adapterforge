"""Detectores estructurales con validación de etiquetas.

Corren sobre el articulado completo (índices de artículo, no números: los
BOE que modifican otras normas repiten números) y devuelven *hits* con el
valor verbatim y la posición, para que el generador pueda anclar cada
ejemplo a la ventana que verá el modelo:

- plazos (:func:`find_deadline_hits`, :func:`deadline_purpose`): duración o
  fecha validadas, con el propósito del plazo para desambiguar;
- obligaciones y sanciones (:func:`find_obligation_hits`,
  :func:`find_sanction_hits`, :func:`find_amount_hits`);
- cuantías y tramos sancionadores (:func:`max_amount_in`,
  :func:`find_sanction_tiers`);
- validaciones (:func:`classify_deadline`, :func:`_future_date`).

Las capturas desbordadas, las frases anafóricas y los «(no se detectan)» se
descartan aquí, no aguas abajo.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from adapterforge.data.extract import Article, RawDocument, split_sentences
from adapterforge.data.synthesize.text import _MONTHS_ES, _ends_sentence

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
