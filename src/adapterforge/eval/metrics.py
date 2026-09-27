"""Métricas de evaluación propias, deterministas y sin dependencias pesadas.

- ``exact_match`` / ``contains_match``: para campos extraíbles cortos
  (plazos, órganos, cuantías) sobre texto normalizado.
- ``rouge_l``: ROUGE-L (F1 sobre la subsecuencia común más larga) propio,
  para resúmenes y respuestas largas.
- ``field_exact_match``: extrae el primer objeto JSON de la predicción y
  compara campo a campo contra el esperado (tareas de extracción).
- ``strip_response_artifacts``: limpia bloques ``<think>`` y tokens
  ``tool_call`` que algunos serves decodifican como texto literal.

La normalización canonicaliza números: elimina separadores de miles
(«7.500» → «7500») y convierte numerales en castellano a dígitos
(«ocho millones» → «8000000»), de modo que una respuesta semánticamente
correcta no puntúe 0.0 por el formato de superficie del número.
"""

from __future__ import annotations

import json
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")
# Separador de miles entre grupos de 3 dígitos: «7.500», «1.719.936», «600,000».
_THOUSANDS = re.compile(r"(?<=\d)[.,](?=\d{3}(?!\d))")
# Fechas: «2028-07-01» (ISO) y «1 de julio de 2028» canonicalizan ambas a
# «1 7 2028» — la base respondía fechas ISO semánticamente correctas que la
# métrica puntuaba 0.0 frente a referencias en formato largo.
_MONTH_NUMBER = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_SPANISH_DATE = re.compile(
    rf"\b(\d{{1,2}})\s+de\s+({'|'.join(_MONTH_NUMBER)})(?:\s+de\s+(\d{{4}}))?\b"
)


def _canonicalize_dates(text: str) -> str:
    text = _ISO_DATE.sub(
        lambda m: f"{int(m.group(3))} {int(m.group(2))} {int(m.group(1))}", text
    )
    return _SPANISH_DATE.sub(
        lambda m: f"{int(m.group(1))} {_MONTH_NUMBER[m.group(2)]}"
        + (f" {m.group(3)}" if m.group(3) else ""),
        text,
    )


# Separador decimal u hora restante entre dígitos -> marcador que sobrevive a
# _PUNCT («3,5», «3.5» y también «9.30»/«9:30» acaban como «3_5»/«9_30»).
_DECIMAL = re.compile(r"(?<=\d)[.,:](?=\d)")
# Números canónicos (con posible marcador decimal) dentro de texto normalizado.
_NUMBER_TOKEN = re.compile(r"\d+(?:_\d+)?")
# Artefactos de respuesta: bloque de razonamiento completo, razonamiento
# truncado (tag <think> sin cerrar: no hay respuesta final que puntuar) y
# tags sueltos. Los bloques <tool_call>...</tool_call> NO se eliminan
# enteros a propósito: los adapters dañados envuelven la respuesta real en
# esas etiquetas, así que solo se quitan los tags.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED = re.compile(r"<think>.*\Z", re.DOTALL)
_SPECIAL_TAGS = re.compile(r"</?(?:think|tool_call)>")

# Numerales en castellano (sin tildes: se aplican tras quitar acentos).
_NUMBER_WORDS = {
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiun": 21, "veintiuno": 21, "veintiuna": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100,
    "doscientos": 200, "doscientas": 200, "trescientos": 300, "trescientas": 300,
    "cuatrocientos": 400, "cuatrocientas": 400, "quinientos": 500, "quinientas": 500,
    "seiscientos": 600, "seiscientas": 600, "setecientos": 700, "setecientas": 700,
    "ochocientos": 800, "ochocientas": 800, "novecientos": 900, "novecientas": 900,
}
_MULTIPLIERS = {"mil": 1_000, "millon": 1_000_000, "millones": 1_000_000}


def strip_response_artifacts(text: str) -> str:
    """Elimina bloques ``<think>...</think>`` y tags ``tool_call`` sueltos.

    Los adapters entrenados con el template thinking de Unsloth abren sus
    respuestas con estos tokens especiales, que Ollama decodifica como texto
    literal y contaminan cualquier métrica de superficie.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    cleaned = _SPECIAL_TAGS.sub("", cleaned)
    return cleaned.strip()


def _parse_number_words(tokens: list[str], start: int) -> tuple[int, int]:
    """Parsea la secuencia numérica máxima desde ``tokens[start]``.

    Devuelve ``(valor, tokens_consumidos)``; ``consumidos == 0`` si en
    ``start`` no empieza un numeral.
    """
    total = 0
    current = 0
    consumed = 0
    saw_word = False
    digit_seed = False
    last_word_value: int | None = None
    i = start
    while i < len(tokens):
        token = tokens[i]
        if token in _NUMBER_WORDS:
            value = _NUMBER_WORDS[token]
            if digit_seed:
                break  # «8 ocho» no es un numeral compuesto
            if last_word_value is not None and value >= last_word_value:
                # La composición castellana solo decrece en magnitud
                # («ciento treinta», «treinta y cinco»); «uno dos» o
                # «dos, tres» son numerales independientes: no se funden.
                break
            current += value
            last_word_value = value
            saw_word = True
        elif token.isdigit():
            # Un dígito solo abre secuencia («8 millones»); nunca se funden
            # dos tokens numéricos consecutivos («3 5 y 7» queda intacto).
            if saw_word or i != start:
                break
            current = int(token)
            saw_word = True
            digit_seed = True
        elif token in _MULTIPLIERS:
            multiplier = _MULTIPLIERS[token]
            if multiplier >= 1_000_000:
                if current == 0 and total == 0:
                    # «millones» sin cuantificador («varios millones de
                    # euros») no es un número concreto: se deja intacto.
                    break
                # «millones» escala todo lo acumulado: «mil millones»,
                # «cuarenta y ocho mil ochocientos setenta y un millones».
                total = (total + current) * multiplier
            else:
                total += (current or 1) * multiplier  # «veinte mil»
            current = 0
            last_word_value = None  # nuevo grupo tras el multiplicador
            saw_word = True
        elif (
            token == "y"
            and (current % 100) % 10 == 0
            and 30 <= current % 100 <= 90
            and i + 1 < len(tokens)
            and _NUMBER_WORDS.get(tokens[i + 1], 10) < 10
        ):
            pass  # conector «treinta y cinco», «ciento treinta y cinco»...
        else:
            break
        consumed = i - start + 1
        i += 1
    if not saw_word:
        return 0, 0
    return total + current, consumed


def _spelled_numbers_to_digits(text: str) -> str:
    """Convierte numerales en castellano a dígitos sobre texto ya normalizado."""
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        value, consumed = _parse_number_words(tokens, i)
        if consumed:
            out.append(str(value))
            i += consumed
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def normalize_text(text: str) -> str:
    """Minúsculas, sin tildes ni puntuación, con números canonicalizados."""
    lowered = text.lower().strip()
    decomposed = unicodedata.normalize("NFKD", lowered)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    with_dates = _canonicalize_dates(without_accents)
    canonical_digits = _DECIMAL.sub("_", _THOUSANDS.sub("", with_dates))
    without_punct = _PUNCT.sub(" ", canonical_digits)
    collapsed = _SPACES.sub(" ", without_punct).strip()
    return _spelled_numbers_to_digits(collapsed)


def exact_match(prediction: str, reference: str) -> bool:
    """Coincidencia exacta sobre texto normalizado."""
    return normalize_text(prediction) == normalize_text(reference)


# Artículos y determinantes sin contenido jurídico: «en un plazo de veinte
# días» y «en el plazo de veinte días» son la misma afirmación. Se eliminan
# SOLO para el containment (simétrico para todos los modelos evaluados) y
# sobre el texto CRUDO: tras normalizar, «un» ya se ha convertido en «1» y
# sería indistinguible de un numeral genuino («1 de julio»).
_ARTICLE_RE = re.compile(r"\b(?:el|la|los|las|un|una|unos|unas|lo)\b", re.IGNORECASE)


def _normalize_without_articles(text: str) -> str:
    return normalize_text(_ARTICLE_RE.sub(" ", text))


def contains_match(prediction: str, reference: str) -> bool:
    """La referencia normalizada aparece dentro de la predicción normalizada.

    Si el containment literal falla, se reintenta sin artículos gramaticales:
    un «un/el» intercambiado no cambia el contenido normativo y puntuaba 0.0
    una respuesta correcta (caso real: eb-142).
    """
    normalized_reference = normalize_text(reference)
    if not normalized_reference:
        return False
    if f" {normalized_reference} " in f" {normalize_text(prediction)} ":
        return True
    stripped_reference = _normalize_without_articles(reference)
    if not stripped_reference:
        return False
    return f" {stripped_reference} " in f" {_normalize_without_articles(prediction)} "


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longitud de la subsecuencia común más larga (DP con fila rodante)."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[len(b)]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 sobre tokens normalizados (0.0-1.0)."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_first_json_object(text: str) -> dict | None:
    """Extrae el primer objeto JSON balanceado de un texto (o ``None``).

    Tolera que el modelo envuelva el JSON en explicaciones o en fences de
    markdown; recorre llaves balanceadas respetando strings.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def _value_match(predicted: str, expected: str) -> bool:
    """Compara un valor de campo con su esperado, en ambas direcciones.

    Además del exact-match y del containment esperado-en-predicción, acepta
    la dirección inversa (predicción dentro del esperado) cuando la parte
    numérica coincide exactamente: cubre respuestas más escuetas que la
    referencia («7500» vs «7.500 euros») sin dar por buena una fecha a la
    que le faltan componentes («2027» vs «1 de enero de 2027»).
    """
    if exact_match(predicted, expected) or contains_match(predicted, expected):
        return True
    normalized_prediction = normalize_text(predicted)
    normalized_expected = normalize_text(expected)
    if not normalized_prediction or normalized_prediction not in normalized_expected:
        return False
    predicted_numbers = _NUMBER_TOKEN.findall(normalized_prediction)
    return bool(predicted_numbers) and predicted_numbers == _NUMBER_TOKEN.findall(
        normalized_expected
    )


def field_exact_match(prediction: str, expected: dict[str, str]) -> tuple[float, dict[str, bool]]:
    """Exact-match por campo contra el JSON de la predicción.

    Para cada campo esperado: acierto si el valor del JSON de la predicción
    casa con el esperado (:func:`_value_match`) o, como red de seguridad, si
    el esperado aparece contenido en la predicción completa (los modelos a
    veces responden en prosa). Devuelve (fracción de aciertos, detalle).
    """
    if not expected:
        return 0.0, {}
    parsed = extract_first_json_object(prediction) or {}
    detail: dict[str, bool] = {}
    for field, expected_value in expected.items():
        predicted_value = parsed.get(field)
        if isinstance(predicted_value, list):
            hit = any(_value_match(str(item), str(expected_value)) for item in predicted_value)
        elif predicted_value is not None:
            hit = _value_match(str(predicted_value), str(expected_value))
        else:
            hit = contains_match(prediction, str(expected_value))
        detail[field] = hit
    score = sum(detail.values()) / len(detail)
    return score, detail
