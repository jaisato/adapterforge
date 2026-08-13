"""Métricas de evaluación propias, deterministas y sin dependencias pesadas.

- ``exact_match`` / ``contains_match``: para campos extraíbles cortos
  (plazos, órganos, cuantías) sobre texto normalizado.
- ``rouge_l``: ROUGE-L (F1 sobre la subsecuencia común más larga) propio,
  para resúmenes y respuestas largas.
- ``field_exact_match``: extrae el primer objeto JSON de la predicción y
  compara campo a campo contra el esperado (tareas de extracción).
"""

from __future__ import annotations

import json
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Minúsculas, sin tildes, sin puntuación y con espacios colapsados."""
    lowered = text.lower().strip()
    decomposed = unicodedata.normalize("NFKD", lowered)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    without_punct = _PUNCT.sub(" ", without_accents)
    return _SPACES.sub(" ", without_punct).strip()


def exact_match(prediction: str, reference: str) -> bool:
    """Coincidencia exacta sobre texto normalizado."""
    return normalize_text(prediction) == normalize_text(reference)


def contains_match(prediction: str, reference: str) -> bool:
    """La referencia normalizada aparece dentro de la predicción normalizada."""
    normalized_reference = normalize_text(reference)
    if not normalized_reference:
        return False
    return normalized_reference in normalize_text(prediction)


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


def field_exact_match(prediction: str, expected: dict[str, str]) -> tuple[float, dict[str, bool]]:
    """Exact-match por campo contra el JSON de la predicción.

    Para cada campo esperado: acierto si el valor normalizado coincide con el
    del JSON de la predicción o, como red de seguridad, si aparece contenido
    en la predicción completa (los modelos a veces responden en prosa).
    Devuelve (fracción de aciertos, detalle por campo).
    """
    if not expected:
        return 0.0, {}
    parsed = extract_first_json_object(prediction) or {}
    detail: dict[str, bool] = {}
    for field, expected_value in expected.items():
        predicted_value = parsed.get(field)
        if isinstance(predicted_value, list):
            hit = any(
                contains_match(str(item), str(expected_value))
                or exact_match(str(item), str(expected_value))
                for item in predicted_value
            )
        elif predicted_value is not None:
            hit = exact_match(str(predicted_value), str(expected_value)) or contains_match(
                str(predicted_value), str(expected_value)
            )
        else:
            hit = contains_match(prediction, str(expected_value))
        detail[field] = hit
    score = sum(detail.values()) / len(detail)
    return score, detail
