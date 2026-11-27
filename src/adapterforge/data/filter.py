"""Filtrado de calidad del dataset sintético.

Cuatro barreras, de más barata a más cara:

1. **Heurísticas**: longitudes razonables, idioma español básico (ratio de
   stopwords) y ratio de copia literal respecto al documento fuente
   (containment de shingles largos).
2. **Deduplicación near-dup**: shingles de palabras + similitud de Jaccard
   (implementación propia, sin datasketch; O(n²) con poda por buckets de
   tamaño, suficiente para datasets de decenas de miles de ejemplos).
3. **LLM-judge opcional** (``--judge``): rúbrica 1-5 de fidelidad/formato vía
   litellm (import perezoso); descarta por debajo del umbral.

Uso CLI::

    python -m adapterforge.data.filter \
        --input data/interim/sft_raw.jsonl \
        --corpus data/interim/documents.jsonl \
        --output data/interim/sft_filtered.jsonl \
        --report data/interim/filter_report.json

TODO(jairo): MinHash-LSH propio para corpus >100k ejemplos (buckets de firmas)
y estadísticas de dataset en notebook de EDA.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from adapterforge.eval.metrics import strip_response_artifacts
from adapterforge.jsonl import read_jsonl, write_jsonl

# Stopwords españolas frecuentes para el detector básico de idioma.
STOPWORDS_ES = frozenset(
    {
        "de", "la", "el", "que", "en", "y", "a", "los", "las", "del", "se",
        "por", "con", "una", "un", "para", "su", "al", "como", "más", "o",
        "este", "esta", "es", "son", "ser", "lo", "no", "sus", "le", "ya",
        "entre", "cuando", "sobre", "también", "hasta", "desde", "durante",
    }
)

_WORD = re.compile(r"[a-záéíóúüñ]+", re.IGNORECASE)

# Umbrales por defecto (ajustables por CLI).
DEFAULT_MIN_RESPONSE_CHARS = 60
DEFAULT_MAX_RESPONSE_CHARS = 8000
DEFAULT_MIN_INSTRUCTION_CHARS = 10
DEFAULT_MAX_INPUT_CHARS = 12000  # red de seguridad: la ventana ya acota en tokens
DEFAULT_STOPWORD_RATIO = 0.15
DEFAULT_COPY_RATIO = 0.90
DEFAULT_DEDUP_THRESHOLD = 0.90
COPY_SHINGLE_K = 13  # shingles largos: solo detectan copias literales extensas
DEDUP_SHINGLE_K = 5

# Mínimos de respuesta por tarea: una extracción legítima puede ser un JSON
# de 20 caracteres; exigirle 60 mataría justo la tarea alineada con el bench.
MIN_RESPONSE_CHARS_BY_TASK = {"extraccion": 15}

# Umbral de copia por tarea: en extracción el citado verbatim ES la tarea
# (el bench puntúa spans exactos); en qa la respuesta enmarca una cita
# literal. El umbral general sigue cazando la regurgitación en resumen/llano.
MAX_COPY_RATIO_BY_TASK = {"extraccion": 1.01, "qa": 0.98}

# Basura conocida: boilerplate del teacher sin parsear, no-respuestas y
# «objetos» que son un apartado numeral pelado (regresión del separador de
# frases: «Su objeto es el siguiente: 1.»).
_BOILERPLATE = re.compile(
    r"\b(?:PREGUNTA|RESPUESTA)\s*:"
    r"|\(no se detectan\)"
    r"|(?:objeto es el siguiente|\*\*Objeto\*\*|¿De qué trata\?\*\*|En pocas palabras,)"
    r"\s*:?\s*\d+\.(?:\s|$)"
)


def tokenize(text: str) -> list[str]:
    """Tokenización simple en minúsculas (solo palabras con letras)."""
    return [match.group(0).lower() for match in _WORD.finditer(text)]


def shingles(text: str, k: int = DEDUP_SHINGLE_K) -> set[tuple[str, ...]]:
    """Conjunto de k-shingles de palabras de un texto.

    Si el texto tiene menos de ``k`` palabras devuelve un único shingle con
    todas ellas (evita conjuntos vacíos artificiales).
    """
    tokens = tokenize(text)
    if not tokens:
        return set()
    if len(tokens) < k:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    """Similitud de Jaccard entre dos conjuntos de shingles."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    return intersection / union


def containment(needle: set[tuple[str, ...]], haystack: set[tuple[str, ...]]) -> float:
    """Fracción de shingles de ``needle`` presentes en ``haystack``."""
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


def stopword_ratio(text: str) -> float:
    """Ratio de stopwords españolas sobre el total de palabras."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in STOPWORDS_ES)
    return hits / len(tokens)


def is_spanish_basic(text: str, min_ratio: float = DEFAULT_STOPWORD_RATIO) -> bool:
    """Detector básico de español: suficiente para descartar salidas raras."""
    return stopword_ratio(text) >= min_ratio


def copy_ratio(response: str, source_text: str, k: int = COPY_SHINGLE_K) -> float:
    """Cuánta parte de la respuesta es copia literal (runs de >= k palabras).

    Una respuesta con menos de ``k`` palabras no puede contener un «run
    largo»: devuelve 0.0. Sin esta exención, el shingle único degenerado
    marcaría como copia justo las respuestas cortas de extracción, cuyo
    citado verbatim es el comportamiento deseado (v3: extraccion cayó 62 %
    por esto antes de la exención).
    """
    if len(tokenize(response)) < k:
        return 0.0
    return containment(shingles(response, k=k), shingles(source_text, k=k))


def _doc_token(doc_id: str) -> str:
    """Identificador de documento con dígitos transliterados a letras."""
    return "".join(
        chr(ord("a") + int(ch)) if ch.isdigit() else ch for ch in doc_id.lower()
    )


def dedup_by_jaccard(
    records: list[dict[str, Any]],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    k: int = DEDUP_SHINGLE_K,
) -> tuple[list[dict[str, Any]], int]:
    """Elimina near-duplicados conservando la primera aparición.

    Compara instrucción+respuesta con Jaccard sobre k-shingles. Devuelve
    (registros únicos, nº de descartados).
    """
    kept: list[dict[str, Any]] = []
    kept_shingles: list[set[tuple[str, ...]]] = []
    removed = 0
    for record in records:
        # El doc_id forma parte de la clave: la misma respuesta corta sobre
        # documentos DISTINTOS es señal legítima (inputs diferentes), no un
        # duplicado. Los clones largos de plantilla entre documentos siguen
        # cayendo (unos tokens distintos apenas mueven el Jaccard de textos
        # largos). ``_doc_token`` translitera los dígitos porque ``tokenize``
        # solo conserva letras y «BOE-A-2026-0001/0002» tokenizarían igual.
        text = (
            f"{_doc_token(str(record.get('doc_id', '')))} "
            f"{record.get('instruction', '')} {record.get('response', '')}"
        )
        current = shingles(text, k=k)
        size = len(current)
        is_duplicate = False
        for other in kept_shingles:
            # Poda barata: si los tamaños difieren mucho, el Jaccard no puede
            # alcanzar el umbral (|A∩B| <= min(|A|,|B|)).
            if min(size, len(other)) / max(size, len(other), 1) < threshold:
                continue
            if jaccard(current, other) >= threshold:
                is_duplicate = True
                break
        if is_duplicate:
            removed += 1
        else:
            kept.append(record)
            kept_shingles.append(current)
    return kept, removed


def apply_heuristics(
    records: Iterable[dict[str, Any]],
    docs_text_by_id: dict[str, str],
    *,
    min_response_chars: int = DEFAULT_MIN_RESPONSE_CHARS,
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
    min_instruction_chars: int = DEFAULT_MIN_INSTRUCTION_CHARS,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    max_copy_ratio: float = DEFAULT_COPY_RATIO,
    min_stopword_ratio: float = DEFAULT_STOPWORD_RATIO,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Aplica los filtros heurísticos y devuelve (aceptados, motivos)."""
    kept: list[dict[str, Any]] = []
    reasons = {
        "too_short": 0,
        "too_long": 0,
        "input_too_long": 0,
        "bad_instruction": 0,
        "boilerplate": 0,
        "not_spanish": 0,
        "copy": 0,
    }
    for record in records:
        instruction = str(record.get("instruction", ""))
        response = str(record.get("response", ""))
        task = str(record.get("task", ""))
        min_chars = MIN_RESPONSE_CHARS_BY_TASK.get(task, min_response_chars)
        if len(instruction) < min_instruction_chars:
            reasons["bad_instruction"] += 1
            continue
        if len(response) < min_chars:
            reasons["too_short"] += 1
            continue
        if len(response) > max_response_chars:
            reasons["too_long"] += 1
            continue
        if len(str(record.get("input", ""))) > max_input_chars:
            reasons["input_too_long"] += 1
            continue
        if _BOILERPLATE.search(response):
            reasons["boilerplate"] += 1
            continue
        # El detector de idioma penaliza los JSON cortos legítimos: se exime
        # a las respuestas que son puro JSON (las valida el formato mismo).
        if not response.lstrip().startswith("{") and not is_spanish_basic(
            response, min_ratio=min_stopword_ratio
        ):
            reasons["not_spanish"] += 1
            continue
        source_text = docs_text_by_id.get(str(record.get("doc_id", "")), "")
        task_copy_ceiling = MAX_COPY_RATIO_BY_TASK.get(task, max_copy_ratio)
        if source_text and copy_ratio(response, source_text) > task_copy_ceiling:
            reasons["copy"] += 1
            continue
        kept.append(record)
    return kept, reasons


# ---------------------------------------------------------------------------
# LLM-judge opcional (lazy)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """Eres un revisor de calidad de datasets instruccionales jurídicos.
Evalúa el siguiente ejemplo (instrucción, documento y respuesta) con una nota
entera de 1 a 5 según esta rúbrica:

- Fidelidad: la respuesta solo afirma cosas presentes en el documento.
- Formato: respeta el formato pedido por la instrucción.
- Utilidad: la respuesta es completa y útil para la tarea.

Responde ÚNICAMENTE con el número entero (1-5), sin texto adicional.

### Instrucción
{instruction}

### Documento
{document}

### Respuesta
{response}
"""


# Primer número de la respuesta, decimales incluidos («2.5», «1,5», «.5»).
_JUDGE_GRADE = re.compile(r"[.,]?\d+(?:[.,]\d+)*", re.ASCII)


def judge_score(
    record: dict[str, Any],
    *,
    model: str,
    backend: str = "litellm",
    api_base: str | None = None,
    api_key: str | None = None,
) -> int:
    """Puntúa un ejemplo 1-5 con el juez LLM (import perezoso del cliente)."""
    from adapterforge.llm import chat_complete  # perezoso

    prompt = _JUDGE_PROMPT.format(
        instruction=record.get("instruction", ""),
        document=str(record.get("input", ""))[:4000],
        response=record.get("response", ""),
    )
    raw = chat_complete(
        [{"role": "user", "content": prompt}],
        model,
        backend=backend,
        api_base=api_base,
        api_key=api_key,
        temperature=0.0,
        max_tokens=8,
    )
    # La nota es el primer número de la respuesta, ya sin bloques <think>, y
    # tiene que ser un entero de 1 a 5: con r"[1-5]" a secas, «<think>comparo
    # 1, 2 y 3</think>2» puntuaba 1 y un «10/10» fuera de escala también (el «1»
    # de «10»). Un decimal no se salta: en «2.5/5» la nota no es el 5 de la escala.
    match = _JUDGE_GRADE.search(strip_response_artifacts(raw))
    grade = match.group(0) if match else ""
    if not grade.isdigit() or not 1 <= int(grade) <= 5:
        raise ValueError(f"El juez no devolvió una nota 1-5: {raw!r}")
    return int(grade)


def apply_judge(
    records: list[dict[str, Any]],
    *,
    model: str,
    min_score: int = 3,
    backend: str = "litellm",
    api_base: str | None = None,
    api_key: str | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Filtra por nota del juez; anota ``judge_score`` en cada registro.

    Devuelve (aceptados, descartados por nota, descartados por error). Un
    fallo del juez en un ejemplo (transporte o nota ilegible) no aborta el
    lote, como en la síntesis online: ese ejemplo no se ha podido verificar
    y se descarta, contado aparte. Si el juez falla en masa, el gate de
    composición por tarea de :func:`main` corta el filtrado.
    """
    from adapterforge.llm import LlmError  # perezoso

    kept: list[dict[str, Any]] = []
    removed = 0
    errors = 0
    for record in records:
        try:
            score = judge_score(
                record, model=model, backend=backend, api_base=api_base, api_key=api_key
            )
        except (LlmError, ValueError) as exc:
            errors += 1
            print(f"[filter] juez sin nota para {record.get('id', '?')} ({exc}); se descarta")
            continue
        record = dict(record)
        record["judge_score"] = score
        if score >= min_score:
            kept.append(record)
        else:
            removed += 1
    return kept, removed, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filtrado de calidad del dataset")
    parser.add_argument("--input", required=True, help="Ejemplos JSONL (de synthesize)")
    parser.add_argument("--corpus", required=True, help="Corpus JSONL (para ratio de copia)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None, help="JSON con estadísticas del filtrado")
    parser.add_argument("--dedup-threshold", type=float, default=DEFAULT_DEDUP_THRESHOLD)
    parser.add_argument("--max-copy-ratio", type=float, default=DEFAULT_COPY_RATIO)
    parser.add_argument(
        "--max-task-drop",
        type=float,
        default=0.5,
        help="Fracción máxima de una tarea que el filtrado puede eliminar (gate)",
    )
    parser.add_argument("--judge", action="store_true", help="Activa el LLM-judge (requiere API)")
    parser.add_argument("--judge-model", default=None, help="Default: AF_JUDGE_MODEL")
    parser.add_argument("--judge-min-score", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--backend", default="litellm", choices=["litellm", "openai"])
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args(argv)

    records = read_jsonl(args.input)
    corpus = read_jsonl(args.corpus)
    docs_text_by_id = {str(doc["id"]): str(doc.get("texto", "")) for doc in corpus}

    kept, reasons = apply_heuristics(
        records, docs_text_by_id, max_copy_ratio=args.max_copy_ratio
    )
    kept, dedup_removed = dedup_by_jaccard(kept, threshold=args.dedup_threshold)

    judge_removed = 0
    judge_errors = 0
    if args.judge:
        from adapterforge.settings import get_settings  # perezoso, coherente con el resto

        judge_model = args.judge_model or get_settings().judge_model
        kept, judge_removed, judge_errors = apply_judge(
            kept,
            model=judge_model,
            min_score=args.judge_min_score,
            backend=args.backend,
            api_base=args.api_base,
            api_key=args.api_key,
        )

    # Gate de composición: ningún filtrado puede vaciar una tarea en silencio
    # (lección v2b: el filtro de longitud aniquiló extraccion y el teacher).
    def _by_task(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            task = str(row.get("task", "?"))
            counts[task] = counts.get(task, 0) + 1
        return counts

    task_in = _by_task(records)
    task_out = _by_task(kept)
    task_drop = {
        task: round(1 - task_out.get(task, 0) / total, 3)
        for task, total in sorted(task_in.items())
        if total
    }
    print(f"[filter] composición por tarea (entrada -> salida): "
          f"{ {t: (task_in[t], task_out.get(t, 0)) for t in sorted(task_in)} }")
    excessive = {t: d for t, d in task_drop.items() if d > args.max_task_drop}
    if excessive:
        raise SystemExit(
            f"[filter] GATE: el filtrado elimina más del {args.max_task_drop:.0%} "
            f"de alguna tarea: {excessive}. Revisa los filtros antes de entrenar."
        )

    count = write_jsonl(args.output, kept)
    stats = {
        "input": len(records),
        "kept": count,
        "removed_heuristics": reasons,
        "removed_dedup": dedup_removed,
        "removed_judge": judge_removed,
        "removed_judge_errors": judge_errors,
        "task_drop": task_drop,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"[filter] {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
