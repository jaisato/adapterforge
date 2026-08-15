"""Benchmark propio del dominio jurídico-administrativo (JurisTune).

Casos en ``evals/domain_bench.jsonl`` con el esquema::

    {"id": "...", "task_type": "extract|qa|resumen",
     "instruction": "...", "input": "<extracto del documento>",
     "expected": {"plazo": "tres meses"} | "respuesta de referencia"}

Métricas por tipo de caso:

- ``extract``: exact-match por campo sobre el JSON de la respuesta
  (:func:`adapterforge.eval.metrics.field_exact_match`).
- ``qa``: containment de la referencia + ROUGE-L (score = máximo de ambos,
  porque una respuesta correcta puede parafrasear o citar).
- ``resumen``: ROUGE-L contra el resumen de referencia.

Antes de puntuar se limpian artefactos de respuesta (bloques ``<think>``,
tags ``tool_call`` decodificados como texto); la predicción cruda se
persiste sin tocar en el results.json.

Modos de ejecución:

- **Live**: pregunta a un endpoint OpenAI-compatible (vLLM/Ollama/gateway)
  con ``--api-base`` y uno o más ``--model``. Con varios modelos las
  peticiones se **intercalan por caso** en la misma sesión del servidor
  (before/after en idénticas condiciones) y se escribe un results.json por
  modelo en ``--output-dir``. ``--repeats`` (por defecto 3) genera n
  predicciones por caso y el score del caso es la media.
- **Offline** (``--predictions``): puntúa un JSONL ya generado con
  ``{"id": ..., "prediction": ...}`` — es el modo de los tests y permite
  separar generación de puntuación.

Cada results.json incluye las predicciones crudas por repetición y los
parámetros de generación, para que toda regresión sea auditable sin volver
a generar.

``--min-score`` convierte el benchmark en quality gate (exit code 1 si no se
alcanza; lo usa release-hub.yml). Con varios modelos, el gate se aplica al
último ``--model`` de la lista (el adapter, por convención).

TODO(jairo): juez LLM con rúbrica 1-5 y doble juez con acuerdo inter-juez
para los resúmenes (flag --judge ya esbozado).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adapterforge.data.format import SYSTEM_PROMPT
from adapterforge.eval.metrics import (
    contains_match,
    field_exact_match,
    rouge_l,
    strip_response_artifacts,
)
from adapterforge.jsonl import read_jsonl

VALID_TASK_TYPES = ("extract", "qa", "resumen")

_GENERATION_RETRIES = 3
_RETRY_BACKOFF_S = 5.0


def load_bench_cases(path: str | Path) -> list[dict[str, Any]]:
    """Carga y valida estructuralmente los casos del benchmark."""
    cases = read_jsonl(path)
    seen: set[str] = set()
    for index, case in enumerate(cases, start=1):
        for key in ("id", "task_type", "instruction", "input", "expected"):
            if key not in case:
                raise ValueError(f"Caso #{index} sin clave obligatoria {key!r}")
        if case["task_type"] not in VALID_TASK_TYPES:
            raise ValueError(
                f"Caso {case['id']}: task_type inválido {case['task_type']!r} "
                f"(válidos: {VALID_TASK_TYPES})"
            )
        if case["task_type"] == "extract" and not isinstance(case["expected"], dict):
            raise ValueError(f"Caso {case['id']}: 'extract' requiere expected como objeto")
        if case["task_type"] in ("qa", "resumen") and not isinstance(case["expected"], str):
            raise ValueError(f"Caso {case['id']}: '{case['task_type']}' requiere expected string")
        if case["id"] in seen:
            raise ValueError(f"Identificador de caso duplicado: {case['id']}")
        seen.add(case["id"])
    return cases


def score_case(case: dict[str, Any], prediction: str) -> dict[str, Any]:
    """Puntúa un caso individual y devuelve el detalle."""
    cleaned = strip_response_artifacts(prediction)
    task_type = case["task_type"]
    if task_type == "extract":
        score, detail = field_exact_match(cleaned, case["expected"])
        return {"id": case["id"], "task_type": task_type, "score": score, "fields": detail}
    if task_type == "qa":
        rouge = rouge_l(cleaned, case["expected"])
        contained = contains_match(cleaned, case["expected"])
        return {
            "id": case["id"],
            "task_type": task_type,
            "score": max(rouge, 1.0 if contained else 0.0),
            "rouge_l": round(rouge, 4),
            "contains": contained,
        }
    rouge = rouge_l(cleaned, case["expected"])
    return {"id": case["id"], "task_type": task_type, "score": rouge, "rouge_l": round(rouge, 4)}


def _merge_repeats(
    predictions: list[str], repeat_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Funde los resultados de n repeticiones de un caso en una entrada.

    ``score`` pasa a ser la media; el detalle de primer nivel (fields,
    rouge_l, contains) es el de la primera repetición por compatibilidad, y
    ``repeats`` conserva predicción cruda + detalle de cada repetición.
    """
    scores = [float(result["score"]) for result in repeat_results]
    merged = dict(repeat_results[0])
    merged["score"] = sum(scores) / len(scores)
    if len(scores) > 1:
        merged["score_min"] = min(scores)
        merged["score_max"] = max(scores)
        merged["score_std"] = round(statistics.stdev(scores), 4)
    merged["repeats"] = [
        {
            **{k: v for k, v in result.items() if k not in ("id", "task_type")},
            "prediction": prediction,
        }
        for result, prediction in zip(repeat_results, predictions, strict=True)
    ]
    return merged


def aggregate(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Media global y por tipo de tarea."""
    if not case_results:
        return {"overall": 0.0, "by_type": {}, "n_cases": 0}
    by_type: dict[str, list[float]] = {}
    for result in case_results:
        by_type.setdefault(result["task_type"], []).append(float(result["score"]))
    return {
        "overall": round(sum(r["score"] for r in case_results) / len(case_results), 4),
        "by_type": {
            task: round(sum(scores) / len(scores), 4) for task, scores in sorted(by_type.items())
        },
        "n_cases": len(case_results),
    }


def build_case_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    """Mensajes de chat para un caso (mismo formato que el entrenamiento)."""
    user = f"{case['instruction']}\n\n### Documento\n{case['input']}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _chat_complete_with_retries(messages: list[dict[str, str]], model: str, **kwargs: Any) -> str:
    """chat_complete con reintentos: un fallo transitorio no aborta el run."""
    from adapterforge.llm import LlmError, chat_complete  # perezoso

    last_error: Exception | None = None
    for attempt in range(1, _GENERATION_RETRIES + 1):
        try:
            return chat_complete(messages, model, **kwargs)
        except (LlmError, json.JSONDecodeError) as exc:
            # JSONDecodeError: respuesta no-JSON del endpoint (proxy caído,
            # HTML de error...) que httpx.Response.json() propaga sin envolver.
            last_error = exc
            if attempt < _GENERATION_RETRIES:
                time.sleep(_RETRY_BACKOFF_S * attempt)
    raise last_error  # type: ignore[misc]


def generate_predictions(
    cases: list[dict[str, Any]],
    *,
    models: list[str],
    api_base: str,
    api_key: str | None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    repeats: int = 1,
) -> dict[str, dict[str, list[str]]]:
    """Genera predicciones contra un endpoint OpenAI-compatible.

    Con varios modelos, las peticiones se intercalan por caso (base, adapter,
    base, adapter...) para que ambos vean el servidor en el mismo estado.
    Devuelve ``{modelo: {case_id: [predicción por repetición]}}``.
    """
    predictions: dict[str, dict[str, list[str]]] = {
        model: {case["id"]: [] for case in cases} for model in models
    }
    total = repeats * len(cases)
    for repeat in range(repeats):
        for index, case in enumerate(cases, start=1):
            for model in models:
                predictions[model][case["id"]].append(
                    _chat_complete_with_retries(
                        build_case_messages(case),
                        model,
                        backend="openai",
                        api_base=api_base,
                        api_key=api_key,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                )
            done = repeat * len(cases) + index
            print(f"[domain_bench] {done}/{total} · r{repeat + 1} · {case['id']}", flush=True)
    return predictions


def run_bench(
    cases: list[dict[str, Any]],
    predictions: dict[str, str | list[str]],
    *,
    model_label: str,
    generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Puntúa todas las predicciones y devuelve el resultado completo.

    ``predictions`` acepta un string por caso (modo offline clásico) o una
    lista de strings (repeticiones); el score del caso es la media.
    """
    case_results = []
    missing = []
    for case in cases:
        raw = predictions.get(case["id"])
        repeat_predictions = [raw] if isinstance(raw, str) else list(raw or [])
        if not repeat_predictions:
            missing.append(case["id"])
            case_results.append(
                {"id": case["id"], "task_type": case["task_type"], "score": 0.0, "missing": True}
            )
            continue
        repeat_results = [score_case(case, prediction) for prediction in repeat_predictions]
        case_results.append(_merge_repeats(repeat_predictions, repeat_results))
    summary = aggregate(case_results)
    results: dict[str, Any] = {
        "model": model_label,
        "summary": summary,
        "missing_predictions": missing,
        "cases": case_results,
    }
    if generation is not None:
        results["generation"] = generation
    return results


def _safe_model_filename(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", model) + ".json"


def _write_results(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[domain_bench] resultados escritos en {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark de dominio JurisTune")
    parser.add_argument("--bench", default="evals/domain_bench.jsonl")
    parser.add_argument("--predictions", default=None, help="JSONL {id, prediction} (offline)")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Modelo del endpoint (modo live); repetible para before/after intercalado",
    )
    parser.add_argument("--api-base", default=None, help="Endpoint OpenAI-compatible (modo live)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--repeats", type=int, default=3, help="Generaciones por caso (live)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output", default=None, help="JSON de resultados (un solo modelo)")
    parser.add_argument(
        "--output-dir", default=None, help="Directorio de resultados (uno o varios modelos)"
    )
    parser.add_argument("--min-score", type=float, default=None, help="Quality gate (exit 1)")
    args = parser.parse_args(argv)

    cases = load_bench_cases(args.bench)

    if args.predictions:
        records = read_jsonl(args.predictions)
        offline_predictions = {str(r["id"]): str(r["prediction"]) for r in records}
        model_label = f"predictions:{Path(args.predictions).name}"
        all_results = [run_bench(cases, offline_predictions, model_label=model_label)]
        if args.output_dir:
            output_paths = [Path(args.output_dir) / _safe_model_filename(model_label)]
        elif args.output:
            output_paths = [Path(args.output)]
        else:
            output_paths = []
    else:
        if not args.models or not args.api_base:
            parser.error("modo live: se requieren --model y --api-base (o usa --predictions)")
        if len(args.models) > 1 and not args.output_dir:
            parser.error("con varios --model se requiere --output-dir")
        # Rutas de salida ANTES de generar: una colisión de nombres
        # sanitizados debe fallar rápido, no tras minutos de generación
        # (y nunca sobreescribir en silencio los resultados de otro modelo).
        output_paths = []
        if args.output_dir:
            output_paths = [
                Path(args.output_dir) / _safe_model_filename(model) for model in args.models
            ]
            if len(set(output_paths)) != len(output_paths):
                parser.error(
                    "los nombres de --model colisionan tras sanitizar el nombre de "
                    "archivo; usa nombres que difieran en algo más que ':' o '/'"
                )
        elif args.output:
            output_paths = [Path(args.output)]
        from adapterforge.settings import get_settings

        api_key = args.api_key or get_settings().api_key
        predictions = generate_predictions(
            cases,
            models=args.models,
            api_base=args.api_base,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            repeats=max(1, args.repeats),
        )
        generation = {
            "api_base": args.api_base,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "repeats": max(1, args.repeats),
            "interleaved_models": args.models,
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        all_results = [
            run_bench(cases, predictions[model], model_label=model, generation=generation)
            for model in args.models
        ]

    for results in all_results:
        summary = results["summary"]
        print(
            f"[domain_bench] {results['model']}: overall={summary['overall']} "
            f"por_tipo={summary['by_type']} n={summary['n_cases']}"
        )
    # strict=False: sin --output/--output-dir no se escribe nada (solo stdout).
    for results, output_path in zip(all_results, output_paths, strict=False):
        _write_results(results, output_path)

    if args.min_score is not None:
        gated = all_results[-1]
        if gated["summary"]["overall"] < args.min_score:
            print(
                f"[domain_bench] GATE FALLIDO ({gated['model']}): overall "
                f"{gated['summary']['overall']} < umbral {args.min_score}"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
