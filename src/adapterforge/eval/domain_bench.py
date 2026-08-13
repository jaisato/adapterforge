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

Modos de ejecución:

- **Live**: pregunta a un endpoint OpenAI-compatible (vLLM/Ollama/gateway)
  con ``--api-base`` y ``--model``; no requiere litellm.
- **Offline** (``--predictions``): puntúa un JSONL ya generado con
  ``{"id": ..., "prediction": ...}`` — es el modo de los tests y permite
  separar generación de puntuación.

``--min-score`` convierte el benchmark en quality gate (exit code 1 si no se
alcanza; lo usa release-hub.yml).

TODO(jairo): juez LLM con rúbrica 1-5 y doble juez con acuerdo inter-juez
para los resúmenes (flag --judge ya esbozado).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adapterforge.data.format import SYSTEM_PROMPT
from adapterforge.eval.metrics import contains_match, field_exact_match, rouge_l
from adapterforge.jsonl import read_jsonl

VALID_TASK_TYPES = ("extract", "qa", "resumen")


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
    task_type = case["task_type"]
    if task_type == "extract":
        score, detail = field_exact_match(prediction, case["expected"])
        return {"id": case["id"], "task_type": task_type, "score": score, "fields": detail}
    if task_type == "qa":
        rouge = rouge_l(prediction, case["expected"])
        contained = contains_match(prediction, case["expected"])
        return {
            "id": case["id"],
            "task_type": task_type,
            "score": max(rouge, 1.0 if contained else 0.0),
            "rouge_l": round(rouge, 4),
            "contains": contained,
        }
    rouge = rouge_l(prediction, case["expected"])
    return {"id": case["id"], "task_type": task_type, "score": rouge, "rouge_l": round(rouge, 4)}


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


def generate_predictions(
    cases: list[dict[str, Any]],
    *,
    model: str,
    api_base: str,
    api_key: str | None,
    max_tokens: int = 1024,
) -> dict[str, str]:
    """Genera predicciones contra un endpoint OpenAI-compatible."""
    from adapterforge.llm import chat_complete  # perezoso

    predictions: dict[str, str] = {}
    for case in cases:
        predictions[case["id"]] = chat_complete(
            build_case_messages(case),
            model,
            backend="openai",
            api_base=api_base,
            api_key=api_key,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    return predictions


def run_bench(
    cases: list[dict[str, Any]], predictions: dict[str, str], *, model_label: str
) -> dict[str, Any]:
    """Puntúa todas las predicciones y devuelve el resultado completo."""
    case_results = []
    missing = []
    for case in cases:
        prediction = predictions.get(case["id"])
        if prediction is None:
            missing.append(case["id"])
            case_results.append(
                {"id": case["id"], "task_type": case["task_type"], "score": 0.0, "missing": True}
            )
            continue
        case_results.append(score_case(case, prediction))
    summary = aggregate(case_results)
    return {
        "model": model_label,
        "summary": summary,
        "missing_predictions": missing,
        "cases": case_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark de dominio JurisTune")
    parser.add_argument("--bench", default="evals/domain_bench.jsonl")
    parser.add_argument("--predictions", default=None, help="JSONL {id, prediction} (offline)")
    parser.add_argument("--model", default=None, help="Modelo del endpoint (modo live)")
    parser.add_argument("--api-base", default=None, help="Endpoint OpenAI-compatible (modo live)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None, help="JSON de resultados")
    parser.add_argument("--min-score", type=float, default=None, help="Quality gate (exit 1)")
    args = parser.parse_args(argv)

    cases = load_bench_cases(args.bench)

    if args.predictions:
        records = read_jsonl(args.predictions)
        predictions = {str(r["id"]): str(r["prediction"]) for r in records}
        model_label = f"predictions:{Path(args.predictions).name}"
    else:
        if not args.model or not args.api_base:
            parser.error("modo live: se requieren --model y --api-base (o usa --predictions)")
        from adapterforge.settings import get_settings

        predictions = generate_predictions(
            cases,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key or get_settings().api_key,
        )
        model_label = args.model

    results = run_bench(cases, predictions, model_label=model_label)
    summary = results["summary"]
    print(
        f"[domain_bench] {model_label}: overall={summary['overall']} "
        f"por_tipo={summary['by_type']} n={summary['n_cases']}"
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if args.min_score is not None and summary["overall"] < args.min_score:
        print(
            f"[domain_bench] GATE FALLIDO: overall {summary['overall']} < "
            f"umbral {args.min_score}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
