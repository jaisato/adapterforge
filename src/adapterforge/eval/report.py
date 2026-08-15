"""Informe before/after en markdown comparando dos runs del benchmark.

Compara los ``results.json`` del modelo base ("before") y del adapter
("after") generados por :mod:`adapterforge.eval.domain_bench` contra una API
OpenAI-compatible configurable, y produce la tabla que va al README.

Además de los deltas, el informe cuantifica la incertidumbre: delta medio
pareado por caso con error estándar e IC 95 %, y — cuando los results.json
traen repeticiones — marca qué regresiones son robustas (los rangos de
score de before y after ni se solapan) frente a ruido de generación.

Uso::

    # 1) Evaluar ambos modelos intercalados en una sesión (n=3 por caso)
    python -m adapterforge.eval.domain_bench --api-base http://localhost:11434/v1 \
        --model qwen3-base-2507-eval --model juristune-v2b-eval \
        --repeats 3 --output-dir artifacts/eval/pair

    # 2) Informe comparado
    python -m adapterforge.eval.report --before artifacts/eval/pair/qwen3-base-2507-eval.json \
        --after artifacts/eval/pair/juristune-v2b-eval.json --output artifacts/eval/report.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _load_results(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "summary" not in data or "cases" not in data:
        raise ValueError(f"{path} no parece un results.json de domain_bench")
    return data


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _delta(after: float, before: float) -> str:
    diff = after - before
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.3f}"


def paired_stats(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float] | None:
    """Delta medio pareado por caso, SD, SE e IC 95 % (normal)."""
    before_scores = {c["id"]: float(c["score"]) for c in before.get("cases", [])}
    deltas = [
        float(c["score"]) - before_scores[c["id"]]
        for c in after.get("cases", [])
        if c["id"] in before_scores
    ]
    if len(deltas) < 2:
        return None
    mean = sum(deltas) / len(deltas)
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(len(deltas))
    return {
        "n": len(deltas),
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
    }


def _case_range(case: dict[str, Any]) -> tuple[float, float] | None:
    """(min, max) de las repeticiones del caso, si el results.json las trae."""
    if "score_min" in case and "score_max" in case:
        return float(case["score_min"]), float(case["score_max"])
    return None


def build_report(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Construye el informe markdown before/after."""
    lines = [
        "# Informe de evaluación — before/after",
        "",
        f"- **Before (base)**: `{before.get('model', '?')}`",
        f"- **After (adapter)**: `{after.get('model', '?')}`",
        f"- **Casos**: {after['summary'].get('n_cases', 0)}",
    ]
    generation = after.get("generation")
    if generation:
        lines.append(
            f"- **Generación**: temp {generation.get('temperature')} · "
            f"{generation.get('repeats')} repetición(es)/caso · "
            f"modelos intercalados en una sesión · {generation.get('timestamp_utc', '?')}"
        )
    lines += [
        "",
        "## Resumen por tipo de tarea",
        "",
        "| Tipo | Before | After | Delta |",
        "|---|---:|---:|---:|",
    ]
    before_by_type = before["summary"].get("by_type", {})
    after_by_type = after["summary"].get("by_type", {})
    for task in sorted(set(before_by_type) | set(after_by_type)):
        b = float(before_by_type.get(task, 0.0))
        a = float(after_by_type.get(task, 0.0))
        lines.append(f"| {task} | {_fmt(b)} | {_fmt(a)} | {_delta(a, b)} |")
    b_overall = float(before["summary"].get("overall", 0.0))
    a_overall = float(after["summary"].get("overall", 0.0))
    lines.append(f"| **overall** | **{_fmt(b_overall)}** | **{_fmt(a_overall)}** "
                 f"| **{_delta(a_overall, b_overall)}** |")

    stats = paired_stats(before, after)
    if stats:
        includes_zero = stats["ci_low"] <= 0.0 <= stats["ci_high"]
        lines += [
            "",
            "## Incertidumbre (deltas pareados por caso)",
            "",
            f"- Delta medio: {_delta(stats['mean'], 0.0)} · SD {stats['sd']:.3f} · "
            f"SE {stats['se']:.3f} (n={stats['n']})",
            f"- IC 95 %: ({stats['ci_low']:+.3f}, {stats['ci_high']:+.3f})",
        ]
        if includes_zero:
            lines.append(
                "- ⚠️ El IC incluye 0: con este número de casos la diferencia global "
                "**no es distinguible del ruido**. Mira las regresiones robustas caso a caso."
            )
        else:
            direction = "mejora" if stats["mean"] > 0 else "empeora"
            lines.append(f"- El IC excluye 0: el adapter {direction} de forma significativa.")

    before_cases = {c["id"]: c for c in before.get("cases", [])}
    regressions = []
    has_ranges = False
    for case in after.get("cases", []):
        base_case = before_cases.get(case["id"])
        if base_case is None:
            continue
        b_score = float(base_case["score"])
        a_score = float(case["score"])
        if a_score >= b_score - 1e-9:
            continue
        b_range = _case_range(base_case)
        a_range = _case_range(case)
        robust = None
        if b_range and a_range:
            has_ranges = True
            robust = a_range[1] < b_range[0] - 1e-9  # rangos disjuntos: no es ruido
        regressions.append((case["id"], b_score, a_score, robust))

    lines += ["", "## Regresiones por caso", ""]
    if regressions and has_ranges:
        lines += ["| Caso | Before | After | ¿Robusta? |", "|---|---:|---:|---|"]
        for cid, b, a, robust in regressions:
            flag = "?" if robust is None else ("sí" if robust else "no (solapa)")
            lines.append(f"| {cid} | {_fmt(b)} | {_fmt(a)} | {flag} |")
    elif regressions:
        lines += ["| Caso | Before | After |", "|---|---:|---:|"]
        lines += [f"| {cid} | {_fmt(b)} | {_fmt(a)} |" for cid, b, a, _ in regressions]
    else:
        lines.append("Sin regresiones: ningún caso puntúa peor con el adapter.")
    lines += [
        "",
        "> Generado con `python -m adapterforge.eval.report`. Métricas: "
        "exact-match por campos (extract), containment/ROUGE-L (qa) y "
        "ROUGE-L (resumen), con normalización numérica (separadores de "
        "miles y numerales en castellano) y limpieza de artefactos "
        "`<think>`/`tool_call` antes de puntuar.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Informe before/after del benchmark")
    parser.add_argument("--before", required=True, help="results.json del modelo base")
    parser.add_argument("--after", required=True, help="results.json del adapter")
    parser.add_argument("--output", required=True, help="Informe markdown de salida")
    args = parser.parse_args(argv)

    report = build_report(_load_results(args.before), _load_results(args.after))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"[report] informe escrito en {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
