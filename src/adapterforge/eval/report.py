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


def _stats_from_deltas(deltas: list[float]) -> dict[str, float] | None:
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


def paired_stats(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float] | None:
    """Delta medio pareado por caso, SD, SE e IC 95 % (normal)."""
    before_scores = {c["id"]: float(c["score"]) for c in before.get("cases", [])}
    deltas = [
        float(c["score"]) - before_scores[c["id"]]
        for c in after.get("cases", [])
        if c["id"] in before_scores
    ]
    return _stats_from_deltas(deltas)


def paired_stats_by_type(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Estadísticas pareadas (media, SD, SE, IC 95 %) desglosadas por ``task_type``."""
    before_cases = {c["id"]: c for c in before.get("cases", [])}
    deltas_by_type: dict[str, list[float]] = {}
    for case in after.get("cases", []):
        base_case = before_cases.get(case["id"])
        if base_case is None:
            continue
        task_type = str(case.get("task_type") or base_case.get("task_type") or "unknown")
        delta = float(case["score"]) - float(base_case["score"])
        deltas_by_type.setdefault(task_type, []).append(delta)

    result: dict[str, dict[str, float]] = {}
    for task_type in sorted(deltas_by_type):
        stats = _stats_from_deltas(deltas_by_type[task_type])
        if stats is not None:
            result[task_type] = stats
    return result


def _case_range(case: dict[str, Any]) -> tuple[float, float] | None:
    """(min, max) de las repeticiones del caso, si el results.json las trae."""
    if "score_min" in case and "score_max" in case:
        return float(case["score_min"]), float(case["score_max"])
    return None


def _regressions(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Casos emparejados que puntúan peor con el adapter, en el orden de ``after``.

    ``robust`` es True si los rangos de las repeticiones no se solapan (no es
    ruido de generación), False si se solapan y None si alguno de los dos
    results.json no trae rangos. Fuente única para el markdown y el JSON.
    """
    before_cases = {c["id"]: c for c in before.get("cases", [])}
    regressions: list[dict[str, Any]] = []
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
        robust: bool | None = None
        if b_range and a_range:
            robust = a_range[1] < b_range[0] - 1e-9  # rangos disjuntos: no es ruido
        regressions.append(
            {
                "id": case["id"],
                "task_type": str(case.get("task_type") or base_case.get("task_type") or "unknown"),
                "before": b_score,
                "after": a_score,
                "robust": robust,
            }
        )
    return regressions


def _unmatched(before: dict[str, Any], after: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Ids solo en ``before`` y solo en ``after`` (results de benches distintos).

    Las estadísticas pareadas y las regresiones ignoran estos casos: sin
    contarlos, un ``after`` evaluado contra otra versión del bench (o sin la
    mitad de los casos) pasaba el gate sin regresiones.
    """
    before_ids = [str(c["id"]) for c in before.get("cases", [])]
    after_ids = [str(c["id"]) for c in after.get("cases", [])]
    after_set, before_set = set(after_ids), set(before_ids)
    return (
        [cid for cid in before_ids if cid not in after_set],
        [cid for cid in after_ids if cid not in before_set],
    )


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

    by_type_stats = paired_stats_by_type(before, after)
    if by_type_stats:
        lines += [
            "",
            "### Incertidumbre por tipo de tarea",
            "",
            "| Tipo | n | Delta medio | IC 95 % |",
            "|---|---:|---:|---:|",
        ]
        for task_type, tstats in by_type_stats.items():
            ci_str = f"({tstats['ci_low']:+.3f}, {tstats['ci_high']:+.3f})"
            lines.append(
                f"| {task_type} | {int(tstats['n'])} | {_delta(tstats['mean'], 0.0)} | {ci_str} |"
            )

    regressions = _regressions(before, after)
    has_ranges = any(r["robust"] is not None for r in regressions)

    lines += ["", "## Regresiones por caso", ""]
    if regressions and has_ranges:
        lines += ["| Caso | Before | After | ¿Robusta? |", "|---|---:|---:|---|"]
        for r in regressions:
            robust = r["robust"]
            flag = "?" if robust is None else ("sí" if robust else "no (solapa)")
            lines.append(f"| {r['id']} | {_fmt(r['before'])} | {_fmt(r['after'])} | {flag} |")
    elif regressions:
        lines += ["| Caso | Before | After |", "|---|---:|---:|"]
        lines += [
            f"| {r['id']} | {_fmt(r['before'])} | {_fmt(r['after'])} |" for r in regressions
        ]
    else:
        lines.append("Sin regresiones: ningún caso puntúa peor con el adapter.")

    only_before, only_after = _unmatched(before, after)
    if only_before or only_after:
        lines += [
            "",
            "## Casos sin pareja",
            "",
            "> ⚠️ Los dos results.json no cubren los mismos casos (¿benches distintos?): "
            "estos casos quedan fuera de los deltas pareados y de las regresiones.",
            "",
        ]
        if only_before:
            lines.append(f"- Solo en before ({len(only_before)}): {', '.join(only_before)}")
        if only_after:
            lines.append(f"- Solo en after ({len(only_after)}): {', '.join(only_after)}")
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


def build_summary_json(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Construye un resumen estructurado JSON para CI gates y automatización."""
    regressions = [
        {
            "id": r["id"],
            "task_type": r["task_type"],
            "before": round(r["before"], 4),
            "after": round(r["after"], 4),
            "delta": round(r["after"] - r["before"], 4),
            "robust": r["robust"],
        }
        for r in _regressions(before, after)
    ]
    only_before, only_after = _unmatched(before, after)
    b_overall = float(before["summary"].get("overall", 0.0))
    a_overall = float(after["summary"].get("overall", 0.0))
    return {
        "before_model": before.get("model", "?"),
        "after_model": after.get("model", "?"),
        "overall_before": round(b_overall, 4),
        "overall_after": round(a_overall, 4),
        "overall_delta": round(a_overall - b_overall, 4),
        "paired_stats": paired_stats(before, after),
        "paired_stats_by_type": paired_stats_by_type(before, after),
        "regressions": regressions,
        "regression_count": len(regressions),
        "robust_regression_count": sum(1 for r in regressions if r["robust"] is True),
        "unmatched_before": only_before,
        "unmatched_after": only_after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Informe before/after del benchmark")
    parser.add_argument("--before", required=True, help="results.json del modelo base")
    parser.add_argument("--after", required=True, help="results.json del adapter")
    parser.add_argument("--output", required=True, help="Informe markdown de salida")
    parser.add_argument(
        "--json-output", default=None, help="Ruta opcional para el resumen JSON estructurado"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help=(
            "Devuelve código de salida 1 si hay regresiones robustas (o regresiones sin "
            "rangos) o si hay casos de before que no están en after"
        ),
    )
    args = parser.parse_args(argv)

    before_data = _load_results(args.before)
    after_data = _load_results(args.after)
    report = build_report(before_data, after_data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"[report] informe escrito en {output_path}")

    summary = build_summary_json(before_data, after_data)
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[report] resumen JSON escrito en {json_path}")

    if args.fail_on_regression:
        blocking = [r["id"] for r in summary["regressions"] if r["robust"] in (True, None)]
        if blocking:
            print(f"[report] GATE: {len(blocking)} regresión(es) bloqueante(s): {blocking}")
            return 1
        if summary["unmatched_before"]:
            # Un caso que el adapter no ha corrido no puede contar como «sin regresión».
            print(
                f"[report] GATE: {len(summary['unmatched_before'])} caso(s) de before "
                f"ausentes en after: {summary['unmatched_before']}"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
