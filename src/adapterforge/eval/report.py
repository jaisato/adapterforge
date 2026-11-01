"""Informe before/after en markdown comparando dos runs del benchmark.

Compara los ``results.json`` del modelo base ("before") y del adapter
("after") generados por :mod:`adapterforge.eval.domain_bench` contra una API
OpenAI-compatible configurable, y produce la tabla que va al README.

Además de los deltas, el informe cuantifica la incertidumbre: delta medio
pareado por caso con error estándar e IC 95 % (media ± t·SE, con el cuantil
t de Student de n−1 grados de libertad; 1.96 con n > 30), global y por tipo
de tarea (el IC por tipo solo con n ≥ 5), y — cuando los results.json traen
repeticiones — marca qué regresiones son robustas (los rangos de score de
before y after ni se solapan) frente a ruido de generación.

El gate (``--fail-on-regression``) bloquea las regresiones robustas y, sin
rangos de repeticiones, cualquier bajada mayor que ``--regression-tolerance``
(0 por defecto: cualquier bajada bloquea).

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


#: Cuantiles t de Student al 97.5 % (IC bilateral del 95 %) por grados de libertad.
#: Con n ≥ 2 deltas hay n−1 g.l.; a partir de 30 g.l. se usa la normal (1.96).
_T_975: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045,
}  # fmt: skip
_Z_975 = 1.96

#: Deltas mínimos por tipo de tarea para mostrar un IC: por debajo, el cuantil t
#: (12.7 con n=2, 4.3 con n=3) hace el intervalo tan ancho que no informa, y
#: el 1.96 que se usaba antes lo hacía absurdamente estrecho.
MIN_N_CI_BY_TYPE = 5


def t_quantile(n: int) -> float:
    """Cuantil del IC 95 % para ``n`` deltas pareados: t(0.975, n−1) si n ≤ 30, si no 1.96."""
    if n < 2:
        raise ValueError(f"hacen falta al menos 2 deltas para un IC (n={n})")
    return _T_975.get(n - 1, _Z_975)


def _stats_from_deltas(deltas: list[float], min_n_ci: int = 2) -> dict[str, float | None] | None:
    """Media, SD, SE e IC 95 % de los deltas; None con menos de 2 deltas.

    Con ``n < min_n_ci`` el IC se marca como insuficiente (``ci_low``/``ci_high``/
    ``quantile`` a None) pero se devuelven n, media, SD y SE.
    """
    n = len(deltas)
    if n < 2:
        return None
    mean = sum(deltas) / n
    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(n)
    stats: dict[str, float | None] = {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "quantile": None,
        "ci_low": None,
        "ci_high": None,
    }
    if n >= min_n_ci:
        q = t_quantile(n)
        stats.update(quantile=q, ci_low=mean - q * se, ci_high=mean + q * se)
    return stats


def paired_stats(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float | None] | None:
    """Delta medio pareado por caso, SD, SE e IC 95 % (t de Student; normal con n > 30)."""
    before_scores = {c["id"]: float(c["score"]) for c in before.get("cases", [])}
    deltas = [
        float(c["score"]) - before_scores[c["id"]]
        for c in after.get("cases", [])
        if c["id"] in before_scores
    ]
    return _stats_from_deltas(deltas)


def paired_stats_by_type(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, dict[str, float | None]]:
    """Estadísticas pareadas (media, SD, SE, IC 95 %) desglosadas por ``task_type``.

    El IC solo se calcula con al menos ``MIN_N_CI_BY_TYPE`` deltas del tipo; con
    menos se devuelven n y media con el IC a None (el informe lo marca como
    insuficiente).
    """
    before_cases = {c["id"]: c for c in before.get("cases", [])}
    deltas_by_type: dict[str, list[float]] = {}
    for case in after.get("cases", []):
        base_case = before_cases.get(case["id"])
        if base_case is None:
            continue
        task_type = str(case.get("task_type") or base_case.get("task_type") or "unknown")
        delta = float(case["score"]) - float(base_case["score"])
        deltas_by_type.setdefault(task_type, []).append(delta)

    result: dict[str, dict[str, float | None]] = {}
    for task_type in sorted(deltas_by_type):
        stats = _stats_from_deltas(deltas_by_type[task_type], min_n_ci=MIN_N_CI_BY_TYPE)
        if stats is not None:
            result[task_type] = stats
    return result


def _case_range(case: dict[str, Any]) -> tuple[float, float] | None:
    """(min, max) de las repeticiones del caso, si el results.json las trae."""
    if "score_min" in case and "score_max" in case:
        return float(case["score_min"]), float(case["score_max"])
    return None


def _blocks_gate(regression: dict[str, Any], tolerance: float) -> bool:
    """Si la regresión bloquea el gate: robusta, o sin rangos y mayor que la tolerancia."""
    if regression["robust"] is True:
        return True
    if regression["robust"] is None:
        return regression["before"] - regression["after"] > tolerance + 1e-9
    return False


def _regressions(
    before: dict[str, Any], after: dict[str, Any], tolerance: float = 0.0
) -> list[dict[str, Any]]:
    """Casos emparejados que puntúan peor con el adapter, en el orden de ``after``.

    ``robust`` es True si los rangos de las repeticiones no se solapan (no es
    ruido de generación), False si se solapan y None si alguno de los dos
    results.json no trae rangos. ``blocking`` dice si la regresión para el gate:
    las robustas siempre; las que no tienen rangos, solo si la bajada supera
    ``tolerance``. Fuente única para el markdown y el JSON.
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
        regression = {
            "id": case["id"],
            "task_type": str(case.get("task_type") or base_case.get("task_type") or "unknown"),
            "before": b_score,
            "after": a_score,
            "robust": robust,
        }
        regression["blocking"] = _blocks_gate(regression, tolerance)
        regressions.append(regression)
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


def _ci(stats: dict[str, float | None]) -> tuple[float, float]:
    """(ci_low, ci_high) como floats; solo para estadísticas con IC calculado."""
    ci_low, ci_high = stats["ci_low"], stats["ci_high"]
    assert ci_low is not None and ci_high is not None
    return ci_low, ci_high


def _quantile_label(stats: dict[str, float | None]) -> str:
    n = int(stats["n"] or 0)
    return f"t de Student, {n - 1} g.l." if n - 1 in _T_975 else "normal"


def build_report(
    before: dict[str, Any], after: dict[str, Any], regression_tolerance: float = 0.0
) -> str:
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
        ci_low, ci_high = _ci(stats)
        includes_zero = ci_low <= 0.0 <= ci_high
        lines += [
            "",
            "## Incertidumbre (deltas pareados por caso)",
            "",
            f"- Delta medio: {_delta(stats['mean'], 0.0)} · SD {stats['sd']:.3f} · "
            f"SE {stats['se']:.3f} (n={stats['n']})",
            f"- IC 95 %: ({ci_low:+.3f}, {ci_high:+.3f}) — media ± {stats['quantile']:.3f}·SE "
            f"({_quantile_label(stats)})",
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
            if tstats["ci_low"] is None:
                ci_str = f"insuficiente (n < {MIN_N_CI_BY_TYPE})"
            else:
                ci_low, ci_high = _ci(tstats)
                ci_str = f"({ci_low:+.3f}, {ci_high:+.3f})"
            lines.append(
                f"| {task_type} | {int(tstats['n'])} | {_delta(tstats['mean'], 0.0)} | {ci_str} |"
            )
        lines += [
            "",
            f"IC = media ± t(0.975, n−1)·SE (1.96 con n > 30); con menos de {MIN_N_CI_BY_TYPE} "
            "casos del tipo no se muestra.",
        ]

    regressions = _regressions(before, after, regression_tolerance)
    has_ranges = any(r["robust"] is not None for r in regressions)

    lines += ["", "## Regresiones por caso", ""]
    if regressions and has_ranges:
        lines += ["| Caso | Before | After | ¿Robusta? |", "|---|---:|---:|---|"]
        for r in regressions:
            robust = r["robust"]
            flag = "?" if robust is None else ("sí" if robust else "no (solapa)")
            lines.append(f"| {r['id']} | {_fmt(r['before'])} | {_fmt(r['after'])} | {flag} |")
    elif regressions:
        lines += ["| Caso | Before | After | Delta |", "|---|---:|---:|---:|"]
        lines += [
            f"| {r['id']} | {_fmt(r['before'])} | {_fmt(r['after'])} "
            f"| {_delta(r['after'], r['before'])} |"
            for r in regressions
        ]
        if regression_tolerance > 0:
            within = sum(1 for r in regressions if not r["blocking"])
            lines += [
                "",
                f"Sin rangos de repeticiones: una bajada de hasta {regression_tolerance:.3f} "
                f"no bloquea el gate (`--regression-tolerance`); {within} de {len(regressions)} "
                "regresión(es) dentro de la tolerancia.",
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


def build_summary_json(
    before: dict[str, Any], after: dict[str, Any], regression_tolerance: float = 0.0
) -> dict[str, Any]:
    """Construye un resumen estructurado JSON para CI gates y automatización."""
    regressions = [
        {
            "id": r["id"],
            "task_type": r["task_type"],
            "before": round(r["before"], 4),
            "after": round(r["after"], 4),
            "delta": round(r["after"] - r["before"], 4),
            "robust": r["robust"],
            "blocking": r["blocking"],
        }
        for r in _regressions(before, after, regression_tolerance)
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
        "regression_tolerance": regression_tolerance,
        "blocking_regression_count": sum(1 for r in regressions if r["blocking"]),
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
            "Devuelve código de salida 1 si hay regresiones robustas (rangos de repeticiones "
            "disjuntos), regresiones sin rangos que superen --regression-tolerance, o casos "
            "de before que no están en after"
        ),
    )
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help=(
            "Bajada máxima de score por caso que el gate tolera cuando los results.json no "
            "traen rangos de repeticiones (sin --repeats). Por defecto 0.0: cualquier bajada "
            "bloquea. No afecta a las regresiones robustas, que bloquean siempre"
        ),
    )
    args = parser.parse_args(argv)
    if args.regression_tolerance < 0:
        parser.error("--regression-tolerance debe ser >= 0")

    before_data = _load_results(args.before)
    after_data = _load_results(args.after)
    report = build_report(before_data, after_data, args.regression_tolerance)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"[report] informe escrito en {output_path}")

    summary = build_summary_json(before_data, after_data, args.regression_tolerance)
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[report] resumen JSON escrito en {json_path}")

    if args.fail_on_regression:
        blocking = [r["id"] for r in summary["regressions"] if r["blocking"]]
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
