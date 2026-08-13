"""Informe before/after en markdown comparando dos runs del benchmark.

Compara los ``results.json`` del modelo base ("before") y del adapter
("after") generados por :mod:`adapterforge.eval.domain_bench` contra una API
OpenAI-compatible configurable, y produce la tabla que va al README.

Uso::

    # 1) Evaluar ambos modelos (mismo endpoint vLLM con --enable-lora, p. ej.)
    python -m adapterforge.eval.domain_bench --api-base http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B-Instruct-2507 --output artifacts/eval/base.json
    python -m adapterforge.eval.domain_bench --api-base http://localhost:8000/v1 \
        --model juristune --output artifacts/eval/adapter.json

    # 2) Informe comparado
    python -m adapterforge.eval.report --before artifacts/eval/base.json \
        --after artifacts/eval/adapter.json --output artifacts/eval/report.md
"""

from __future__ import annotations

import argparse
import json
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


def build_report(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Construye el informe markdown before/after."""
    lines = [
        "# Informe de evaluación — before/after",
        "",
        f"- **Before (base)**: `{before.get('model', '?')}`",
        f"- **After (adapter)**: `{after.get('model', '?')}`",
        f"- **Casos**: {after['summary'].get('n_cases', 0)}",
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

    before_scores = {c["id"]: float(c["score"]) for c in before.get("cases", [])}
    regressions = [
        (c["id"], before_scores.get(c["id"], 0.0), float(c["score"]))
        for c in after.get("cases", [])
        if c["id"] in before_scores and float(c["score"]) < before_scores[c["id"]] - 1e-9
    ]
    lines += ["", "## Regresiones por caso", ""]
    if regressions:
        lines += ["| Caso | Before | After |", "|---|---:|---:|"]
        lines += [f"| {cid} | {_fmt(b)} | {_fmt(a)} |" for cid, b, a in regressions]
    else:
        lines.append("Sin regresiones: ningún caso puntúa peor con el adapter.")
    lines += [
        "",
        "> Generado con `python -m adapterforge.eval.report`. Métricas: "
        "exact-match por campos (extract), containment/ROUGE-L (qa) y "
        "ROUGE-L (resumen).",
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
