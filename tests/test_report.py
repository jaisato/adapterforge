"""Tests del informe before/after y de helpers de entrenamiento puros."""

from __future__ import annotations

from adapterforge.eval.report import build_report
from adapterforge.train.common import extract_loss_curve


def _results(model: str, extract: float, qa: float, case_scores: dict[str, float]) -> dict:
    return {
        "model": model,
        "summary": {
            "overall": round((extract + qa) / 2, 4),
            "by_type": {"extract": extract, "qa": qa},
            "n_cases": len(case_scores),
        },
        "cases": [
            {"id": cid, "task_type": "extract", "score": score}
            for cid, score in case_scores.items()
        ],
    }


def test_build_report_shows_deltas_and_regressions():
    before = _results("base", extract=0.40, qa=0.50, case_scores={"c1": 1.0, "c2": 0.0})
    after = _results("adapter", extract=0.80, qa=0.70, case_scores={"c1": 0.5, "c2": 1.0})
    report = build_report(before, after)
    assert "| extract | 0.400 | 0.800 | +0.400 |" in report
    assert "`base`" in report and "`adapter`" in report
    # c1 empeora (1.0 -> 0.5): debe aparecer como regresión.
    assert "| c1 | 1.000 | 0.500 |" in report


def test_build_report_without_regressions():
    before = _results("base", extract=0.2, qa=0.2, case_scores={"c1": 0.0})
    after = _results("adapter", extract=0.9, qa=0.9, case_scores={"c1": 1.0})
    report = build_report(before, after)
    assert "Sin regresiones" in report


def test_build_report_includes_uncertainty_section():
    before = _results("base", extract=0.40, qa=0.50, case_scores={"c1": 1.0, "c2": 0.0})
    after = _results("adapter", extract=0.80, qa=0.70, case_scores={"c1": 0.5, "c2": 1.0})
    report = build_report(before, after)
    assert "## Incertidumbre" in report
    assert "IC 95 %" in report


def test_build_report_flags_robust_regressions_with_repeat_ranges():
    before = _results("base", extract=0.9, qa=0.9, case_scores={"c1": 1.0, "c2": 1.0})
    after = _results("adapter", extract=0.3, qa=0.3, case_scores={"c1": 0.4, "c2": 0.8})
    for case in before["cases"]:
        case["score_min"] = case["score_max"] = case["score"]
    # c1: rango after (0.3-0.5) disjunto del before (1.0) -> regresión robusta.
    after["cases"][0]["score_min"] = 0.3
    after["cases"][0]["score_max"] = 0.5
    # c2: rango after (0.6-1.0) solapa con before (1.0) -> ruido.
    after["cases"][1]["score_min"] = 0.6
    after["cases"][1]["score_max"] = 1.0
    report = build_report(before, after)
    assert "¿Robusta?" in report
    assert "| c1 | 1.000 | 0.400 | sí |" in report
    assert "| c2 | 1.000 | 0.800 | no (solapa) |" in report


def test_extract_loss_curve_ignores_non_loss_entries():
    history = [
        {"learning_rate": 1e-4},
        {"loss": 2.5, "step": 1},
        {"loss": 1.9, "step": 2},
        {"train_loss": 1.7},  # resumen final de TRL: no cuenta
        {"loss": 1.2, "step": 3},
    ]
    first, last = extract_loss_curve(history)
    assert first == 2.5
    assert last == 1.2
    assert extract_loss_curve([]) == (None, None)
