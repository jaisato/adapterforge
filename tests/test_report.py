"""Tests del informe before/after y de helpers de entrenamiento puros."""

from __future__ import annotations

from adapterforge.eval.report import build_report, build_summary_json, main, paired_stats_by_type
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


def test_paired_stats_by_type_and_cli_json_gate(tmp_path):
    import json

    before = _results("base", extract=0.40, qa=0.50, case_scores={"c1": 1.0, "c2": 0.2})
    after = _results("adapter", extract=0.80, qa=0.70, case_scores={"c1": 0.5, "c2": 0.9})
    by_type = paired_stats_by_type(before, after)
    assert "extract" in by_type
    assert by_type["extract"]["n"] == 2

    summary = build_summary_json(before, after)
    assert summary["regression_count"] == 1
    assert summary["regressions"][0]["id"] == "c1"

    b_file = tmp_path / "before.json"
    a_file = tmp_path / "after.json"
    out_md = tmp_path / "report.md"
    out_json = tmp_path / "summary.json"
    b_file.write_text(json.dumps(before), encoding="utf-8")
    a_file.write_text(json.dumps(after), encoding="utf-8")

    rc = main(
        [
            "--before",
            str(b_file),
            "--after",
            str(a_file),
            "--output",
            str(out_md),
            "--json-output",
            str(out_json),
            "--fail-on-regression",
        ]
    )
    assert rc == 1
    assert out_md.exists() and "### Incertidumbre por tipo de tarea" in out_md.read_text("utf-8")
    assert json.loads(out_json.read_text("utf-8"))["regression_count"] == 1


def _cli(tmp_path, before: dict, after: dict) -> tuple[int, dict]:
    import json

    b_file, a_file = tmp_path / "before.json", tmp_path / "after.json"
    b_file.write_text(json.dumps(before), encoding="utf-8")
    a_file.write_text(json.dumps(after), encoding="utf-8")
    out_json = tmp_path / "summary.json"
    rc = main(
        [
            "--before", str(b_file), "--after", str(a_file),
            "--output", str(tmp_path / "report.md"), "--json-output", str(out_json),
            "--fail-on-regression",
        ]
    )
    return rc, json.loads(out_json.read_text("utf-8"))


def test_gate_falla_si_after_no_cubre_los_casos_de_before(tmp_path):
    """Sin regresiones entre los emparejados, pero el adapter no corrió c2 ni c3."""
    before = _results("base", 0.5, 0.5, case_scores={"c1": 0.5, "c2": 0.9, "c3": 0.9})
    after = _results("adapter", 0.9, 0.9, case_scores={"c1": 0.9, "c4": 0.1})
    rc, summary = _cli(tmp_path, before, after)
    assert summary["regression_count"] == 0
    assert summary["unmatched_before"] == ["c2", "c3"]
    assert summary["unmatched_after"] == ["c4"]
    assert rc == 1
    report = (tmp_path / "report.md").read_text("utf-8")
    assert "## Casos sin pareja" in report
    assert "Solo en before (2): c2, c3" in report


def test_gate_no_bloquea_regresiones_que_solapan_con_el_ruido(tmp_path):
    before = _results("base", 0.5, 0.5, case_scores={"c1": 0.8})
    after = _results("adapter", 0.5, 0.5, case_scores={"c1": 0.7})
    before["cases"][0].update(score_min=0.6, score_max=1.0)
    after["cases"][0].update(score_min=0.5, score_max=0.9)
    rc, summary = _cli(tmp_path, before, after)
    assert summary["regressions"][0]["robust"] is False
    assert summary["unmatched_before"] == [] and summary["unmatched_after"] == []
    assert rc == 0
    report = (tmp_path / "report.md").read_text("utf-8")
    assert "| c1 | 0.800 | 0.700 | no (solapa) |" in report
    assert "Casos sin pareja" not in report
