"""Tests del informe before/after y de helpers de entrenamiento puros."""

from __future__ import annotations

import pytest

from adapterforge.eval.report import (
    MIN_N_CI_BY_TYPE,
    build_report,
    build_summary_json,
    main,
    paired_stats,
    paired_stats_by_type,
    t_quantile,
)
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


def _cli(tmp_path, before: dict, after: dict, *extra: str) -> tuple[int, dict]:
    import json

    b_file, a_file = tmp_path / "before.json", tmp_path / "after.json"
    b_file.write_text(json.dumps(before), encoding="utf-8")
    a_file.write_text(json.dumps(after), encoding="utf-8")
    out_json = tmp_path / "summary.json"
    rc = main(
        [
            "--before", str(b_file), "--after", str(a_file),
            "--output", str(tmp_path / "report.md"), "--json-output", str(out_json),
            "--fail-on-regression", *extra,
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


# ----------------------------------------------------------------- cuantil t y n mínimo


def test_t_quantile_usa_student_hasta_30_deltas_y_normal_despues():
    assert t_quantile(2) == 12.706  # 1 g.l.
    assert t_quantile(3) == 4.303
    assert t_quantile(5) == 2.776  # 4 g.l.
    assert t_quantile(30) == 2.045  # 29 g.l.: último de la tabla
    assert t_quantile(31) == 1.96
    assert t_quantile(111) == 1.96  # el bench v2 completo
    with pytest.raises(ValueError):
        t_quantile(1)


def test_ic_con_dos_deltas_ya_no_excluye_cero():
    """Deltas [0.10, 0.30]: con 1.96 el IC era (0.004, 0.396) y «excluía 0»; con t(1) no."""
    before = _results("base", 0.5, 0.5, case_scores={"c1": 0.5, "c2": 0.5})
    after = _results("adapter", 0.7, 0.7, case_scores={"c1": 0.6, "c2": 0.8})
    stats = paired_stats(before, after)
    assert stats["n"] == 2 and stats["quantile"] == 12.706
    assert stats["mean"] == pytest.approx(0.2)
    assert stats["se"] == pytest.approx(0.1)
    assert stats["ci_low"] == pytest.approx(0.2 - 12.706 * 0.1)  # -1.07
    assert stats["ci_high"] == pytest.approx(0.2 + 12.706 * 0.1)  # +1.47
    report = build_report(before, after)
    assert "IC 95 %: (-1.071, +1.471)" in report
    assert "t de Student, 1 g.l." in report
    assert "no es distinguible del ruido" in report
    assert "excluye 0" not in report


def test_ic_global_con_muchos_deltas_usa_la_normal():
    n = 40
    before = _results("base", 0.5, 0.5, case_scores={f"c{i}": 0.5 for i in range(n)})
    after = _results(
        "adapter", 0.6, 0.6, case_scores={f"c{i}": 0.5 + 0.1 * (i % 3) for i in range(n)}
    )
    stats = paired_stats(before, after)
    assert stats["n"] == n and stats["quantile"] == 1.96
    assert stats["ci_low"] == pytest.approx(stats["mean"] - 1.96 * stats["se"])
    assert "(normal)" in build_report(before, after)


def test_ic_por_tipo_exige_cinco_deltas():
    assert MIN_N_CI_BY_TYPE == 5
    before = _results("base", 0.5, 0.5, case_scores={"c1": 0.5, "c2": 0.5})
    after = _results("adapter", 0.7, 0.7, case_scores={"c1": 0.6, "c2": 0.8})
    extract = paired_stats_by_type(before, after)["extract"]
    assert extract["n"] == 2 and extract["mean"] == pytest.approx(0.2)
    assert extract["ci_low"] is None and extract["ci_high"] is None
    assert extract["quantile"] is None
    report = build_report(before, after)
    assert "| extract | 2 | +0.200 | insuficiente (n < 5) |" in report
    # En el JSON el IC insuficiente es null, no un intervalo
    summary = build_summary_json(before, after)
    assert summary["paired_stats_by_type"]["extract"]["ci_low"] is None

    # Con 5 deltas sí hay IC, con t(4 g.l.) = 2.776
    before5 = _results("base", 0.5, 0.5, case_scores={f"c{i}": 0.5 for i in range(5)})
    after5 = _results(
        "adapter", 0.7, 0.7, case_scores={f"c{i}": 0.5 + 0.1 * (i % 2) for i in range(5)}
    )
    extract5 = paired_stats_by_type(before5, after5)["extract"]
    assert extract5["n"] == 5 and extract5["quantile"] == 2.776
    assert extract5["ci_high"] == pytest.approx(extract5["mean"] + 2.776 * extract5["se"])
    assert "insuficiente" not in build_report(before5, after5)


# ----------------------------------------------------------------- tolerancia del gate


def test_tolerancia_sin_rangos_deja_pasar_bajadas_pequenas(tmp_path):
    """Sin --repeats no hay rangos: por defecto cualquier bajada bloquea; con tolerancia, no."""
    before = _results("base", 0.5, 0.5, case_scores={"c1": 0.80, "c2": 0.90})
    after = _results("adapter", 0.5, 0.5, case_scores={"c1": 0.75, "c2": 0.90})
    rc, summary = _cli(tmp_path, before, after)
    assert rc == 1 and summary["regression_tolerance"] == 0.0
    assert summary["regressions"][0]["robust"] is None
    assert summary["regressions"][0]["blocking"] is True
    assert summary["blocking_regression_count"] == 1

    rc, summary = _cli(tmp_path, before, after, "--regression-tolerance", "0.1")
    assert rc == 0 and summary["regression_tolerance"] == 0.1
    assert summary["regression_count"] == 1  # se sigue informando
    assert summary["regressions"][0]["blocking"] is False
    assert summary["blocking_regression_count"] == 0
    report = (tmp_path / "report.md").read_text("utf-8")
    assert "| c1 | 0.800 | 0.750 | -0.050 |" in report
    assert "1 de 1 regresión(es) dentro de la tolerancia" in report

    # Justo por encima de la tolerancia vuelve a bloquear
    rc, summary = _cli(tmp_path, before, after, "--regression-tolerance", "0.04")
    assert rc == 1 and summary["regressions"][0]["blocking"] is True


def test_tolerancia_no_perdona_regresiones_robustas(tmp_path):
    before = _results("base", 0.5, 0.5, case_scores={"c1": 0.80})
    after = _results("adapter", 0.5, 0.5, case_scores={"c1": 0.75})
    before["cases"][0].update(score_min=0.80, score_max=0.80)
    after["cases"][0].update(score_min=0.70, score_max=0.78)  # rangos disjuntos
    rc, summary = _cli(tmp_path, before, after, "--regression-tolerance", "1.0")
    assert summary["regressions"][0]["robust"] is True
    assert summary["regressions"][0]["blocking"] is True
    assert rc == 1


def test_tolerancia_negativa_se_rechaza(tmp_path):
    before = _results("base", 0.5, 0.5, case_scores={"c1": 0.8})
    with pytest.raises(SystemExit):
        _cli(tmp_path, before, before, "--regression-tolerance", "-0.1")
