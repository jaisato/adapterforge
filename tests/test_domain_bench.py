"""Tests del benchmark de dominio con predicciones conocidas (sin red)."""

from __future__ import annotations

import pytest

from adapterforge.eval.domain_bench import (
    aggregate,
    build_case_messages,
    load_bench_cases,
    run_bench,
    score_case,
)


def test_shipped_bench_is_valid(repo_root):
    cases = load_bench_cases(repo_root / "evals" / "domain_bench.jsonl")
    assert len(cases) == 25
    types = {case["task_type"] for case in cases}
    assert types == {"extract", "qa", "resumen"}
    extract_cases = [case for case in cases if case["task_type"] == "extract"]
    assert all(isinstance(case["expected"], dict) for case in extract_cases)


def test_shipped_bench_v2_is_valid_and_anchored(repo_root):
    from adapterforge.eval.metrics import normalize_text

    cases = load_bench_cases(repo_root / "evals" / "domain_bench_v2.jsonl")
    assert len(cases) >= 100, "el bench v2 debe tener 100+ casos para resolver deltas"
    # Incluye íntegro el bench original (comparabilidad histórica por caso).
    v1_ids = {c["id"] for c in load_bench_cases(repo_root / "evals" / "domain_bench.jsonl")}
    assert v1_ids <= {c["id"] for c in cases}
    for case in cases:
        if case["task_type"] != "extract":
            continue
        # Anclaje: todo valor esperado aparece en el input (métrica canónica).
        for field, value in case["expected"].items():
            assert normalize_text(str(value)) in normalize_text(case["input"]), (
                f"{case['id']}: campo {field} no anclado al input"
            )


def test_score_case_extract_qa_and_resumen():
    extract_case = {
        "id": "c1",
        "task_type": "extract",
        "instruction": "i",
        "input": "d",
        "expected": {"plazo": "tres meses"},
    }
    assert score_case(extract_case, '{"plazo": "tres meses"}')["score"] == 1.0
    assert score_case(extract_case, '{"plazo": "un año"}')["score"] == 0.0

    qa_case = {
        "id": "c2",
        "task_type": "qa",
        "instruction": "i",
        "input": "d",
        "expected": "antes del 30 de junio de 2027",
    }
    result = score_case(qa_case, "Deben disponer de ella antes del 30 de junio de 2027 [X].")
    assert result["score"] == 1.0
    assert result["contains"] is True

    resumen_case = {
        "id": "c3",
        "task_type": "resumen",
        "instruction": "i",
        "input": "d",
        "expected": "la norma regula el registro digital de jornada",
    }
    assert score_case(resumen_case, "la norma regula el registro digital de jornada")[
        "score"
    ] == pytest.approx(1.0)
    assert 0.0 < score_case(resumen_case, "la norma regula otra cosa distinta")["score"] < 1.0


def test_run_bench_aggregates_and_flags_missing(repo_root):
    cases = load_bench_cases(repo_root / "evals" / "domain_bench.jsonl")[:3]
    predictions = {
        cases[0]["id"]: '{"plazo": "tres meses"}',
        cases[1]["id"]: '{"plazo": "seis meses"}',
        # cases[2] sin predicción -> score 0 y aviso.
    }
    results = run_bench(cases, predictions, model_label="test")
    assert results["model"] == "test"
    assert results["missing_predictions"] == [cases[2]["id"]]
    assert results["summary"]["n_cases"] == 3
    assert 0.0 <= results["summary"]["overall"] <= 1.0


def test_aggregate_by_type():
    case_results = [
        {"id": "a", "task_type": "extract", "score": 1.0},
        {"id": "b", "task_type": "extract", "score": 0.0},
        {"id": "c", "task_type": "qa", "score": 0.5},
    ]
    summary = aggregate(case_results)
    assert summary["by_type"] == {"extract": 0.5, "qa": 0.5}
    assert summary["overall"] == 0.5
    assert aggregate([])["n_cases"] == 0


def test_score_case_strips_response_artifacts():
    case = {
        "id": "c1",
        "task_type": "extract",
        "instruction": "i",
        "input": "d",
        "expected": {"plazo": "tres meses"},
    }
    # Preámbulo basura real de los adapters con template thinking.
    garbage = '</tool_call>\n\n<tool_call>\n\n{"plazo": "tres meses"}'
    assert score_case(case, garbage)["score"] == 1.0


def test_run_bench_with_repeats_persists_predictions():
    case = {
        "id": "c1",
        "task_type": "extract",
        "instruction": "i",
        "input": "d",
        "expected": {"plazo": "tres meses"},
    }
    predictions = {"c1": ['{"plazo": "tres meses"}', '{"plazo": "un año"}',
                          '{"plazo": "tres meses"}']}
    results = run_bench([case], predictions, model_label="test")
    entry = results["cases"][0]
    assert entry["score"] == pytest.approx(2 / 3)
    assert entry["score_min"] == 0.0
    assert entry["score_max"] == 1.0
    assert len(entry["repeats"]) == 3
    assert entry["repeats"][1]["prediction"] == '{"plazo": "un año"}'
    assert entry["repeats"][1]["score"] == 0.0
    # Lista vacía cuenta como predicción ausente.
    empty = run_bench([case], {"c1": []}, model_label="test")
    assert empty["missing_predictions"] == ["c1"]


def test_run_bench_records_generation_metadata():
    case = {
        "id": "c1", "task_type": "qa", "instruction": "i", "input": "d", "expected": "resp"
    }
    generation = {"temperature": 0.0, "repeats": 3}
    results = run_bench([case], {"c1": "resp"}, model_label="m", generation=generation)
    assert results["generation"] == generation


def test_main_offline_honors_output_dir(repo_root, tmp_path):
    from adapterforge.eval.domain_bench import main

    predictions_file = tmp_path / "preds.jsonl"
    cases = load_bench_cases(repo_root / "evals" / "domain_bench.jsonl")
    predictions_file.write_text(
        "\n".join(
            f'{{"id": "{case["id"]}", "prediction": "x"}}' for case in cases
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "resultados"
    assert main([
        "--bench", str(repo_root / "evals" / "domain_bench.jsonl"),
        "--predictions", str(predictions_file),
        "--output-dir", str(out_dir),
    ]) == 0
    written = list(out_dir.glob("*.json"))
    assert len(written) == 1


def test_main_rejects_colliding_model_filenames(tmp_path):
    import pytest as _pytest

    from adapterforge.eval.domain_bench import main

    # 'a:b' y 'a-b' sanitizan al mismo archivo: debe fallar antes de generar.
    with _pytest.raises(SystemExit):
        main([
            "--model", "a:b", "--model", "a-b",
            "--api-base", "http://localhost:1",
            "--output-dir", str(tmp_path),
        ])


def test_build_case_messages_shape():
    case = {
        "id": "x", "task_type": "qa", "instruction": "¿Plazo?", "input": "Texto.", "expected": "p"
    }
    messages = build_case_messages(case)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "### Documento" in messages[1]["content"]
