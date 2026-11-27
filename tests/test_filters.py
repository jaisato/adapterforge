"""Tests de los filtros heurísticos (longitud, idioma, ratio de copia)."""

from __future__ import annotations

import pytest

from adapterforge.data.filter import (
    apply_heuristics,
    apply_judge,
    copy_ratio,
    dedup_by_jaccard,
    is_spanish_basic,
    judge_score,
    stopword_ratio,
)

SPANISH = (
    "La presente orden tiene por objeto regular las obligaciones de las empresas "
    "en materia de registro de la jornada y de conservación de los datos."
)
ENGLISH = (
    "The quick brown fox jumps over the lazy dog while the committee reviews "
    "quarterly compliance reports without any Spanish words at all."
)


def test_language_detector_accepts_spanish_rejects_english():
    assert is_spanish_basic(SPANISH)
    assert not is_spanish_basic(ENGLISH)
    assert stopword_ratio(SPANISH) > stopword_ratio(ENGLISH)


def test_copy_ratio_detects_verbatim_and_allows_original():
    source = (
        "Las empresas deberán implantar un sistema de registro de jornada digital "
        "e interoperable antes del uno de octubre de dos mil veintiséis y los "
        "registros deberán conservarse durante cuatro años a disposición de la "
        "inspección de trabajo y de la seguridad social en todo momento."
    )
    verbatim = source  # copia literal completa
    original = (
        "Resumen propio: la norma obliga a registrar la jornada con medios "
        "digitales, fija una fecha límite y un periodo de conservación."
    )
    assert copy_ratio(verbatim, source) > 0.9
    assert copy_ratio(original, source) < 0.3
    # Una respuesta corta de extracción que cita verbatim NO es una copia:
    # con menos palabras que el tamaño de shingle no hay «run largo» posible.
    assert copy_ratio("antes del uno de octubre", source) == 0.0


def test_dedup_keeps_same_short_answer_across_documents():
    base = {"instruction": "Extrae el plazo en un objeto JSON",
            "response": '{"plazo": "tres meses"}'}
    records = [
        dict(base, id="a", doc_id="BOE-A-2026-0001"),
        dict(base, id="b", doc_id="BOE-A-2026-0002"),  # otro documento: legítimo
        dict(base, id="c", doc_id="BOE-A-2026-0001"),  # mismo doc: duplicado real
    ]
    kept, removed = dedup_by_jaccard(records)
    assert [record["id"] for record in kept] == ["a", "b"]
    assert removed == 1


def test_apply_heuristics_filters_by_reason(corpus):
    doc = corpus[0]
    docs_text = {doc.id: doc.texto}
    valid_response = (
        "La norma regula las ayudas del programa y establece que las solicitudes "
        "se presentan en un plazo de tres meses desde la convocatoria oficial."
    )
    instruction = "Resume la norma"
    records = [
        {"id": "ok", "doc_id": doc.id, "instruction": instruction, "response": valid_response},
        {"id": "short", "doc_id": doc.id, "instruction": instruction, "response": "Muy corto."},
        {"id": "eng", "doc_id": doc.id, "instruction": "Resume la norma", "response": (
            "This answer is written entirely in English and should therefore be "
            "rejected by the basic language filter of the pipeline system."
        )},
        {"id": "copy", "doc_id": doc.id, "instruction": "Resume la norma", "response": doc.texto},
        {"id": "noinstr", "doc_id": doc.id, "instruction": "x", "response": valid_response},
    ]
    kept, reasons = apply_heuristics(records, docs_text)
    assert [record["id"] for record in kept] == ["ok"]
    assert reasons["too_short"] == 1
    assert reasons["not_spanish"] == 1
    assert reasons["copy"] == 1
    assert reasons["bad_instruction"] == 1


def test_apply_heuristics_task_minimums_boilerplate_and_input_gate():
    valid_response = (
        "La norma regula las ayudas del programa y establece que las solicitudes "
        "se presentan en un plazo de tres meses desde la convocatoria oficial."
    )
    records = [
        # JSON corto de extracción: legítimo (mínimo por tarea + exención idioma).
        {"id": "json", "doc_id": "", "task": "extraccion",
         "instruction": "Extrae el plazo en un objeto JSON",
         "response": '{"plazo": "tres meses"}'},
        # Boilerplate del teacher sin parsear: fuera.
        {"id": "boiler", "doc_id": "", "task": "qa",
         "instruction": "Pregunta sobre el artículo tres de la norma",
         "response": "PREGUNTA: ¿Qué establece la norma sobre los plazos de la "
                     "convocatoria? RESPUESTA: Establece un plazo de tres meses."},
        # No-respuesta «(no se detectan)»: fuera.
        {"id": "nodetect", "doc_id": "", "task": "obligaciones",
         "instruction": "Enumera las obligaciones de la norma",
         "response": "Obligaciones:\n- (no se detectan)\n\nPlazos:\n- de la que en "
                     "un máximo de tres meses según la convocatoria oficial."},
        # Input desbocado (la ventana debería haberlo acotado): fuera.
        {"id": "biginput", "doc_id": "", "task": "resumen",
         "instruction": "Resume la disposición siguiente",
         "input": "x" * 20000, "response": valid_response},
    ]
    kept, reasons = apply_heuristics(records, {})
    assert [record["id"] for record in kept] == ["json"]
    assert reasons["boilerplate"] == 2
    assert reasons["input_too_long"] == 1


def test_apply_heuristics_length_bounds():
    records = [
        {"id": "long", "doc_id": "", "instruction": "Resume la disposición", "response": (
            "palabra de la que en " * 900
        )},
    ]
    kept, reasons = apply_heuristics(records, {})
    assert not kept
    assert reasons["too_long"] == 1


# ---------------------------------------------------------------------------
# LLM-judge (cliente simulado, sin red)
# ---------------------------------------------------------------------------

_JUDGED = {"id": "x", "instruction": "Resume la norma", "input": "doc", "response": "resp"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4", 4),
        ("**3**", 3),
        ("Nota: 5/5", 5),
        ("La nota es 2.", 2),
        # El razonamiento no cuenta: antes puntuaba 1 (el primer dígito 1-5).
        ("<think>Comparo con los niveles 1, 2 y 3</think>\n4", 4),
    ],
)
def test_judge_score_reads_the_grade(monkeypatch, raw, expected):
    import adapterforge.llm as llm

    monkeypatch.setattr(llm, "chat_complete", lambda *args, **kwargs: raw)
    assert judge_score(_JUDGED, model="stub/judge") == expected


@pytest.mark.parametrize(
    "raw",
    [
        "10/10",  # fuera de escala: antes puntuaba 1 (el «1» de «10»)
        "0",
        "4.5",
        # Una nota decimal no se salta para leer la escala: antes puntuaban 5.
        "2.5/5",
        "1,5/5",
        "Nota: 2.5 de 5",
        "0.5/5",
        ".5/5",
        "Excelente",
        "<think>Reviso el formato y la fidelidad",  # razonamiento truncado
    ],
)
def test_judge_score_rejects_unreadable_grades(monkeypatch, raw):
    import adapterforge.llm as llm

    monkeypatch.setattr(llm, "chat_complete", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="nota 1-5"):
        judge_score(_JUDGED, model="stub/judge")


def test_apply_judge_survives_per_record_failures(monkeypatch):
    import adapterforge.llm as llm

    answers = iter([llm.LlmError("timeout simulado"), "4", "Excelente", "2"])

    def fake_chat_complete(*args, **kwargs):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(llm, "chat_complete", fake_chat_complete)
    records = [dict(_JUDGED, id=f"r{i}") for i in range(4)]
    kept, removed, errors = apply_judge(records, model="stub/judge", min_score=3)
    assert [record["id"] for record in kept] == ["r1"]
    assert kept[0]["judge_score"] == 4
    assert removed == 1  # r3: nota 2 < 3
    assert errors == 2  # r0: fallo de transporte; r2: nota ilegible


def test_main_reports_judge_errors(tmp_path, monkeypatch):
    import json

    import adapterforge.llm as llm
    from adapterforge.data import filter as filter_module

    response = (
        "La norma regula las ayudas del programa y establece que las solicitudes "
        "se presentan en un plazo de tres meses desde la convocatoria oficial."
    )
    records = [
        {"id": f"r{i}", "doc_id": "", "task": "resumen",
         "instruction": f"Resume la disposición número {i} con detalle",
         "response": f"{response} Caso {word}."}
        for i, word in enumerate(("uno", "dos", "tres"))
    ]
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("", encoding="utf-8")
    answers = iter(["5", "<think>sin cerrar", "4"])
    monkeypatch.setattr(llm, "chat_complete", lambda *args, **kwargs: next(answers))

    report_path = tmp_path / "report.json"
    assert filter_module.main([
        "--input", str(input_path), "--corpus", str(corpus_path),
        "--output", str(tmp_path / "out.jsonl"), "--report", str(report_path),
        "--judge", "--judge-model", "stub/judge", "--dedup-threshold", "1.01",
    ]) == 0
    stats = json.loads(report_path.read_text(encoding="utf-8"))
    assert stats["kept"] == 2
    assert stats["removed_judge"] == 0
    assert stats["removed_judge_errors"] == 1
