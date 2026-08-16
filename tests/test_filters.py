"""Tests de los filtros heurísticos (longitud, idioma, ratio de copia)."""

from __future__ import annotations

from adapterforge.data.filter import (
    apply_heuristics,
    copy_ratio,
    dedup_by_jaccard,
    is_spanish_basic,
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
