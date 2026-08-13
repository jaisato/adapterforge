"""Tests de shingles + Jaccard y deduplicación near-dup (implementación propia)."""

from __future__ import annotations

from adapterforge.data.filter import dedup_by_jaccard, jaccard, shingles


def test_shingles_counts_and_short_text():
    text = "uno dos tres cuatro cinco seis"
    result = shingles(text, k=5)
    # 6 tokens con k=5 -> 2 shingles deslizantes.
    assert len(result) == 2
    assert ("uno", "dos", "tres", "cuatro", "cinco") in result
    # Texto más corto que k: un único shingle con todos los tokens.
    short = shingles("uno dos", k=5)
    assert short == {("uno", "dos")}
    assert shingles("", k=5) == set()


def test_jaccard_identical_disjoint_and_empty():
    a = shingles("la ley regula el registro digital de jornada laboral", k=3)
    b = shingles("la ley regula el registro digital de jornada laboral", k=3)
    c = shingles("contenido completamente distinto sobre residuos textiles urbanos", k=3)
    assert jaccard(a, b) == 1.0
    assert jaccard(a, c) == 0.0
    assert jaccard(set(), set()) == 1.0
    assert jaccard(a, set()) == 0.0


def test_dedup_removes_near_duplicates_keeps_first():
    base = (
        "Las empresas deberán implantar un sistema de registro de jornada digital "
        "e interoperable antes del uno de octubre y conservar los registros durante "
        "cuatro años a disposición de la inspección de trabajo"
    )
    records = [
        {"id": "a", "instruction": "resume", "response": base},
        # Casi idéntico (cambia una palabra al final): debe caer.
        {"id": "b", "instruction": "resume", "response": base + " correspondiente"},
        # Distinto de verdad: debe sobrevivir.
        {
            "id": "c",
            "instruction": "resume",
            "response": (
                "La ayuda consistirá en doscientos cincuenta euros mensuales durante "
                "un máximo de veinticuatro meses para personas jóvenes arrendatarias "
                "de vivienda habitual con ingresos limitados"
            ),
        },
    ]
    kept, removed = dedup_by_jaccard(records, threshold=0.8)
    assert removed == 1
    assert [record["id"] for record in kept] == ["a", "c"]


def test_dedup_no_false_positives_on_distinct_records():
    records = [
        {"id": "x", "instruction": "i1", "response": "texto sobre subvenciones a pymes"},
        {"id": "y", "instruction": "i2", "response": "texto sobre ciberseguridad municipal"},
    ]
    kept, removed = dedup_by_jaccard(records, threshold=0.9)
    assert removed == 0
    assert len(kept) == 2
