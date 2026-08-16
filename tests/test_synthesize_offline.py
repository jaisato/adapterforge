"""Tests de la síntesis offline v3: anclaje, validación, presupuesto y mezcla."""

from __future__ import annotations

import json
import re

import pytest

from adapterforge.data.budget import TokenCounter, assert_token_budget
from adapterforge.data.extract import RawDocument, split_articles
from adapterforge.data.synthesize import (
    DEFAULT_WINDOW_TOKENS,
    SFT_TASKS,
    SftExample,
    SynthContext,
    _amount_value,
    _first_sentence,
    _future_date,
    build_dpo_offline,
    build_sft_offline,
    classify_deadline,
    compose_window,
    find_deadlines,
    find_obligation_sentences,
    max_amount_in,
    offline_extraccion,
    parse_teacher_qa,
    rebalance,
)


def _make_doc(texto: str) -> RawDocument:
    return RawDocument(
        id="BOE-A-2026-99999",
        rango="Real Decreto",
        titulo="Real Decreto 999/2026, de prueba para tests unitarios",
        organo_emisor="Ministerio de Pruebas",
        fecha_publicacion="2026-06-01",
        materia=["pruebas"],
        url="https://boe.es/eli/9999",
        texto=texto + " " + "Texto de relleno normativo. " * 10,
    )

_CITA = re.compile(r"\bcita|\bcitando", re.IGNORECASE)


def test_offline_generates_examples_for_all_tasks(corpus):
    examples = build_sft_offline(corpus, seed=42)
    assert len(examples) >= 40, "el corpus de 12 docs debe producir >= 40 ejemplos"
    tasks_seen = {example.task for example in examples}
    assert tasks_seen == set(SFT_TASKS)
    doc_ids = {doc.id for doc in corpus}
    for example in examples:
        assert example.doc_id in doc_ids
        assert example.instruction.strip()
        assert example.response.strip()
        # La ventana empieza por la cabecera del documento (título primero).
        assert example.input.startswith(
            next(d for d in corpus if d.id == example.doc_id).titulo
        )
        assert example.source == "offline"


def test_offline_examples_are_anchored(corpus):
    """Todo valor objetivo debe aparecer literalmente en la ventana."""
    examples = build_sft_offline(corpus, seed=42)
    for example in examples:
        if example.task == "extraccion" and example.response.lstrip().startswith("{"):
            payload = json.loads(example.response)
            for value in payload.values():
                assert str(value) in example.input, (
                    f"{example.id}: valor JSON no anclado a la ventana"
                )
        # Si la instrucción pide cita, la respuesta cita el identificador.
        if _CITA.search(example.instruction):
            assert example.doc_id in example.response, example.id


def test_offline_no_placeholder_or_boilerplate_targets(corpus):
    examples = build_sft_offline(corpus, seed=42)
    for example in examples:
        assert "(no se detectan)" not in example.response
        assert "PREGUNTA:" not in example.response
        assert "RESPUESTA:" not in example.response


def test_offline_respects_token_budget(corpus):
    counter = TokenCounter()
    examples = build_sft_offline(corpus, seed=42, counter=counter, token_budget=400)
    assert examples
    for example in examples:
        # Margen pequeño: unir bloques puede variar la tokenización en ±unos tokens.
        assert counter.count(example.input) <= 440, example.id


def test_assert_token_budget_fails_loudly():
    counter = TokenCounter()
    oversized = SftExample(
        id="x", doc_id="d", task="qa", instruction="i",
        input="palabra " * 3000, response="r", source="offline",
    )
    with pytest.raises(ValueError, match="Presupuesto de tokens violado"):
        assert_token_budget([oversized], counter, seq_len=512, max_over_frac=0.0)


def test_offline_is_deterministic_and_respects_max_per_task(corpus):
    first = build_sft_offline(corpus, seed=42, max_per_task=5)
    second = build_sft_offline(corpus, seed=42, max_per_task=5)
    assert [e.id for e in first] == [e.id for e in second]
    per_task: dict[str, int] = {}
    for example in first:
        per_task[example.task] = per_task.get(example.task, 0) + 1
    assert all(count <= 5 for count in per_task.values())


def test_classify_deadline_validates_labels():
    assert classify_deadline("tres meses") == "duracion"
    assert classify_deadline("quince días hábiles") == "duracion"
    assert classify_deadline("31 de diciembre de 2026") == "fecha"
    # Capturas desbordadas o sin unidad temporal: fuera.
    assert classify_deadline("constitución del Consejo y la renovación") is None
    assert classify_deadline(
        "seis meses el Ministerio de Vivienda y Agenda Urbana desarrollará"
    ) is None


def test_structural_detectors_find_deadlines_and_obligations(corpus):
    doc = next(d for d in corpus if d.id == "BOE-A-2026-04812")
    deadlines = find_deadlines(doc.texto)
    assert "tres meses" in deadlines
    assert "seis meses" in deadlines
    obligations = find_obligation_sentences(doc.texto)
    assert any("deberán" in sentence for sentence in obligations)


def test_rebalance_moves_mix_towards_weights():
    def _fake(task: str, i: int) -> SftExample:
        return SftExample(
            id=f"{task}-{i}", doc_id="d", task=task,
            instruction="i", input="x", response="r", source="offline",
        )

    examples = (
        [_fake("extraccion", i) for i in range(100)]
        + [_fake("qa", i) for i in range(100)]
        + [_fake("resumen", i) for i in range(10)]
    )
    balanced = rebalance(examples, {"extraccion": 0.5, "qa": 0.3, "resumen": 0.2}, seed=42)
    counts: dict[str, int] = {}
    for example in balanced:
        counts[example.task] = counts.get(example.task, 0) + 1
    # N* = 10/0.2 = 50 -> objetivos 25/15/10.
    assert counts == {"extraccion": 25, "qa": 15, "resumen": 10}
    # Determinista.
    again = rebalance(examples, {"extraccion": 0.5, "qa": 0.3, "resumen": 0.2}, seed=42)
    assert [e.id for e in again] == [e.id for e in balanced]


def test_amount_parsing_handles_decimal_commas_and_tiers():
    assert _amount_value("30.000 euros") == 30000.0
    assert _amount_value("601,04 euros") == 601.04  # coma decimal, no millares
    assert max_amount_in("indemnización de 601,04 euros o multa de 30.000 euros") == (
        "30.000 euros"
    )
    tiers = (
        "Las leves, con multa de hasta 18.000 euros. Las graves, con multa de "
        "hasta 90.000 euros. Las muy graves, con multa de hasta 900.000 euros."
    )
    assert max_amount_in(tiers) == "900.000 euros"


def test_multa_label_is_the_window_maximum():
    texto = (
        "Artículo 1. Objeto. El presente real decreto regula el régimen "
        "sancionador aplicable a las entidades del sector.\n"
        "Artículo 2. Sanciones. Las infracciones leves se castigan con multa de "
        "hasta 18.000 euros según su gravedad concreta. Las infracciones muy "
        "graves se castigan con multa de hasta 900.000 euros en todo caso."
    )
    doc = _make_doc(texto)
    _, articles = split_articles(doc.texto)
    ctx = SynthContext(counter=TokenCounter(), token_budget=DEFAULT_WINDOW_TOKENS, seed=42)
    multa = [e for e in offline_extraccion(doc, articles, ctx) if "multa" in e.id]
    assert multa, "el régimen sancionador debe producir un ejemplo de multa"
    assert "900.000 euros" in multa[0].response
    assert "18.000 euros" not in multa[0].response.replace("900.000", "")


def test_first_sentence_skips_bare_apartado_numerals():
    body = "1. El presente real decreto tiene por objeto regular las ayudas."
    assert _first_sentence(body) == (
        "El presente real decreto tiene por objeto regular las ayudas."
    )


def test_future_date_filters_pre_publication_references():
    doc = _make_doc("Artículo 1. Objeto. Texto suficiente para validar el modelo.")
    assert _future_date("31 de diciembre de 2027", doc)
    assert not _future_date("31 de diciembre de 2024", doc)  # anterior a 2026
    assert not _future_date("31 de diciembre", doc)  # sin año: indecidible


def test_compose_window_uses_indices_not_article_numbers():
    # Dos «Artículo 1.»: el índice debe distinguirlos (números duplicados
    # son habituales en normas que modifican otras normas).
    texto = (
        "Artículo 1. Objeto primero. Texto del primer artículo original aquí.\n"
        "Artículo 1. Objeto duplicado. Texto del artículo uno de la norma "
        "modificada con contenido distinto."
    )
    doc = _make_doc(texto)
    _, articles = split_articles(doc.texto)
    assert len(articles) == 2
    counter = TokenCounter()
    window = compose_window(doc, articles, [1], counter, 300)
    assert "norma modificada con contenido distinto" in window
    assert "primer artículo original" not in window


def test_parse_teacher_qa_contract():
    parsed = parse_teacher_qa(
        "PREGUNTA: ¿Qué plazo fija el artículo 3?\n"
        "RESPUESTA: El plazo es de tres meses [BOE-A-2026-04812, art. 3]."
    )
    assert parsed is not None
    question, answer = parsed
    assert question.startswith("¿Qué plazo")
    assert answer.endswith("art. 3].")
    assert parse_teacher_qa("texto sin bloques") is None
    # Boilerplate anidado en la respuesta: rechazado.
    assert parse_teacher_qa("PREGUNTA: ¿a? RESPUESTA: PREGUNTA: otra") is None
    # Negritas markdown del teacher: aceptadas y limpiadas.
    bold = parse_teacher_qa(
        "**PREGUNTA:** ¿Qué plazo fija la norma?\n"
        "**RESPUESTA:** Tres meses [BOE-A-2026-04812, art. 2]."
    )
    assert bold is not None
    assert bold[0] == "¿Qué plazo fija la norma?"
    assert "**" not in bold[0] and "**" not in bold[1]


def test_dpo_pairs_offline(corpus):
    pairs = build_dpo_offline(corpus, seed=42, n_pairs=15)
    assert len(pairs) == 15
    for pair in pairs:
        assert pair.chosen != pair.rejected
        assert pair.doc_id in pair.input  # el input ancla siempre el identificador
    again = build_dpo_offline(corpus, seed=42, n_pairs=15)
    assert [p.id for p in again] == [p.id for p in pairs]
