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


_TIERED_SANCTIONS_TEXT = (
    "Artículo 1. Objeto. El presente real decreto regula el régimen "
    "sancionador aplicable a las entidades del sector.\n"
    "Artículo 2. Sanciones. Las infracciones leves se castigan con multa de "
    "hasta 18.000 euros según su gravedad concreta. Las infracciones muy "
    "graves se castigan con multa de hasta 900.000 euros en todo caso."
)


def test_tiered_sanctions_get_per_tier_labels():
    """v5: cada tramo pregunta y responde SU tope, con el distractor visible.

    v4 etiquetaba siempre el máximo de la ventana y el modelo aprendió a
    responder «muy graves» aunque se preguntara por las «graves» (eb-128).
    """
    doc = _make_doc(_TIERED_SANCTIONS_TEXT)
    _, articles = split_articles(doc.texto)
    ctx = SynthContext(counter=TokenCounter(), token_budget=DEFAULT_WINDOW_TOKENS, seed=42)
    multa = [e for e in offline_extraccion(doc, articles, ctx) if "multa" in e.id]
    by_suffix = {e.id.rsplit("extraccion-", 1)[1]: e for e in multa}
    assert "multa-leves" in by_suffix and "multa-muy-graves" in by_suffix
    leves = by_suffix["multa-leves"]
    assert "leves" in leves.instruction
    assert "18.000 euros" in leves.response
    assert "900.000" not in leves.response  # NUNCA el máximo global para un tramo
    muy_graves = by_suffix["multa-muy-graves"]
    assert "muy graves" in muy_graves.instruction
    assert "900.000 euros" in muy_graves.response
    # La pregunta global explícita responde el máximo absoluto.
    assert "multa-global" in by_suffix
    assert "900.000 euros" in by_suffix["multa-global"].response
    # Y ambos tramos son visibles en todas las ventanas (distractor presente).
    for example in multa:
        assert "18.000 euros" in example.input and "900.000 euros" in example.input


def test_find_sanction_tiers_splits_list_scales():
    """El patrón real de los escalados BOE: lista «a) ... b) ...» en una frase."""
    from adapterforge.data.synthesize import find_sanction_tiers

    texto = (
        "Artículo 1. Objeto. El presente real decreto regula el régimen "
        "sancionador de las entidades colaboradoras del sector regulado.\n"
        "Artículo 2. Sanciones. Las infracciones se sancionarán del siguiente "
        "modo: a) Las infracciones leves, con multa de hasta 10.000 euros. "
        "b) Las infracciones graves, con multa de hasta 100.000 euros. "
        "c) Las infracciones muy graves, con multa de hasta 1.000.000 euros."
    )
    doc = _make_doc(texto)
    _, articles = split_articles(doc.texto)
    tiers = {t.tier: t.amount for t in find_sanction_tiers(articles)}
    assert tiers == {
        "leves": "10.000 euros",
        "graves": "100.000 euros",
        "muy graves": "1.000.000 euros",
    }


def test_amount_regex_handles_boe_millions_form():
    # Forma BOE de los millones: «N de euros». La regex anterior se quedaba
    # con el límite INFERIOR del rango (hallazgo de la revisión v5).
    assert _amount_value("2.000.000 de euros") == 2000000.0
    assert max_amount_in(
        "se sancionarán con multa de 120.001 euros a 900.000 de euros"
    ) == "900.000 de euros"


def test_find_sanction_tiers_handles_long_list_scales():
    """Escalados >45 palabras: el tope de _sentence_hits ya no los descarta."""
    from adapterforge.data.synthesize import find_sanction_tiers

    texto = (
        "Artículo 1. Objeto. El presente real decreto regula el régimen "
        "sancionador aplicable a todas las entidades del sector regulado.\n"
        "Artículo 2. Sanciones. Las infracciones tipificadas en la presente "
        "disposición se sancionarán conforme al siguiente escalado de multas, "
        "atendiendo en todo caso a la gravedad de la conducta, a la "
        "intencionalidad apreciada y al beneficio ilícito obtenido por la "
        "entidad infractora en el ejercicio de su actividad: a) Las "
        "infracciones leves, con multa de hasta 10.000 euros. b) Las "
        "infracciones graves, con multa de hasta 100.000 euros. c) Las "
        "infracciones muy graves, con multa de hasta 2.000.000 de euros."
    )
    doc = _make_doc(texto)
    _, articles = split_articles(doc.texto)
    tiers = {t.tier: t.amount for t in find_sanction_tiers(articles)}
    assert tiers.get("leves") == "10.000 euros"
    assert tiers.get("graves") == "100.000 euros"
    assert tiers.get("muy graves") == "2.000.000 de euros"


def test_generic_multa_skipped_when_window_shows_larger_amount():
    """La etiqueta genérica solo vale si es el máximo VISIBLE en la ventana."""
    texto = (
        "Artículo 1. Objeto. El presente real decreto regula el régimen "
        "sancionador de referencia de las entidades del sector regulado.\n"
        "Artículo 2. Sanciones. El incumplimiento leve se castiga con multa "
        "de hasta 4.000 euros según su alcance. Las restantes infracciones "
        "tipificadas en esta disposición se sancionarán conforme al cuadro "
        "general aplicable al conjunto del sector y de sus entidades "
        "colaboradoras, con multas que en los supuestos de reincidencia "
        "acreditada podrán alcanzar, previa instrucción del oportuno "
        "procedimiento y ponderando la capacidad económica del infractor, "
        "la cifra de 6.000 euros por cada incumplimiento constatado."
    )
    doc = _make_doc(texto)
    _, articles = split_articles(doc.texto)
    ctx = SynthContext(counter=TokenCounter(), token_budget=DEFAULT_WINDOW_TOKENS, seed=42)
    multa = [e for e in offline_extraccion(doc, articles, ctx) if "multa" in e.id]
    # La frase larga (>45 palabras) con 6.000 euros no entra en las frases
    # sancionadoras cortas, pero SÍ en la ventana: etiquetar 4.000 mentiría.
    assert multa == []


def test_trim_purpose_rejects_embedded_plazos_and_dangling_words():
    from adapterforge.data.synthesize import _trim_purpose

    assert _trim_purpose("su informe en el plazo de un mes") is None
    assert _trim_purpose("desarrollar dicho plan y") is None  # anafórico
    trimmed = _trim_purpose("presentar las solicitudes de la ayuda y")
    assert trimmed == "presentar las solicitudes de la ayuda"


def test_find_sanction_tiers_skips_unattributable_sentences():
    from adapterforge.data.synthesize import find_sanction_tiers

    # Dos tramos en la MISMA frase: la cuantía no es atribuible por regex.
    texto = (
        "Artículo 1. Objeto. El presente real decreto regula el régimen "
        "sancionador de referencia para el sector de actividades reguladas.\n"
        "Artículo 2. Sanciones. Las infracciones leves y las infracciones "
        "graves se castigan con multa de hasta 18.000 euros o de hasta "
        "90.000 euros respectivamente según su gravedad."
    )
    doc = _make_doc(texto)
    _, articles = split_articles(doc.texto)
    assert find_sanction_tiers(articles) == []


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


def test_multiple_deadlines_require_disambiguated_questions():
    texto = (
        "Artículo 1. Objeto. El presente real decreto regula las ayudas del "
        "programa de impulso a la digitalización de las pequeñas empresas.\n"
        "Artículo 2. Solicitudes. Las solicitudes se presentarán en el plazo "
        "de tres meses para presentar las solicitudes de ayuda desde la "
        "publicación de la convocatoria oficial.\n"
        "Artículo 3. Justificación. Las beneficiarias dispondrán de un plazo "
        "de seis meses para justificar el empleo de los fondos recibidos."
    )
    doc = _make_doc(texto)
    _, articles = split_articles(doc.texto)
    ctx = SynthContext(counter=TokenCounter(), token_budget=DEFAULT_WINDOW_TOKENS, seed=42)
    plazo_examples = [
        e for e in offline_extraccion(doc, articles, ctx) if "plazo" in e.id
    ]
    assert plazo_examples, "dos plazos con propósito deben producir ejemplos"
    for example in plazo_examples:
        # Con varios plazos en el documento la pregunta NUNCA es genérica.
        assert "para" in example.instruction or "en relación con" in example.instruction
        # Y la etiqueta corresponde al plazo preguntado.
        if "solicitud" in example.instruction:
            assert "tres meses" in example.response
        if "justificar" in example.instruction:
            assert "seis meses" in example.response
        # v5: las respuestas en prosa ECO el propósito (liga pregunta-valor).
        if not example.response.lstrip().startswith("{"):
            assert "para" in example.response


def test_same_purpose_siblings_are_suppressed():
    """Dos plazos cuyo propósito derivado coincide -> pares contradictorios: fuera.

    Ocurre cuando ambos caen al fallback de rúbrica del mismo artículo (sin
    complemento «para ...» en sus frases).
    """
    texto = (
        "Artículo 1. Objeto. El presente real decreto regula el procedimiento "
        "aplicable a las entidades del sector para su registro oficial.\n"
        "Artículo 2. Tramitación registral. La inscripción se practicará en el "
        "plazo de tres meses desde la entrada en vigor. La renovación se "
        "practicará en el plazo de seis meses desde la primera inscripción."
    )
    doc = _make_doc(texto)
    _, articles = split_articles(doc.texto)
    ctx = SynthContext(counter=TokenCounter(), token_budget=DEFAULT_WINDOW_TOKENS, seed=42)
    plazo_examples = [
        e for e in offline_extraccion(doc, articles, ctx) if "plazo" in e.id
    ]
    # Ambos plazos derivarían «en relación con tramitación registral»: la
    # instrucción no discriminaría cuál se pide, así que no se emite ninguno.
    assert plazo_examples == []


def test_single_deadline_allows_generic_question():
    texto = (
        "Artículo 1. Objeto. El presente real decreto regula el registro de "
        "actividad de las entidades colaboradoras del sector energético.\n"
        "Artículo 2. Plazo. Las entidades deberán inscribirse en el plazo de "
        "tres meses desde la entrada en vigor de la presente disposición."
    )
    doc = _make_doc(texto)
    _, articles = split_articles(doc.texto)
    ctx = SynthContext(counter=TokenCounter(), token_budget=DEFAULT_WINDOW_TOKENS, seed=42)
    plazo_examples = [
        e for e in offline_extraccion(doc, articles, ctx) if "plazo" in e.id
    ]
    assert plazo_examples
    assert any("tres meses" in e.response for e in plazo_examples)


def test_resumen_prosa_uses_bench_reference_style(corpus):
    examples = [
        e for e in build_sft_offline(corpus, tasks=("resumen",), seed=42)
        if not e.response.startswith("###")  # solo las variantes en prosa
    ]
    assert examples
    for example in examples:
        # Nada de metadatos de ficha en la prosa: voz activa desde la norma.
        assert "es una disposición de rango" not in example.response
        # Sin cita cuando la instrucción no la pide.
        if "cita" not in example.instruction.lower():
            assert "Fuente:" not in example.response


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
