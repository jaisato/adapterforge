"""Tests de la síntesis offline: N ejemplos válidos, anclados y deterministas."""

from __future__ import annotations

import json

from adapterforge.data.synthesize import (
    SFT_TASKS,
    build_dpo_offline,
    build_sft_offline,
    find_deadlines,
    find_obligation_sentences,
)


def test_offline_generates_examples_for_all_tasks(corpus):
    examples = build_sft_offline(corpus, seed=42)
    assert len(examples) >= 40, "el corpus de 12 docs debe producir >= 40 ejemplos"
    tasks_seen = {example.task for example in examples}
    assert tasks_seen == set(SFT_TASKS)
    doc_ids = {doc.id for doc in corpus}
    for example in examples:
        assert example.doc_id in doc_ids
        assert example.instruction.strip()
        assert len(example.response) >= 60
        assert example.input.startswith(
            next(d for d in corpus if d.id == example.doc_id).titulo
        )
        assert example.source == "offline"


def test_offline_examples_are_anchored(corpus):
    examples = build_sft_offline(corpus, tasks=("resumen", "qa", "obligaciones"), seed=42)
    for example in examples:
        # Toda respuesta cita el identificador oficial del documento.
        assert example.doc_id in example.response
    json_examples = [example for example in examples if example.task == "obligaciones"]
    assert json_examples
    for example in json_examples:
        payload = json.loads(example.response)
        assert set(payload) == {"obligaciones", "plazos", "sanciones", "fuente"}
        assert payload["fuente"] == example.doc_id
        assert payload["plazos"], "el corpus de muestra siempre contiene plazos"


def test_offline_is_deterministic_and_respects_max_per_task(corpus):
    first = build_sft_offline(corpus, seed=42, max_per_task=5)
    second = build_sft_offline(corpus, seed=42, max_per_task=5)
    assert [e.id for e in first] == [e.id for e in second]
    per_task: dict[str, int] = {}
    for example in first:
        per_task[example.task] = per_task.get(example.task, 0) + 1
    assert all(count <= 5 for count in per_task.values())


def test_structural_detectors_find_deadlines_and_obligations(corpus):
    doc = next(d for d in corpus if d.id == "BOE-A-2026-04812")
    deadlines = find_deadlines(doc.texto)
    assert "tres meses" in deadlines
    assert "seis meses" in deadlines
    obligations = find_obligation_sentences(doc.texto)
    assert any("deberán" in sentence for sentence in obligations)


def test_dpo_pairs_offline(corpus):
    pairs = build_dpo_offline(corpus, seed=42, n_pairs=15)
    assert len(pairs) == 15
    for pair in pairs:
        assert pair.chosen != pair.rejected
        assert len(pair.chosen) > len(pair.rejected) * 0.3
        assert pair.doc_id in pair.chosen  # la respuesta buena cita la fuente
    # Determinismo con la misma semilla.
    again = build_dpo_offline(corpus, seed=42, n_pairs=15)
    assert [p.id for p in again] == [p.id for p in pairs]
