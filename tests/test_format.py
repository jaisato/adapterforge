"""Tests del formateo chat y de los splits deterministas con semilla."""

from __future__ import annotations

from adapterforge.data.format import (
    NEUTRAL_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    split_records,
    split_records_by_doc,
    to_chat_record,
    to_dpo_record,
)


def _make_records(n: int) -> list[dict]:
    return [{"id": f"r{i}", "value": i} for i in range(n)]


def test_to_chat_record_structure():
    record = {
        "id": "x1",
        "doc_id": "BOE-A-2026-04812",
        "task": "resumen",
        "instruction": "Resume la disposición.",
        "input": "Texto del documento.",
        "response": "Resumen estructurado.",
        "source": "offline",
    }
    chat = to_chat_record(record)
    roles = [message["role"] for message in chat["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert chat["messages"][0]["content"] == SYSTEM_PROMPT
    assert "### Documento" in chat["messages"][1]["content"]
    assert chat["messages"][1]["content"].startswith("Resume la disposición.")
    assert chat["messages"][2]["content"] == "Resumen estructurado."
    assert chat["meta"]["doc_id"] == "BOE-A-2026-04812"


def test_to_dpo_record_structure():
    record = {
        "id": "p1",
        "doc_id": "BOE-A-2026-05733",
        "task": "resumen",
        "instruction": "Resume.",
        "input": "Documento.",
        "chosen": "Respuesta buena.",
        "rejected": "Respuesta vaga.",
    }
    pair = to_dpo_record(record)
    assert [m["role"] for m in pair["prompt"]] == ["system", "user"]
    assert pair["chosen"][0]["role"] == "assistant"
    assert pair["rejected"][0]["content"] == "Respuesta vaga."
    assert pair["chosen"][0]["content"] != pair["rejected"][0]["content"]


def test_general_records_get_neutral_system_prompt():
    general = {
        "id": "g1",
        "task": "general",
        "instruction": "Nombra cinco animales con rayas.",
        "response": "Cebra, tigre...",
    }
    chat = to_chat_record(general)
    assert chat["messages"][0]["content"] == NEUTRAL_SYSTEM_PROMPT
    anchored = dict(general, doc_id="BOE-A-2026-1", input="Texto del documento.")
    assert to_chat_record(anchored)["messages"][0]["content"] == SYSTEM_PROMPT


def test_split_by_doc_never_leaks_documents():
    records = [
        {"id": f"doc{d}-e{i}", "meta": {"doc_id": f"doc{d}"}}
        for d in range(30)
        for i in range(4)
    ]
    splits = split_records_by_doc(records, seed=42)
    docs_per_split = {
        name: {r["meta"]["doc_id"] for r in part} for name, part in splits.items()
    }
    assert docs_per_split["train"] & docs_per_split["val"] == set()
    assert docs_per_split["train"] & docs_per_split["test"] == set()
    assert docs_per_split["val"] & docs_per_split["test"] == set()
    assert docs_per_split["val"] and docs_per_split["test"]
    all_ids = [r["id"] for part in splits.values() for r in part]
    assert sorted(all_ids) == sorted(r["id"] for r in records)
    # Determinista y sensible a la semilla.
    assert split_records_by_doc(records, seed=42) == splits
    assert split_records_by_doc(records, seed=7) != splits


def test_split_by_doc_handles_general_records():
    anchored = [
        {"id": f"doc{d}-e{i}", "meta": {"doc_id": f"doc{d}"}}
        for d in range(20)
        for i in range(3)
    ]
    general = [{"id": f"gen{i}", "meta": {"doc_id": None}} for i in range(20)]
    splits = split_records_by_doc(anchored + general, seed=42)
    all_ids = [r["id"] for part in splits.values() for r in part]
    assert sorted(all_ids) == sorted(r["id"] for r in anchored + general)


def test_split_deterministic_same_seed():
    records = _make_records(40)
    first = split_records(records, seed=42)
    second = split_records(records, seed=42)
    assert first == second
    different = split_records(records, seed=7)
    assert different != first  # otra semilla, otro reparto


def test_split_ratios_and_partition():
    records = _make_records(100)
    splits = split_records(records, seed=42)
    assert len(splits["train"]) == 90
    assert len(splits["val"]) == 5
    assert len(splits["test"]) == 5
    # Partición exacta: sin pérdidas ni duplicados entre splits.
    all_ids = [r["id"] for part in splits.values() for r in part]
    assert sorted(all_ids) == sorted(r["id"] for r in records)
    assert len(set(all_ids)) == 100
