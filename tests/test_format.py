"""Tests del formateo chat y de los splits deterministas con semilla."""

from __future__ import annotations

from adapterforge.data.format import (
    SYSTEM_PROMPT,
    split_records,
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
