"""Formateo a mensajes de chat y splits reproducibles 90/5/5.

Convierte los ejemplos filtrados al formato conversacional estándar
(``{"messages": [system, user, assistant]}``) que TRL consume directamente
(aplica el chat template del modelo objetivo en entrenamiento), y genera
splits train/val/test deterministas con semilla.

Uso CLI::

    # Splits 90/5/5
    python -m adapterforge.data.format --input data/interim/sft_filtered.jsonl \
        --output-dir data/processed --seed 42

    # Un único fichero (p. ej. el sample versionado para el smoke-train)
    python -m adapterforge.data.format --input ... \
        --single-output data/processed/sample_sft.jsonl

    # Pares DPO
    python -m adapterforge.data.format --input pares.jsonl --dpo \
        --single-output data/processed/sample_dpo.jsonl

TODO(jairo): packing de secuencias y curriculum (ordenar por dificultad) como
opciones de formateo avanzado.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from adapterforge.jsonl import read_jsonl, write_jsonl

# Prompt de sistema del adapter JurisTune (idéntico en train, eval y serving).
SYSTEM_PROMPT = (
    "Eres JurisTune, un asistente especializado en normativa española y de la "
    "UE. Respondes en español, con rigor y estructura, anclándote al documento "
    "aportado y citando la fuente con el formato [ID, art. N]. Si el documento "
    "no contiene la respuesta, lo dices explícitamente."
)

DEFAULT_RATIOS = (0.90, 0.05, 0.05)


def build_user_content(record: dict[str, Any]) -> str:
    """Compone el mensaje de usuario: instrucción + documento anclado."""
    instruction = str(record["instruction"]).strip()
    doc_input = str(record.get("input", "")).strip()
    if doc_input:
        return f"{instruction}\n\n### Documento\n{doc_input}"
    return instruction


def to_chat_record(record: dict[str, Any]) -> dict[str, Any]:
    """Ejemplo SFT -> formato conversacional ``messages``."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(record)},
            {"role": "assistant", "content": str(record["response"]).strip()},
        ],
        "meta": {
            "id": record.get("id"),
            "doc_id": record.get("doc_id"),
            "task": record.get("task"),
            "source": record.get("source"),
        },
    }


def to_dpo_record(record: dict[str, Any]) -> dict[str, Any]:
    """Par de preferencia -> formato conversacional prompt/chosen/rejected."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(record)},
        ],
        "chosen": [{"role": "assistant", "content": str(record["chosen"]).strip()}],
        "rejected": [{"role": "assistant", "content": str(record["rejected"]).strip()}],
        "meta": {
            "id": record.get("id"),
            "doc_id": record.get("doc_id"),
            "task": record.get("task"),
        },
    }


def split_records(
    records: list[dict[str, Any]],
    seed: int = 42,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[str, list[dict[str, Any]]]:
    """Baraja con semilla y reparte en train/val/test.

    Determinista: la misma lista y semilla producen exactamente los mismos
    splits. val y test reciben al menos 1 ejemplo si hay suficientes.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Los ratios deben sumar 1.0 (recibido {ratios})")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    n_val = max(1, round(total * ratios[1])) if total >= 3 else 0
    n_test = max(1, round(total * ratios[2])) if total >= 3 else 0
    n_train = total - n_val - n_test
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Formato chat + splits del dataset")
    parser.add_argument("--input", required=True, help="Ejemplos filtrados JSONL")
    parser.add_argument("--output-dir", default=None, help="Directorio para splits 90/5/5")
    parser.add_argument("--single-output", default=None, help="Todo en un único JSONL")
    parser.add_argument("--prefix", default="sft", help="Prefijo de los ficheros de split")
    parser.add_argument("--dpo", action="store_true", help="Formatear pares de preferencia")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.output_dir and not args.single_output:
        parser.error("indica --output-dir y/o --single-output")

    records = read_jsonl(args.input)
    converter = to_dpo_record if args.dpo else to_chat_record
    formatted = [converter(record) for record in records]

    if args.single_output:
        count = write_jsonl(args.single_output, formatted)
        print(f"[format] {count} registros -> {args.single_output}")

    if args.output_dir:
        splits = split_records(formatted, seed=args.seed)
        output_dir = Path(args.output_dir)
        for split_name, split_records_ in splits.items():
            path = output_dir / f"{args.prefix}_{split_name}.jsonl"
            write_jsonl(path, split_records_)
            print(f"[format] split {split_name}: {len(split_records_)} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
