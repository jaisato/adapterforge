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
import math
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

# Prompt neutro para ejemplos SIN documento anclado (p. ej. instrucciones
# generales): entrenarlos bajo el contrato de grounding de JurisTune enseña
# al modelo que ese contrato puede ignorarse.
NEUTRAL_SYSTEM_PROMPT = "Eres un asistente útil que responde en español con claridad y rigor."

DEFAULT_RATIOS = (0.90, 0.05, 0.05)


def build_user_content(record: dict[str, Any]) -> str:
    """Compone el mensaje de usuario: instrucción + documento anclado."""
    instruction = str(record["instruction"]).strip()
    doc_input = str(record.get("input", "")).strip()
    if doc_input:
        return f"{instruction}\n\n### Documento\n{doc_input}"
    return instruction


def to_chat_record(record: dict[str, Any]) -> dict[str, Any]:
    """Ejemplo SFT -> formato conversacional ``messages``.

    Los ejemplos anclados a documento llevan el system prompt de JurisTune;
    los generales (sin ``doc_id``/``input``) llevan el prompt neutro para no
    contradecir el contrato de grounding.
    """
    anchored = bool(record.get("doc_id")) and bool(str(record.get("input", "")).strip())
    system = SYSTEM_PROMPT if anchored else NEUTRAL_SYSTEM_PROMPT
    return {
        "messages": [
            {"role": "system", "content": system},
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


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if (
        len(ratios) != 3
        or any(not math.isfinite(ratio) or ratio < 0 for ratio in ratios)
        or abs(sum(ratios) - 1.0) > 1e-6
    ):
        raise ValueError(
            f"Los ratios deben ser tres valores finitos no negativos que sumen 1: {ratios}"
        )


def split_records(
    records: list[dict[str, Any]],
    seed: int = 42,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[str, list[dict[str, Any]]]:
    """Baraja con semilla y reparte en train/val/test A NIVEL DE EJEMPLO.

    OJO: con varios ejemplos por documento esto FUGA documentos entre splits
    (la val-loss deja de detectar sobreajuste). Para datasets anclados usa
    :func:`split_records_by_doc`. Se mantiene para ejemplos sin doc_id.
    """
    _validate_ratios(ratios)
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    minimum = 1 if total >= 3 else 0
    n_val = min(total, max(minimum, round(total * ratios[1]))) if ratios[1] else 0
    n_test = (
        min(total - n_val, max(minimum, round(total * ratios[2]))) if ratios[2] else 0
    )
    n_train = total - n_val - n_test
    if not ratios[0] and n_train:
        if ratios[1] >= ratios[2]:
            n_val += n_train
        else:
            n_test += n_train
        n_train = 0
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _record_doc_id(record: dict[str, Any]) -> str | None:
    meta = record.get("meta") or {}
    doc_id = meta.get("doc_id") or record.get("doc_id")
    return str(doc_id) if doc_id else None


def split_records_by_doc(
    records: list[dict[str, Any]],
    seed: int = 42,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[str, list[dict[str, Any]]]:
    """Split agrupado por documento: ningún doc_id aparece en dos splits.

    Baraja los documentos con semilla y los asigna con avidez a val y test
    hasta cubrir sus cuotas de EJEMPLOS; el resto va a train. Los registros
    sin doc_id (ejemplos generales) se reparten aparte a nivel de ejemplo.
    """
    _validate_ratios(ratios)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    general: list[dict[str, Any]] = []
    for record in records:
        doc_id = _record_doc_id(record)
        if doc_id is None:
            general.append(record)
        else:
            by_doc.setdefault(doc_id, []).append(record)

    doc_ids = sorted(by_doc)
    random.Random(seed).shuffle(doc_ids)
    total_anchored = sum(len(v) for v in by_doc.values())
    quota_val = round(total_anchored * ratios[1])
    quota_test = round(total_anchored * ratios[2])
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    filled_val = filled_test = 0
    for doc_id in doc_ids:
        bucket = by_doc[doc_id]
        if filled_val < quota_val:
            splits["val"].extend(bucket)
            filled_val += len(bucket)
        elif filled_test < quota_test:
            splits["test"].extend(bucket)
            filled_test += len(bucket)
        else:
            fallback = "train" if ratios[0] else ("val" if ratios[1] >= ratios[2] else "test")
            splits[fallback].extend(bucket)

    if general:
        general_splits = split_records(general, seed=seed, ratios=ratios)
        for name in splits:
            splits[name].extend(general_splits[name])
    for offset, name in enumerate(("train", "val", "test")):
        random.Random(seed + offset).shuffle(splits[name])
    return splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Formato chat + splits del dataset")
    parser.add_argument("--input", required=True, help="Ejemplos filtrados JSONL")
    parser.add_argument("--output-dir", default=None, help="Directorio para splits 90/5/5")
    parser.add_argument("--single-output", default=None, help="Todo en un único JSONL")
    parser.add_argument("--prefix", default="sft", help="Prefijo de los ficheros de split")
    parser.add_argument("--dpo", action="store_true", help="Formatear pares de preferencia")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-by",
        choices=["doc", "example"],
        default="doc",
        help="'doc' (defecto): ningún documento aparece en dos splits",
    )
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
        splitter = split_records_by_doc if args.split_by == "doc" else split_records
        splits = splitter(formatted, seed=args.seed)
        output_dir = Path(args.output_dir)
        for split_name, split_records_ in splits.items():
            path = output_dir / f"{args.prefix}_{split_name}.jsonl"
            write_jsonl(path, split_records_)
            print(f"[format] split {split_name}: {len(split_records_)} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
