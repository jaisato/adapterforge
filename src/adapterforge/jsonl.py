"""Utilidades JSONL con stdlib (sin dependencia de ``datasets`` en el núcleo).

El pipeline de datos trabaja siempre con JSON Lines: un objeto JSON por línea,
UTF-8, sin BOM. Estas funciones centralizan lectura/escritura con mensajes de
error útiles (número de línea incluido).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Itera los objetos de un fichero JSONL (ignora líneas en blanco)."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number}: JSON inválido: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{file_path}:{line_number}: se esperaba un objeto JSON, "
                    f"no {type(record).__name__}"
                )
            yield record


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Lee un fichero JSONL completo en memoria."""
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Escribe registros como JSONL (crea directorios) y devuelve el total."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            count += 1
    return count
