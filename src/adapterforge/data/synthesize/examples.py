"""Modelos de los ejemplos sintetizados y parámetros compartidos.

- :class:`SftExample` y :class:`DpoPair`: registros previos al formateo chat
  (``adapterforge.data.format`` los convierte en ``messages``).
- :class:`SynthContext`: tokenizer, presupuesto de ventana y semilla que
  comparten los generadores offline.
- ``SFT_TASKS``, ``DEFAULT_BALANCE`` y ``DEFAULT_WINDOW_TOKENS``: tareas,
  mezcla objetivo y presupuesto por defecto.
- RNG deterministas por documento (:func:`_variant`) y por ejemplo
  (:func:`_example_rng`): la reproducibilidad del dataset depende de ellos.
"""

from __future__ import annotations

import random
import zlib
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from adapterforge.data.budget import TokenCounter

SFT_TASKS = ("resumen", "qa", "obligaciones", "llano", "extraccion")

# Mezcla objetivo por tarea, alineada con el domain_bench (40 % extract,
# 32 % qa, 28 % resumen) más las tareas de estilo del producto.
DEFAULT_BALANCE = {
    "extraccion": 0.30,
    "qa": 0.25,
    "resumen": 0.20,
    "llano": 0.15,
    "obligaciones": 0.10,
}

# Presupuesto de tokens de la ventana de documento (cabecera + artículos).
# Con sistema (~60) + instrucción (~40) + respuesta (~250-500) + plantilla,
# deja el render completo holgadamente bajo seq_len 2048.
DEFAULT_WINDOW_TOKENS = 1100


def _variant(doc_id: str, n: int, salt: str = "") -> int:
    """Índice de variante determinista por documento (reproducible)."""
    return zlib.crc32(f"{doc_id}:{salt}".encode()) % n


def _example_rng(seed: int, *keys: str) -> random.Random:
    """RNG determinista por ejemplo (no por documento): rompe el monocultivo."""
    return random.Random(f"{seed}|" + "|".join(keys))


class SftExample(BaseModel):
    """Ejemplo instruccional previo al formateo chat."""

    model_config = ConfigDict(extra="forbid")

    id: str
    doc_id: str
    task: str
    instruction: str
    input: str
    response: str
    source: str = Field(description="'offline' o nombre del modelo teacher")


class DpoPair(BaseModel):
    """Par de preferencia (chosen/rejected) previo al formateo chat."""

    model_config = ConfigDict(extra="forbid")

    id: str
    doc_id: str
    task: str
    instruction: str
    input: str
    chosen: str
    rejected: str
    source: str = "offline"


class SynthContext(NamedTuple):
    """Parámetros compartidos de la síntesis offline."""

    counter: TokenCounter
    token_budget: int
    seed: int
