"""Presupuesto de tokens del dataset, medido con el tokenizer real.

La lección de v1/v2/v2b: el presupuesto de longitud debe imponerse al
CONSTRUIR el dataset (con el tokenizer del modelo objetivo), no confiarse al
truncado silencioso del trainer ni a un filtro que descarta ejemplos con
sesgo. Este módulo ofrece:

- :class:`TokenCounter`: cuenta tokens con el ``tokenizer.json`` del modelo
  si está disponible (``tokenizers`` es dependencia ligera, sin torch) y cae
  a una estimación conservadora de ``chars/3`` si no lo está (el ratio real
  medido en el corpus BOE es ~3.7 chars/token, mínimo 2.69).
- :func:`budget_report`: fracción de ejemplos cuyo render completo
  (sistema + usuario + respuesta + plantilla de chat) excedería ``seq_len``.
  El build falla si supera ``max_over_frac`` — nunca más un dataset donde el
  98 % de las respuestas desaparecen por truncado.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# Rutas candidatas al tokenizer del modelo objetivo (la más reciente primero).
DEFAULT_TOKENIZER_CANDIDATES = (
    "artifacts/qlora_qwen3-4b-v2b/adapter/tokenizer.json",
    "artifacts/qlora_qwen3-4b-v2/adapter/tokenizer.json",
    "artifacts/qlora_qwen3-4b/adapter/tokenizer.json",
)

# Chars/token de reserva cuando no hay tokenizer: conservador frente al ~3.7
# medido (sobreestimar tokens = ventanas más cortas, nunca truncado).
FALLBACK_CHARS_PER_TOKEN = 3.0

# Tokens que añade el render del chat template (im_start/im_end, roles...)
# más margen; medido sobre el template oficial de Qwen3-Instruct-2507.
CHAT_TEMPLATE_OVERHEAD_TOKENS = 40


class TokenCounter:
    """Contador de tokens con tokenizer real y fallback por caracteres."""

    def __init__(self, tokenizer_path: str | Path | None = None) -> None:
        self.tokenizer = None
        self.tokenizer_path: str | None = None
        candidates = (
            [Path(tokenizer_path)] if tokenizer_path else
            [Path(p) for p in DEFAULT_TOKENIZER_CANDIDATES]
        )
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                from tokenizers import Tokenizer  # dependencia ligera, opcional

                self.tokenizer = Tokenizer.from_file(str(candidate))
                self.tokenizer_path = str(candidate)
                break
            except Exception:  # noqa: BLE001 — cualquier fallo cae al estimador
                continue

    def count(self, text: str) -> int:
        """Número de tokens del texto (real o estimación conservadora)."""
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text).ids)
        return math.ceil(len(text) / FALLBACK_CHARS_PER_TOKEN)

    def describe(self) -> str:
        if self.tokenizer is not None:
            return f"tokenizer real ({self.tokenizer_path})"
        return f"estimación chars/{FALLBACK_CHARS_PER_TOKEN:g} (sin tokenizer)"


def example_total_tokens(
    example: Any, counter: TokenCounter, system_prompt: str = ""
) -> int:
    """Tokens del ejemplo renderizado completo (aprox. al render del trainer)."""
    parts = [
        system_prompt,
        example.instruction,
        example.input,
        example.response,
    ]
    return sum(counter.count(part) for part in parts) + CHAT_TEMPLATE_OVERHEAD_TOKENS


def budget_report(
    examples: list[Any],
    counter: TokenCounter,
    *,
    seq_len: int = 2048,
    system_prompt: str = "",
) -> dict[str, Any]:
    """Estadísticas de longitud total renderizada frente a ``seq_len``."""
    if not examples:
        return {"n": 0, "over": 0, "over_frac": 0.0, "seq_len": seq_len, "p50": 0, "max": 0}
    totals = sorted(
        example_total_tokens(example, counter, system_prompt) for example in examples
    )
    over = sum(1 for total in totals if total > seq_len)
    return {
        "n": len(totals),
        "seq_len": seq_len,
        "over": over,
        "over_frac": round(over / len(totals), 4),
        "p50": totals[len(totals) // 2],
        "p95": totals[int(len(totals) * 0.95)],
        "max": totals[-1],
        "counter": counter.describe(),
    }


def assert_token_budget(
    examples: list[Any],
    counter: TokenCounter,
    *,
    seq_len: int = 2048,
    max_over_frac: float = 0.02,
    system_prompt: str = "",
) -> dict[str, Any]:
    """Falla ruidosamente si demasiados ejemplos exceden ``seq_len``."""
    report = budget_report(
        examples, counter, seq_len=seq_len, system_prompt=system_prompt
    )
    if report["n"] and report["over_frac"] > max_over_frac:
        raise ValueError(
            f"Presupuesto de tokens violado: {report['over']}/{report['n']} ejemplos "
            f"({report['over_frac']:.1%}) exceden seq_len={seq_len} "
            f"(límite {max_over_frac:.1%}). p95={report['p95']} max={report['max']}. "
            "Revisa la ventana de contexto de synthesize antes de entrenar."
        )
    return report
