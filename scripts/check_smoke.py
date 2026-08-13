"""Verifica el resultado del smoke-train en CI (solo stdlib).

Lee el metrics.json del run y comprueba que la loss final es menor que la
inicial (aprendizaje real en 30 pasos) y que se completaron pasos. Lo usa
.github/workflows/smoke-train.yml en el runner self-hosted con GPU.

Uso::

    python scripts/check_smoke.py artifacts/smoke/metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Uso: python scripts/check_smoke.py <ruta/metrics.json>")
        return 2
    metrics_path = Path(sys.argv[1])
    if not metrics_path.exists():
        print(f"[check_smoke] FALLO: no existe {metrics_path}")
        return 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    initial = metrics.get("train_loss_initial")
    final = metrics.get("train_loss_final")
    steps = metrics.get("steps", 0)

    if initial is None or final is None:
        print(f"[check_smoke] FALLO: métricas de loss ausentes en {metrics_path}")
        return 1
    if steps <= 0:
        print(f"[check_smoke] FALLO: 0 pasos de entrenamiento (steps={steps})")
        return 1
    if not final < initial:
        print(
            f"[check_smoke] FALLO: la loss no desciende "
            f"(inicial={initial:.4f}, final={final:.4f})"
        )
        return 1
    print(
        f"[check_smoke] OK: loss {initial:.4f} -> {final:.4f} en {steps} pasos "
        f"(unsloth={metrics.get('used_unsloth')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
