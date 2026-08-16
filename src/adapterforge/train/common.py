"""Utilidades de entrenamiento sin dependencias pesadas (testables sin GPU)."""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path
from typing import Any

from adapterforge.config import ExperimentConfig

# Template de chat OFICIAL de Qwen3-*-Instruct-2507, versionado en el repo.
# El mirror de Unsloth trae la variante thinking (inyecta '<think>\n\n</think>'
# en cada turno de assistant): entrenar con ella quemó ese prefijo en los
# pesos de v1/v2/v2b. El template correcto se fuerza SIEMPRE antes de
# renderizar y se propaga al adapter guardado y al modelo mergeado.
OFFICIAL_CHAT_TEMPLATE_PATH = (
    Path(__file__).parent / "templates" / "qwen3_instruct_2507.chat_template.jinja"
)


def load_official_chat_template() -> str:
    """Lee el template oficial versionado (falla ruidosamente si no está)."""
    template = OFFICIAL_CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    if "<think>" in template:
        raise RuntimeError(
            f"{OFFICIAL_CHAT_TEMPLATE_PATH} contiene '<think>': no es el "
            "template de la variante Instruct-2507 (no pensante)"
        )
    return template


def git_sha() -> str | None:
    """SHA corto del HEAD actual (None si git no está disponible)."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001 — best effort: sin git no se bloquea el run
        return None


def emit_vram_warnings(cfg: ExperimentConfig) -> list[str]:
    """Emite (y devuelve) los avisos suaves de presupuesto de VRAM.

    Son *asserts suaves*: no detienen el run, pero quedan visibles en logs y
    en CI. La validación dura (combinaciones inviables) ya ocurrió al cargar
    el ``ExperimentConfig``.
    """
    messages = cfg.vram_warnings()
    for message in messages:
        warnings.warn(message, stacklevel=2)
    return messages


def extract_loss_curve(log_history: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """Devuelve (primera, última) loss de entrenamiento del historial de TRL.

    Ignora entradas sin clave ``loss`` (p. ej. métricas de evaluación o el
    resumen final ``train_loss``).
    """
    losses = [float(entry["loss"]) for entry in log_history if "loss" in entry]
    if not losses:
        return None, None
    return losses[0], losses[-1]


def filter_kwargs_for_dataclass(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filtra kwargs a los campos aceptados por un dataclass (TRL cambia de
    nombres entre versiones; así el script sobrevive a esos cambios)."""
    accepted = {field.name for field in dataclasses.fields(cls)}
    return {key: value for key, value in kwargs.items() if key in accepted}


def build_run_metrics(
    cfg: ExperimentConfig,
    *,
    phase: str,
    steps: int,
    loss_initial: float | None,
    loss_final: float | None,
    runtime_s: float,
    used_unsloth: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construye el diccionario de métricas que se persiste como JSON."""
    metrics: dict[str, Any] = {
        "experiment": cfg.name,
        "phase": phase,
        "base_model": cfg.base_model,
        "steps": steps,
        "train_loss_initial": loss_initial,
        "train_loss_final": loss_final,
        "train_runtime_s": round(runtime_s, 2),
        "effective_batch_size": cfg.effective_batch_size(),
        "seq_len": cfg.seq_len,
        "lora_r": cfg.lora.r,
        "used_unsloth": used_unsloth,
        "vram_estimate_gb": cfg.estimate_vram_gb(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra:
        metrics.update(extra)
    return metrics


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> Path:
    """Escribe las métricas del run como JSON (crea directorios)."""
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics_path


def log_to_mlflow(cfg: ExperimentConfig, metrics: dict[str, Any], tracking_uri: str) -> None:
    """Registra parámetros y métricas en MLflow (import perezoso).

    TODO(jairo): registrar también el adapter como artefacto y comparar runs
    (SFT vs SFT+DPO) desde la UI de MLflow.
    """
    import mlflow  # perezoso: solo si se pide tracking

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("adapterforge")
    with mlflow.start_run(run_name=f"{cfg.name}-{metrics.get('phase', 'run')}"):
        mlflow.log_params(
            {
                "base_model": cfg.base_model,
                "seq_len": cfg.seq_len,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "grad_accum": cfg.grad_accum,
                "lora_r": cfg.lora.r,
                "lora_alpha": cfg.lora.alpha,
                "lora_dropout": cfg.lora.dropout,
                "seed": cfg.seed,
                "max_steps": cfg.max_steps,
            }
        )
        numeric = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, int | float) and value is not None
        }
        mlflow.log_metrics(numeric)
