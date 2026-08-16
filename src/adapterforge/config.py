"""Modelos pydantic para las configuraciones de experimento (``configs/*.yaml``).

La validación ocurre ANTES de cargar cualquier dependencia pesada: un YAML
inviable falla en milisegundos, no tras 30 segundos importando torch. Además
de validar tipos y rangos, se estima el consumo de VRAM para el presupuesto
de 8 GB: combinaciones claramente inviables (``seq_len * batch_size``
excesivo) se rechazan en duro; el resto de riesgos se reporta como warnings
suaves en los scripts de entrenamiento.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Presupuesto máximo de tokens por paso de dispositivo (seq_len * batch_size)
# considerado viable en 8 GB con QLoRA NF4 + gradient checkpointing. Por encima
# el OOM es casi seguro y la validación falla en duro.
MAX_TOKENS_PER_DEVICE_STEP = 16_384

# Heurísticas de estimación de VRAM (GB). Son estimaciones de planificación:
# verifica siempre con nvidia-smi (ver README, matriz de fallback OOM).
_NF4_GB_PER_BILLION = 0.55  # pesos NF4 por cada 10^9 parámetros
_FIXED_OVERHEAD_GB = 1.5  # contexto CUDA + optimizador paginado + fragmentación
_ACT_GB_PER_B_PER_KTOKEN = 0.35  # activaciones por B de params y 1024 tokens/paso

_DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class LoraSettings(BaseModel):
    """Hiperparámetros del adapter LoRA."""

    model_config = ConfigDict(extra="forbid")

    r: int = Field(default=16, ge=1, le=256)
    alpha: int = Field(default=32, ge=1, le=1024)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_TARGET_MODULES),
        min_length=1,
    )


class DpoSettings(BaseModel):
    """Hiperparámetros específicos de la fase DPO."""

    model_config = ConfigDict(extra="forbid")

    beta: float = Field(default=0.1, gt=0.0, le=5.0)
    max_prompt_len: int = Field(default=512, ge=64, le=8192)


class ExperimentConfig(BaseModel):
    """Configuración completa y reproducible de un experimento.

    Un mismo YAML sirve para SFT y (con la sección ``dpo``) para DPO. El campo
    ``max_steps`` permite runs de humo (smoke) cortos en CI.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    base_model: str = Field(min_length=1)
    lora: LoraSettings = Field(default_factory=LoraSettings)
    dpo: DpoSettings | None = None

    seq_len: int = Field(default=1024, ge=64, le=8192)
    lr: float = Field(default=2e-4, gt=0.0, lt=0.1)
    epochs: float = Field(default=2.0, gt=0.0, le=20.0)
    batch_size: int = Field(default=1, ge=1, le=64)
    grad_accum: int = Field(default=16, ge=1, le=512)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    seed: int = Field(default=42, ge=0)
    max_steps: int | None = Field(default=None, ge=1)

    dataset_path: Path
    val_dataset_path: Path | None = Field(
        default=None,
        description="Split de validación (eval_loss + early stopping); sin él "
        "el entrenamiento vuela a ciegas y publica el estado más sobreajustado",
    )
    output_dir: Path
    vram_budget_gb: float = Field(default=8.0, gt=0.0, le=80.0)

    @model_validator(mode="after")
    def _reject_infeasible_step(self) -> ExperimentConfig:
        """Rechaza en duro combinaciones seq_len*batch inviables en 8 GB."""
        tokens_per_step = self.seq_len * self.batch_size
        if tokens_per_step > MAX_TOKENS_PER_DEVICE_STEP:
            raise ValueError(
                f"seq_len*batch_size={tokens_per_step} supera el máximo viable "
                f"({MAX_TOKENS_PER_DEVICE_STEP}) para una GPU de 8 GB con QLoRA. "
                "Reduce seq_len o batch_size y compensa con grad_accum "
                "(ver matriz de fallback OOM en el README)."
            )
        return self

    def model_size_billion(self) -> float | None:
        """Extrae el tamaño en miles de millones de parámetros del nombre.

        Ejemplos: ``Qwen/Qwen3-4B-Instruct-2507`` -> 4.0;
        ``meta-llama/Llama-3.2-3B-Instruct`` -> 3.0. Devuelve ``None`` si el
        nombre no codifica el tamaño (p. ej. ``SmolLM2-360M``).
        """
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])", self.base_model)
        if not matches:
            return None
        return float(matches[-1])

    def estimate_vram_gb(self) -> float | None:
        """Estimación heurística de pico de VRAM en GB (None si no hay tamaño)."""
        size_b = self.model_size_billion()
        if size_b is None:
            return None
        weights = size_b * _NF4_GB_PER_BILLION
        ktokens = (self.seq_len * self.batch_size) / 1024.0
        activations = size_b * ktokens * _ACT_GB_PER_B_PER_KTOKEN
        return round(_FIXED_OVERHEAD_GB + weights + activations, 2)

    def vram_warnings(self) -> list[str]:
        """Avisos suaves (no bloqueantes) sobre el presupuesto de VRAM."""
        warnings: list[str] = []
        estimate = self.estimate_vram_gb()
        if estimate is not None and estimate > self.vram_budget_gb:
            warnings.append(
                f"VRAM estimada {estimate:.1f} GB > presupuesto "
                f"{self.vram_budget_gb:.1f} GB para '{self.base_model}'. "
                "Riesgo de OOM: usa la matriz de fallback del README "
                "(modelo menor, seq_len menor o batch 1 + grad_accum)."
            )
        if self.seq_len * self.batch_size > MAX_TOKENS_PER_DEVICE_STEP // 2:
            warnings.append(
                f"seq_len*batch_size={self.seq_len * self.batch_size} está en la "
                "zona alta del presupuesto de 8 GB; activa gradient checkpointing "
                "y vigila nvidia-smi durante los primeros pasos."
            )
        return warnings

    def effective_batch_size(self) -> int:
        """Batch efectivo por paso de optimización."""
        return self.batch_size * self.grad_accum


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Carga y valida un YAML de experimento.

    Lanza ``FileNotFoundError`` si no existe, ``ValueError`` si el YAML no es
    un mapeo, y ``pydantic.ValidationError`` si los valores son inválidos.
    """
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"El fichero de configuración {config_path} no es un mapeo YAML")
    return ExperimentConfig.model_validate(raw)
