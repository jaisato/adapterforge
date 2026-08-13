"""Tests de validación de ExperimentConfig y de los YAML incluidos en configs/."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adapterforge.config import (
    MAX_TOKENS_PER_DEVICE_STEP,
    ExperimentConfig,
    load_experiment_config,
)


def _base_payload(**overrides) -> dict:
    payload = {
        "name": "test-run",
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "dataset_path": "data/processed/sample_sft.jsonl",
        "output_dir": "artifacts/test-run",
    }
    payload.update(overrides)
    return payload


def test_valid_config_defaults():
    cfg = ExperimentConfig.model_validate(_base_payload())
    assert cfg.lora.r == 16
    assert cfg.lora.alpha == 32
    assert cfg.seq_len == 1024
    assert cfg.effective_batch_size() == cfg.batch_size * cfg.grad_accum
    assert cfg.model_size_billion() == 4.0


def test_rejects_infeasible_seq_len_times_batch():
    # 4096 * 8 = 32768 > presupuesto de tokens por paso en 8 GB.
    with pytest.raises(ValidationError, match="supera el máximo viable"):
        ExperimentConfig.model_validate(_base_payload(seq_len=4096, batch_size=8))
    # Justo en el límite: válido.
    cfg = ExperimentConfig.model_validate(_base_payload(seq_len=8192, batch_size=2))
    assert cfg.seq_len * cfg.batch_size == MAX_TOKENS_PER_DEVICE_STEP


def test_rejects_out_of_range_values():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_base_payload(lora={"dropout": 1.5}))
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_base_payload(lr=0.0))
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_base_payload(epochs=-1))
    # extra="forbid": los typos en el YAML no pasan en silencio.
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_base_payload(learning_rate=1e-4))


def test_vram_estimate_and_warnings():
    small = ExperimentConfig.model_validate(
        _base_payload(base_model="meta-llama/Llama-3.2-3B-Instruct", seq_len=1024, batch_size=1)
    )
    estimate = small.estimate_vram_gb()
    assert estimate is not None and 2.0 < estimate < 8.0
    assert small.vram_warnings() == []

    big = ExperimentConfig.model_validate(
        _base_payload(base_model="meta-llama/Llama-3.1-8B-Instruct", seq_len=2048, batch_size=2)
    )
    warnings = big.vram_warnings()
    assert warnings, "un 8B con 4096 tokens/paso debe avisar de riesgo de OOM"
    # Modelo sin tamaño en el nombre: sin estimación, sin warning de estimación.
    unknown = ExperimentConfig.model_validate(_base_payload(base_model="local/mystery-model"))
    assert unknown.estimate_vram_gb() is None


def test_shipped_configs_are_valid(repo_root):
    config_paths = sorted((repo_root / "configs").glob("*.yaml"))
    assert len(config_paths) >= 3, "deben existir al menos 3 configs de ejemplo"
    for path in config_paths:
        cfg = load_experiment_config(path)
        assert cfg.name
        assert cfg.seq_len * cfg.batch_size <= MAX_TOKENS_PER_DEVICE_STEP
