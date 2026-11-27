"""Alineamiento DPO con TRL ``DPOTrainer`` sobre el adapter SFT.

Consume pares de preferencia en formato conversacional
(``prompt``/``chosen``/``rejected``, ver ``data/processed/sample_dpo.jsonl``)
y continúa el entrenamiento del adapter para mejorar formato y estilo.

Con PEFT no hace falta un modelo de referencia separado: TRL desactiva el
adapter para computar las log-probs de referencia (ahorro clave en 8 GB).

Mismas salvaguardas que :mod:`adapterforge.train.sft`: el chat template
OFICIAL de Qwen3-Instruct-2507 se fuerza antes de renderizar (DPOTrainer
aplica el del tokenizer, y el del mirror de Unsloth es el thinking), un
render de muestra con ``<think>``/``<tool_call>`` aborta, y una loss no
finita para el run sin guardar el adapter.

Uso::

    python -m adapterforge.train.dpo --config configs/dpo_qwen3-4b.yaml
    python -m adapterforge.train.dpo --config ... --smoke --mlflow

TODO(jairo): ORPO como alternativa aún más barata en memoria (sin fase SFT
separada ni referencia); ver docs/adr/0002.
"""
from __future__ import annotations

try:  # Unsloth debe importarse ANTES que trl/transformers/peft (parcheo global coherente).
    import unsloth  # noqa: F401
except ImportError:  # entornos sin GPU/unsloth (smoke CPU): sin efecto
    pass

import argparse
import time
from pathlib import Path

from adapterforge.config import DpoSettings, ExperimentConfig, load_experiment_config
from adapterforge.settings import get_settings
from adapterforge.train.common import (
    build_run_metrics,
    emit_vram_warnings,
    extract_loss_curve,
    filter_kwargs_for_dataclass,
    load_official_chat_template,
    log_to_mlflow,
    save_metrics,
)
from adapterforge.train.sft import _assert_clean_render

SMOKE_MAX_STEPS = 30


def run_dpo(cfg: ExperimentConfig, *, use_mlflow: bool = False) -> dict:
    """Ejecuta la fase DPO. Imports pesados solo aquí."""
    import torch
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    from adapterforge.train.callbacks import AbortOnNonFiniteLoss
    from adapterforge.train.runtime import load_model_and_tokenizer

    emit_vram_warnings(cfg)
    dpo_settings = cfg.dpo or DpoSettings()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, used_unsloth = load_model_and_tokenizer(cfg)
    # DPOTrainer renderiza prompt/chosen/rejected con el template del
    # tokenizer, y el del mirror de Unsloth es el thinking: sin forzar el
    # oficial, DPO volvería a quemar «<think>» en los pesos (defecto v1/v2/v2b).
    tokenizer.chat_template = load_official_chat_template()
    print("[dpo] chat template oficial de Qwen3-Instruct-2507 forzado (sin <think>)")

    dataset = load_dataset("json", data_files=str(cfg.dataset_path), split="train")
    if "meta" in dataset.column_names:
        dataset = dataset.remove_columns(["meta"])
    samples = [dataset[i] for i in range(min(5, len(dataset)))]
    _assert_clean_render(
        [
            tokenizer.apply_chat_template(
                list(sample["prompt"]) + list(sample["chosen"]), tokenize=False
            )
            for sample in samples
            if isinstance(sample.get("prompt"), list)  # formato conversacional
        ],
        label="dpo",
    )

    cuda_available = torch.cuda.is_available()
    bf16 = cuda_available and torch.cuda.is_bf16_supported()
    raw_kwargs = {
        "output_dir": str(output_dir / "trainer"),
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.lr,
        "num_train_epochs": cfg.epochs,
        "max_steps": cfg.max_steps or -1,
        "warmup_ratio": cfg.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "seed": cfg.seed,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "optim": "paged_adamw_8bit" if cuda_available else "adamw_torch",
        "gradient_checkpointing": not used_unsloth and cuda_available,
        "bf16": bf16,
        "fp16": cuda_available and not bf16,
        "beta": dpo_settings.beta,
        "max_length": cfg.seq_len,
        "max_prompt_length": dpo_settings.max_prompt_len,
    }
    dpo_config = DPOConfig(**filter_kwargs_for_dataclass(DPOConfig, raw_kwargs))

    abort_on_nan = AbortOnNonFiniteLoss(label="dpo")
    trainer_kwargs = {
        "model": model,
        "args": dpo_config,
        "train_dataset": dataset,
        "callbacks": [abort_on_nan],
    }
    try:
        trainer = DPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = DPOTrainer(tokenizer=tokenizer, **trainer_kwargs)

    started = time.monotonic()
    trainer.train()
    runtime_s = time.monotonic() - started

    loss_initial, loss_final = extract_loss_curve(trainer.state.log_history)
    # Como en SFT: un run divergido deja sus métricas y sale != 0 ANTES de
    # save_model, para que ningún adapter con NaN llegue a merge/GGUF/Hub.
    if abort_on_nan.diverged:
        metrics = build_run_metrics(
            cfg,
            phase="dpo",
            steps=int(trainer.state.global_step),
            loss_initial=loss_initial,
            loss_final=loss_final,
            runtime_s=runtime_s,
            used_unsloth=used_unsloth,
            extra={
                "dpo_beta": dpo_settings.beta,
                "diverged": True,
                "diverged_step": abort_on_nan.bad_step,
            },
        )
        save_metrics(output_dir / "metrics.json", metrics)
        print("[dpo] ERROR: la loss divergió; NO se guarda ningún adapter")
        raise SystemExit(1)

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = build_run_metrics(
        cfg,
        phase="dpo",
        steps=int(trainer.state.global_step),
        loss_initial=loss_initial,
        loss_final=loss_final,
        runtime_s=runtime_s,
        used_unsloth=used_unsloth,
        extra={"dpo_beta": dpo_settings.beta},
    )
    save_metrics(output_dir / "metrics.json", metrics)
    (output_dir / "experiment.json").write_text(
        cfg.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    if use_mlflow:
        log_to_mlflow(cfg, metrics, get_settings().mlflow_tracking_uri)

    print(
        f"[dpo] fin: steps={metrics['steps']} loss {loss_initial} -> {loss_final} "
        f"({runtime_s:.0f}s). Adapter en {adapter_dir}"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DPO (TRL) sobre pares de preferencia")
    parser.add_argument("--config", required=True, help="YAML de experimento con sección dpo")
    parser.add_argument("--smoke", action="store_true", help="Run corto (max_steps=30)")
    parser.add_argument("--mlflow", action="store_true", help="Tracking en MLflow")
    parser.add_argument("--dataset", default=None, help="Override de dataset_path")
    parser.add_argument("--output-dir", default=None, help="Override de output_dir")
    args = parser.parse_args(argv)

    cfg = load_experiment_config(args.config)
    updates: dict = {}
    if args.smoke:
        updates["max_steps"] = cfg.max_steps or SMOKE_MAX_STEPS
    if args.dataset:
        updates["dataset_path"] = args.dataset
    if args.output_dir:
        updates["output_dir"] = args.output_dir
    if updates:
        cfg = cfg.model_copy(update=updates)

    run_dpo(cfg, use_mlflow=args.mlflow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
