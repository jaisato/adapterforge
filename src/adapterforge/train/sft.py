"""SFT QLoRA con TRL ``SFTTrainer`` + PEFT sobre base 4-bit (NF4).

Lee un ``ExperimentConfig`` (configs/*.yaml), entrena el adapter y guarda:

- ``<output_dir>/adapter/``: pesos LoRA + tokenizer.
- ``<output_dir>/metrics.json``: loss inicial/final, pasos, runtime, flags.
- ``<output_dir>/experiment.json``: configuración resuelta (reproducibilidad).

Modo ``--smoke``: fuerza ``max_steps=30`` para el test de humo en CI (runner
self-hosted con GPU); el workflow verifica que la loss final < inicial.

Uso::

    python -m adapterforge.train.sft --config configs/qlora_qwen3-4b.yaml
    python -m adapterforge.train.sft --config configs/qlora_smoke.yaml --smoke
    python -m adapterforge.train.sft --config ... --mlflow   # tracking opcional

TODO(jairo): reanudación desde checkpoint (resume_from_checkpoint) y packing
de secuencias para acelerar en corpus grandes.
"""
from __future__ import annotations

try:  # Unsloth debe importarse ANTES que trl/transformers/peft (parcheo global coherente).
    import unsloth  # noqa: F401
except ImportError:  # entornos sin GPU/unsloth (smoke CPU): sin efecto
    pass

import argparse
import time
from pathlib import Path

from adapterforge.config import ExperimentConfig, load_experiment_config
from adapterforge.settings import get_settings
from adapterforge.train.common import (
    build_run_metrics,
    emit_vram_warnings,
    extract_loss_curve,
    filter_kwargs_for_dataclass,
    log_to_mlflow,
    save_metrics,
)

SMOKE_MAX_STEPS = 30


def run_sft(cfg: ExperimentConfig, *, use_mlflow: bool = False) -> dict:
    """Ejecuta el SFT completo. Todos los imports pesados viven aquí."""
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    from adapterforge.train.runtime import load_model_and_tokenizer

    emit_vram_warnings(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        if total_gb + 0.5 < cfg.vram_budget_gb:
            # Assert suave: la GPU real tiene menos memoria que el presupuesto.
            print(
                f"[sft] AVISO: GPU con {total_gb:.1f} GB < presupuesto "
                f"{cfg.vram_budget_gb:.1f} GB; revisa la matriz de fallback OOM."
            )

    model, tokenizer, used_unsloth = load_model_and_tokenizer(cfg)

    dataset = load_dataset("json", data_files=str(cfg.dataset_path), split="train")
    if "meta" in dataset.column_names:
        dataset = dataset.remove_columns(["meta"])

    # El SFTTrainer parcheado por Unsloth no auto-formatea el formato "messages":
    # aplicamos la chat template del tokenizer y entrenamos sobre la columna "text".
    if "messages" in dataset.column_names and "text" not in dataset.column_names:

        def _to_text(example: dict) -> dict:
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False
                )
            }

        dataset = dataset.map(_to_text, remove_columns=dataset.column_names)

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
        # TRL renombró max_seq_length -> max_length entre versiones; pasamos
        # ambos y filter_kwargs_for_dataclass se queda con el que exista.
        "max_length": cfg.seq_len,
        "max_seq_length": cfg.seq_len,
        "dataset_text_field": "text",
        "eos_token": getattr(tokenizer, "eos_token", None),
    }
    sft_config = SFTConfig(**filter_kwargs_for_dataclass(SFTConfig, raw_kwargs))

    try:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    except TypeError:
        # Compatibilidad con TRL antiguos (tokenizer= en lugar de processing_class=).
        trainer = SFTTrainer(
            model=model, args=sft_config, train_dataset=dataset, tokenizer=tokenizer
        )

    started = time.monotonic()
    result = trainer.train()
    runtime_s = time.monotonic() - started

    loss_initial, loss_final = extract_loss_curve(trainer.state.log_history)
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = build_run_metrics(
        cfg,
        phase="sft",
        steps=int(trainer.state.global_step),
        loss_initial=loss_initial,
        loss_final=loss_final,
        runtime_s=runtime_s,
        used_unsloth=used_unsloth,
        extra={"train_loss_mean": getattr(result, "training_loss", None)},
    )
    save_metrics(output_dir / "metrics.json", metrics)
    (output_dir / "experiment.json").write_text(
        cfg.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    if use_mlflow:
        log_to_mlflow(cfg, metrics, get_settings().mlflow_tracking_uri)

    print(
        f"[sft] fin: steps={metrics['steps']} loss {loss_initial} -> {loss_final} "
        f"({runtime_s:.0f}s, unsloth={used_unsloth}). Adapter en {adapter_dir}"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SFT QLoRA (TRL + PEFT, base 4-bit)")
    parser.add_argument("--config", required=True, help="YAML de experimento")
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

    run_sft(cfg, use_mlflow=args.mlflow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
