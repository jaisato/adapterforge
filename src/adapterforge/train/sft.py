"""SFT QLoRA con TRL ``SFTTrainer`` + PEFT sobre base 4-bit (NF4).

Lee un ``ExperimentConfig`` (configs/*.yaml), entrena el adapter y guarda:

- ``<output_dir>/adapter/``: pesos LoRA + tokenizer (con el chat template
  OFICIAL de Qwen3-Instruct-2507, no el del mirror de Unsloth).
- ``<output_dir>/metrics.json``: loss inicial/final, eval_loss, pasos,
  runtime, git_sha, flags.
- ``<output_dir>/experiment.json``: configuración resuelta (reproducibilidad).

Salvaguardas (lecciones de v1/v2/v2b):

- El chat template oficial se fuerza SIEMPRE antes de renderizar y se
  aborta si un render de muestra contiene ``<think>``/``<tool_call>``.
- Con ``val_dataset_path`` en el config: eval_loss periódica, checkpoints,
  ``load_best_model_at_end`` y early stopping — se publica el mejor estado,
  no el último (el más sobreajustado).
- Una loss no finita aborta el run ANTES de guardar nada
  (:class:`adapterforge.train.callbacks.AbortOnNonFiniteLoss`).

Modo ``--smoke``: fuerza ``max_steps=30`` para el test de humo en CI (runner
self-hosted con GPU); el workflow verifica que la loss final < inicial.

Uso::

    python -m adapterforge.train.sft --config configs/qlora_qwen3-4b_v3.yaml
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
    git_sha,
    load_official_chat_template,
    log_to_mlflow,
    save_metrics,
)

SMOKE_MAX_STEPS = 30
# Cada 50 pasos: ~10 evals en un run de ~512 pasos. A batch 1 (sin riesgo de
# OOM en 8 GB), evaluar 233 ejemplos cuesta ~100 s; a cadencia 25 el run se
# alargaba ~34 min (medido por la revisión contra la imagen real).
EVAL_EVERY_STEPS = 50
EARLY_STOPPING_PATIENCE = 4  # evals sin mejora (4 x 50 pasos) antes de parar

# Tokens que jamás deben aparecer en un render de entrenamiento de la
# variante Instruct-2507 (no pensante, sin tools en nuestros datos).
_FORBIDDEN_RENDER_TOKENS = ("<think>", "</think>", "<tool_call>", "</tool_call>")


def _assert_clean_render(texts: list[str]) -> None:
    """Aborta si el render de muestra está contaminado o mal formado."""
    for text in texts:
        for token in _FORBIDDEN_RENDER_TOKENS:
            if token in text:
                raise SystemExit(
                    f"[sft] ABORTADO: el render contiene {token!r} — el chat "
                    "template no es el oficial de Instruct-2507. Ningún "
                    "adapter debe entrenarse así (defecto v1/v2/v2b)."
                )
        if "<|im_start|>assistant\n" not in text:
            raise SystemExit(
                "[sft] ABORTADO: el render no contiene el turno de assistant "
                "esperado; revisa el chat template y el formato de mensajes."
            )


def run_sft(cfg: ExperimentConfig, *, use_mlflow: bool = False, resume: bool = False) -> dict:
    """Ejecuta el SFT completo. Todos los imports pesados viven aquí."""
    import torch
    from datasets import load_dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    from adapterforge.train.callbacks import AbortOnNonFiniteLoss
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

    # El tokenizer del mirror de Unsloth trae el template de la variante
    # thinking; se sobreescribe SIEMPRE con el oficial versionado en el repo.
    tokenizer.chat_template = load_official_chat_template()
    print("[sft] chat template oficial de Qwen3-Instruct-2507 forzado (sin <think>)")

    def _to_text(example: dict) -> dict:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    # Red anti-NaN: descarta ejemplos que no caben en seq_len (si el truncado se
    # tragara la respuesta, train_on_responses_only dejaría 0 tokens -> loss NaN).
    def _fits(example: dict) -> bool:
        return len(tokenizer(example["text"], add_special_tokens=False).input_ids) <= cfg.seq_len

    def _prepare(path: Path | str, label: str):
        ds = load_dataset("json", data_files=str(path), split="train")
        if "meta" in ds.column_names:
            ds = ds.remove_columns(["meta"])
        if "messages" in ds.column_names and "text" not in ds.column_names:
            ds = ds.map(_to_text, remove_columns=ds.column_names)
        n_before = len(ds)
        ds = ds.filter(_fits)
        dropped = n_before - len(ds)
        print(f"[sft] {label}: {len(ds)} ejemplos ({dropped} descartados > seq_len)")
        if n_before and dropped / n_before > 0.05:
            raise SystemExit(
                f"[sft] ABORTADO: el filtro de longitud descarta {dropped}/{n_before} "
                f"ejemplos de {label} (>5%). El presupuesto de tokens del dataset "
                "está roto; regenera los datos (lección v2b: el descarte silencioso "
                "sesgó la composición)."
            )
        return ds

    dataset = _prepare(cfg.dataset_path, "train")
    _assert_clean_render([dataset[i]["text"] for i in range(min(5, len(dataset)))])

    eval_dataset = None
    if cfg.val_dataset_path is not None:
        if not Path(cfg.val_dataset_path).exists():
            raise SystemExit(f"[sft] val_dataset_path no existe: {cfg.val_dataset_path}")
        eval_dataset = _prepare(cfg.val_dataset_path, "val")

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
    if eval_dataset is not None:
        raw_kwargs.update(
            {
                # eval_strategy es el nombre actual; evaluation_strategy el
                # antiguo — filter_kwargs_for_dataclass escoge el que exista.
                "eval_strategy": "steps",
                "evaluation_strategy": "steps",
                "eval_steps": EVAL_EVERY_STEPS,
                "per_device_eval_batch_size": 1,
                "save_strategy": "steps",
                "save_steps": EVAL_EVERY_STEPS,
                "save_total_limit": 2,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    sft_config = SFTConfig(**filter_kwargs_for_dataclass(SFTConfig, raw_kwargs))

    abort_on_nan = AbortOnNonFiniteLoss()
    callbacks = [abort_on_nan]
    if eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))

    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
        "eval_dataset": eval_dataset,
        "callbacks": callbacks,
    }
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        # Compatibilidad con TRL antiguos (tokenizer= en lugar de processing_class=).
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    if used_unsloth:
        try:
            from unsloth.chat_templates import train_on_responses_only

            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
            print("[sft] train_on_responses_only activado (loss solo en respuestas)")
        except Exception as exc:  # noqa: BLE001 — no bloquear el train si la API cambia
            print(f"[sft] aviso: train_on_responses_only no disponible: {exc}")

    started = time.monotonic()
    # Reanudación: retoma del último checkpoint de <output_dir>/trainer con
    # optimizador, scheduler y contador de pasos (un segfault del stack GPU
    # de WSL a mitad de run ya no cuesta el entrenamiento entero).
    last_checkpoint = None
    if resume:
        from transformers.trainer_utils import get_last_checkpoint

        last_checkpoint = get_last_checkpoint(str(output_dir / "trainer"))
        if last_checkpoint is None:
            print("[sft] --resume sin checkpoints previos: se entrena desde cero")
        else:
            print(f"[sft] reanudando desde {last_checkpoint}")
    result = trainer.train(resume_from_checkpoint=last_checkpoint)
    runtime_s = time.monotonic() - started

    loss_initial, loss_final = extract_loss_curve(trainer.state.log_history)
    eval_losses = [
        float(entry["eval_loss"])
        for entry in trainer.state.log_history
        if "eval_loss" in entry
    ]
    extra = {
        "train_loss_mean": getattr(result, "training_loss", None),
        "git_sha": git_sha(),
        "chat_template": "qwen3_instruct_2507-oficial",
        "val_examples": len(eval_dataset) if eval_dataset is not None else 0,
    }
    if eval_losses:
        extra.update(
            {
                "eval_loss_final": eval_losses[-1],
                "eval_loss_best": min(eval_losses),
                "n_evals": len(eval_losses),
            }
        )

    # Un run divergido no guarda adapter: se persiguen métricas y se sale != 0
    # ANTES de save_model (en v2 este orden estaba invertido).
    if abort_on_nan.diverged:
        metrics = build_run_metrics(
            cfg,
            phase="sft",
            steps=int(trainer.state.global_step),
            loss_initial=loss_initial,
            loss_final=loss_final,
            runtime_s=runtime_s,
            used_unsloth=used_unsloth,
            extra={**extra, "diverged": True, "diverged_step": abort_on_nan.bad_step},
        )
        save_metrics(output_dir / "metrics.json", metrics)
        print("[sft] ERROR: la loss divergió; NO se guarda ningún adapter")
        raise SystemExit(1)

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
        extra=extra,
    )
    save_metrics(output_dir / "metrics.json", metrics)

    # Guardia redundante: métricas con NaN nunca deben pasar desapercibidas.
    import json as _json
    import math as _math

    saved_metrics = _json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    if any(isinstance(v, float) and _math.isnan(v) for v in saved_metrics.values()):
        print("[sft] ERROR: métricas con NaN — la loss divergió; el adapter NO es válido")
        raise SystemExit(1)

    (output_dir / "experiment.json").write_text(
        cfg.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    if use_mlflow:
        log_to_mlflow(cfg, metrics, get_settings().mlflow_tracking_uri)

    eval_note = (
        f" eval_loss best={extra.get('eval_loss_best'):.4f}" if eval_losses else ""
    )
    print(
        f"[sft] fin: steps={metrics['steps']} loss {loss_initial} -> {loss_final}"
        f"{eval_note} ({runtime_s:.0f}s, unsloth={used_unsloth}). Adapter en {adapter_dir}"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SFT QLoRA (TRL + PEFT, base 4-bit)")
    parser.add_argument("--config", required=True, help="YAML de experimento")
    parser.add_argument("--smoke", action="store_true", help="Run corto (max_steps=30)")
    parser.add_argument("--mlflow", action="store_true", help="Tracking en MLflow")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reanudar desde el último checkpoint de <output_dir>/trainer",
    )
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

    run_sft(cfg, use_mlflow=args.mlflow, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
