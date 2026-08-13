"""Carga del modelo base 4-bit con integración opcional de Unsloth.

Orden de preferencia:

1. **Unsloth** (si está instalado): kernels fusionados, ~1.5-2x más rápido y
   menos VRAM; es lo que permite el stretch de 7-8B en 8 GB.
2. **transformers + bitsandbytes (NF4 double quant) + PEFT**: camino estándar,
   funciona en cualquier entorno CUDA.
3. **Sin CUDA** (smoke en CPU): se carga sin cuantizar — bitsandbytes requiere
   GPU — para poder validar el pipeline con modelos diminutos.

Todos los imports pesados son perezosos.
"""

from __future__ import annotations

from typing import Any

from adapterforge.config import ExperimentConfig


def load_model_and_tokenizer(cfg: ExperimentConfig) -> tuple[Any, Any, bool]:
    """Devuelve ``(modelo_con_lora, tokenizer, used_unsloth)``."""
    try:
        from unsloth import FastLanguageModel  # perezoso y opcional
    except ImportError:
        FastLanguageModel = None  # noqa: N806 — nombre de clase externo

    if FastLanguageModel is not None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.seq_len,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
        )
        return model, tokenizer, True

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cuda_available = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        # Muchos modelos instruct no definen pad_token; reutilizamos eos.
        tokenizer.pad_token = tokenizer.eos_token

    if cuda_available:
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=quant_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # Smoke sin GPU: bitsandbytes no funciona en CPU; cargamos sin 4-bit.
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model)

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer, False
