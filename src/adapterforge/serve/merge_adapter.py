"""Fusiona el adapter LoRA con la base en FP16/BF16 (paso previo a GGUF).

El merge se hace en CPU (con 64 GB de RAM sobra para modelos de 3-8B); la
GPU no participa. La salida es un modelo "completo" en safetensors que
``serve/export_gguf.sh`` convierte y cuantiza con llama.cpp.

Uso::

    python -m adapterforge.serve.merge_adapter \
        --base Qwen/Qwen3-4B-Instruct-2507 \
        --adapter artifacts/qlora_qwen3-4b/adapter \
        --output artifacts/merged/juristune-qwen3-4b
"""

from __future__ import annotations

import argparse
from pathlib import Path


def merge(base_model: str, adapter_dir: str, output_dir: str) -> Path:
    """Carga base + adapter, fusiona y guarda en ``output_dir`` (lazy)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[merge] cargando base {base_model} en CPU (esto usa RAM, no VRAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print(f"[merge] aplicando adapter {adapter_dir}...")
    merged = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    merged.save_pretrained(str(output_path), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(str(output_path))
    print(f"[merge] modelo fusionado en {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge de adapter LoRA con la base")
    parser.add_argument("--base", required=True, help="Modelo base (HF Hub o ruta local)")
    parser.add_argument("--adapter", required=True, help="Directorio del adapter entrenado")
    parser.add_argument("--output", required=True, help="Directorio del modelo fusionado")
    args = parser.parse_args(argv)
    merge(args.base, args.adapter, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
