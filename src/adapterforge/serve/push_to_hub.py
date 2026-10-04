"""Publica el adapter en Hugging Face Hub con model card renderizada.

Renderiza ``serve/templates/model_card.md`` con las métricas del run
(``metrics.json``) y los resultados del benchmark (``results.json`` de
domain_bench, opcional), la escribe como ``README.md`` del adapter y sube la
carpeta completa al repo del Hub.

``huggingface_hub`` se importa de forma perezosa. El token se toma de
``--token`` o, mejor (en la línea de comandos queda en el historial y en
``ps``), de ``AF_HF_TOKEN``/``HF_TOKEN`` en el entorno o en el ``.env``
(ver :func:`adapterforge.settings.resolve_hf_token`).

Uso::

    python -m adapterforge.serve.push_to_hub \
        --adapter-dir artifacts/qlora_qwen3-4b/adapter \
        --metrics artifacts/qlora_qwen3-4b/metrics.json \
        --bench-results artifacts/eval/adapter.json \
        --repo-id Jaisato/juristune-qwen3-4b-lora

TODO(jairo): subir también el dataset con dataset card (licencia BOE
reutilizable con cita) y el GGUF cuantizado como repo aparte.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from adapterforge.settings import get_settings, resolve_hf_token

_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_0-9]+)\s*\}\}")
DEFAULT_TEMPLATE = Path("serve/templates/model_card.md")

# Lo que ``trainer.save_model`` deja junto al adapter y no debe publicarse.
# ``training_args.bin`` es el pickle completo de la configuración del Trainer
# (incluye ``hub_token`` si se pasó y rutas locales); el resto son estados de
# optimizador/scheduler/RNG que sólo sirven para reanudar en esta máquina.
UPLOAD_IGNORE_PATTERNS: tuple[str, ...] = (
    "training_args.bin",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "rng_state*",
    "checkpoint-*",
    ".env",
)


def render_model_card(template_text: str, context: dict[str, Any]) -> str:
    """Sustituye ``{{claves}}`` del template; falla si falta alguna."""
    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            missing.append(key)
            return match.group(0)
        return str(context[key])

    rendered = _PLACEHOLDER.sub(_replace, template_text)
    if missing:
        raise ValueError(f"Faltan claves en el contexto de la model card: {sorted(set(missing))}")
    return rendered


def build_card_context(
    metrics: dict[str, Any], bench_results: dict[str, Any] | None
) -> dict[str, Any]:
    """Prepara el contexto de la model card a partir de las métricas."""
    summary = (bench_results or {}).get("summary", {})
    by_type = summary.get("by_type", {})
    return {
        "experiment": metrics.get("experiment", "juristune"),
        "base_model": metrics.get("base_model", "?"),
        "phase": metrics.get("phase", "sft"),
        "steps": metrics.get("steps", "?"),
        "train_loss_initial": metrics.get("train_loss_initial", "?"),
        "train_loss_final": metrics.get("train_loss_final", "?"),
        "lora_r": metrics.get("lora_r", "?"),
        "seq_len": metrics.get("seq_len", "?"),
        "used_unsloth": metrics.get("used_unsloth", False),
        "bench_overall": summary.get("overall", "pendiente"),
        "bench_extract": by_type.get("extract", "pendiente"),
        "bench_qa": by_type.get("qa", "pendiente"),
        "bench_resumen": by_type.get("resumen", "pendiente"),
        "timestamp": metrics.get("timestamp", "?"),
    }


def push(
    adapter_dir: str | Path,
    repo_id: str,
    *,
    metrics_path: str | Path,
    bench_results_path: str | Path | None = None,
    template_path: str | Path = DEFAULT_TEMPLATE,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Renderiza la card, crea el repo si no existe y sube el adapter."""
    from huggingface_hub import HfApi  # perezoso

    adapter_path = Path(adapter_dir)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"No existe el directorio del adapter: {adapter_path}")

    resolved_token = resolve_hf_token(token)
    if not resolved_token:
        raise RuntimeError(
            "No hay token de Hugging Face: define HF_TOKEN o AF_HF_TOKEN (entorno o "
            ".env) o usa --token. El token necesita permiso de escritura sobre el "
            "namespace Jaisato."
        )

    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    bench_results = None
    if bench_results_path:
        bench_results = json.loads(Path(bench_results_path).read_text(encoding="utf-8"))

    template_text = Path(template_path).read_text(encoding="utf-8")
    card = render_model_card(template_text, build_card_context(metrics, bench_results))
    (adapter_path / "README.md").write_text(card, encoding="utf-8")

    api = HfApi(token=resolved_token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=list(UPLOAD_IGNORE_PATTERNS),
        # ``ignore_patterns`` solo evita subidas nuevas: borra también en el
        # mismo commit las copias que una publicación anterior dejara en el Hub.
        delete_patterns=list(UPLOAD_IGNORE_PATTERNS),
        commit_message=f"Publica adapter {metrics.get('experiment', '')} "
        f"({metrics.get('phase', 'sft')}, {metrics.get('steps', '?')} steps)",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"[push_to_hub] adapter publicado en {url}")
    return url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publicación del adapter en HF Hub")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--repo-id", default=None, help="Default: AF_HF_REPO_ID")
    parser.add_argument("--metrics", required=True, help="metrics.json del run")
    parser.add_argument("--bench-results", default=None, help="results.json de domain_bench")
    parser.add_argument("--card-template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument(
        "--token",
        default=None,
        help="Desaconsejado (queda en el historial y en `ps`): mejor HF_TOKEN o "
        "AF_HF_TOKEN en el entorno o en .env",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)

    push(
        args.adapter_dir,
        args.repo_id or get_settings().hf_repo_id,
        metrics_path=args.metrics,
        bench_results_path=args.bench_results,
        template_path=args.card_template,
        token=args.token,
        private=args.private,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
