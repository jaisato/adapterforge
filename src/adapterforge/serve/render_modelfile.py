"""Renderiza el Modelfile de Ollama desde ``serve/ollama/Modelfile.j2``.

La plantilla usa sintaxis ``{{ variable }}`` (subconjunto de Jinja2, resuelto
con una sustitución propia para no añadir jinja2 a las dependencias core).

El TEMPLATE de chat (ChatML/Qwen3) se inyecta desde la constante
``OLLAMA_CHAT_TEMPLATE`` — así el Go-template de Ollama (``{{ .Messages }}``,
``{{ end }}``...) nunca pasa por el motor de sustitución propio y el formato
de prompt queda fijado explícitamente en vez de heredarse del template
embebido en el GGUF (que en los adapters mergeados era el de la variante
thinking, incorrecta para Qwen3-*-Instruct-2507).

Uso::

    python -m adapterforge.serve.render_modelfile \
        --gguf artifacts/gguf/juristune-q4_k_m.gguf \
        --output artifacts/gguf/Modelfile

    # y después:  ollama create juristune -f artifacts/gguf/Modelfile
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from adapterforge.data.format import SYSTEM_PROMPT

_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_0-9]+)\s*\}\}")
DEFAULT_TEMPLATE = Path("serve/ollama/Modelfile.j2")

# Template de chat de Ollama (Go template) para Qwen3-Instruct: ChatML con
# soporte de .Messages y fallback legacy .System/.Prompt. Sin rama <think>:
# la 2507 es la variante no pensante.
OLLAMA_CHAT_TEMPLATE = (
    "{{- if .Messages }}\n"
    "{{- range .Messages }}<|im_start|>{{ .Role }}\n"
    "{{ .Content }}<|im_end|>\n"
    "{{ end }}<|im_start|>assistant\n"
    "{{ else -}}\n"
    "{{ if .System }}<|im_start|>system\n"
    "{{ .System }}<|im_end|>\n"
    "{{ end }}{{ if .Prompt }}<|im_start|>user\n"
    "{{ .Prompt }}<|im_end|>\n"
    "{{ end }}<|im_start|>assistant\n"
    "{{ end -}}"
)


def render_modelfile(
    template_text: str,
    *,
    gguf_path: str,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.3,
    num_ctx: int = 4096,
    repeat_penalty: float = 1.0,
) -> str:
    """Sustituye las variables de la plantilla; falla si queda alguna."""
    context = {
        "gguf_path": gguf_path,
        "system_prompt": system_prompt,
        "temperature": str(temperature),
        "num_ctx": str(num_ctx),
        "repeat_penalty": str(repeat_penalty),
        "chat_template": OLLAMA_CHAT_TEMPLATE,
    }
    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            missing.append(key)
            return match.group(0)
        return context[key]

    rendered = _PLACEHOLDER.sub(_replace, template_text)
    if missing:
        raise ValueError(f"Variables sin valor en la plantilla: {sorted(set(missing))}")
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render del Modelfile de Ollama")
    parser.add_argument("--gguf", required=True, help="Ruta al GGUF cuantizado")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.0,
        help="1.0 por defecto: una penalización >1 castiga el citado textual "
        "que exigen las tareas de extracción",
    )
    parser.add_argument("--system-file", default=None, help="Fichero con system prompt alternativo")
    args = parser.parse_args(argv)

    system_prompt = SYSTEM_PROMPT
    if args.system_file:
        system_prompt = Path(args.system_file).read_text(encoding="utf-8").strip()

    template_text = Path(args.template).read_text(encoding="utf-8")
    rendered = render_modelfile(
        template_text,
        gguf_path=str(Path(args.gguf).resolve()),
        system_prompt=system_prompt,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        repeat_penalty=args.repeat_penalty,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"[render_modelfile] Modelfile escrito en {output_path}")
    print(f"  Siguiente paso:  ollama create juristune -f {output_path}")
    print("  Verifica el template con:  ollama show <modelo> --modelfile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
