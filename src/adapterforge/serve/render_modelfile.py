"""Renderiza el Modelfile de Ollama desde ``serve/ollama/Modelfile.j2``.

La plantilla usa sintaxis ``{{ variable }}`` (subconjunto de Jinja2, resuelto
con una sustitución propia para no añadir jinja2 a las dependencias core).

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


def render_modelfile(
    template_text: str,
    *,
    gguf_path: str,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.3,
    num_ctx: int = 4096,
) -> str:
    """Sustituye las variables de la plantilla; falla si queda alguna."""
    context = {
        "gguf_path": gguf_path,
        "system_prompt": system_prompt,
        "temperature": str(temperature),
        "num_ctx": str(num_ctx),
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
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"[render_modelfile] Modelfile escrito en {output_path}")
    print(f"  Siguiente paso:  ollama create juristune -f {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
