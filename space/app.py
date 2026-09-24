"""Demo Gradio de JurisTune en CPU (HF Space, tier gratuito).

Descarga el GGUF Q4_K_M publicado en el Hub de Jaisato y lo sirve con
llama-cpp-python. Sin GPU: pensado para contextos cortos y demos guiadas.
"""

from __future__ import annotations

import os

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Configurables desde los settings del Space (Variables), sin tocar código.
GGUF_REPO = os.environ.get("AF_GGUF_REPO", "Jaisato/juristune-qwen3-4b-gguf")
GGUF_FILE = os.environ.get("AF_GGUF_FILE", "juristune-q4_k_m.gguf")
N_CTX = int(os.environ.get("AF_N_CTX", "2048"))

SYSTEM_PROMPT = (
    "Eres JurisTune, un asistente especializado en normativa española y de la "
    "UE. Respondes en español, con rigor y estructura, anclándote al documento "
    "aportado y citando la fuente con el formato [ID, art. N]. Si el documento "
    "no contiene la respuesta, lo dices explícitamente."
)

EXAMPLES = [
    "Resume de forma estructurada: Artículo 3. Obligaciones de registro. Las "
    "empresas deberán implantar un sistema de registro de jornada digital e "
    "interoperable antes del 1 de octubre de 2026. Los registros deberán "
    "conservarse durante cuatro años. (BOE-A-2026-05733)",
    "Extrae en JSON las obligaciones, plazos y sanciones: Artículo 5. El "
    "incumplimiento podrá sancionarse con multa de hasta 7.500 euros por "
    "centro de trabajo. (BOE-A-2026-05733)",
    "Explica en lenguaje llano qué implica un plazo de justificación de seis "
    "meses desde el cobro de una subvención.",
]

print(f"[space] descargando {GGUF_REPO}/{GGUF_FILE}...")
model_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
llm = Llama(model_path=model_path, n_ctx=N_CTX, n_threads=2, verbose=False)
print("[space] modelo cargado.")


def _text(content: str | list | dict) -> str:
    """Texto plano de un mensaje del historial de Gradio.

    Gradio 6 normaliza el historial: `content` llega como lista de partes
    (``{"type": "text", "text": ...}``) en vez de como cadena. El chat template
    del GGUF espera cadenas; una lista acabaría como contenido vacío o como su
    repr, y el modelo perdería los turnos anteriores.
    """
    if isinstance(content, str):
        return content
    parts = content if isinstance(content, list) else [content]
    texts = (p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")
    return "".join(texts)


def respond(message: str, history: list[dict]) -> str:
    """Genera la respuesta del adapter con el historial de la sesión."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": _text(turn["content"])})
    messages.append({"role": "user", "content": message})
    result = llm.create_chat_completion(
        messages=messages,
        temperature=0.3,
        max_tokens=512,
    )
    return result["choices"][0]["message"]["content"]


# Gradio 6 solo usa el formato "messages" y ya no acepta el parámetro `type`.
# api_name mantiene el endpoint /chat que publicaba Gradio 5 (en Gradio 6 se
# llamaría /respond, como la función), para no romper clientes de gradio_client.
demo = gr.ChatInterface(
    fn=respond,
    api_name="chat",
    title="JurisTune — AdapterForge (demo CPU)",
    description=(
        "Adapter QLoRA jurídico-administrativo entrenado en una GPU de 8 GB. "
        "Pega un extracto de una disposición y pide resumen, extracción JSON o "
        "lenguaje llano. Repositorio: github.com/jairosarabia/adapterforge"
    ),
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch()
