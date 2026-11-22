"""Síntesis online: un modelo *teacher* responde sobre la ventana anclada.

El teacher (litellm o endpoint OpenAI-compatible, vía
:mod:`adapterforge.llm`, importado de forma perezosa) ve la MISMA ventana
estructural que se guarda como ``input``; sus salidas se validan contra el
contrato de cada plantilla (``PREGUNTA:``/``RESPUESTA:`` en ``qa``, sin
boilerplate en el resto) y las que no lo cumplen se descartan y se cuentan.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from adapterforge.data.budget import TokenCounter
from adapterforge.data.extract import RawDocument, split_articles
from adapterforge.data.synthesize.examples import DEFAULT_WINDOW_TOKENS, SftExample, _example_rng
from adapterforge.data.synthesize.prompts import (
    _LLANO_INSTRUCTIONS,
    _OBLIGACIONES_JSON_INSTRUCTION,
    _RESUMEN_INSTRUCTIONS_ESTRUCTURADO,
    load_prompt_template,
    render_prompt,
)
from adapterforge.data.synthesize.text import _short_ref
from adapterforge.data.synthesize.window import compose_window

# Tolerante a negritas markdown: «**PREGUNTA:** ...» también cumple contrato.
_TEACHER_QA_BLOCKS = re.compile(
    r"\*{0,2}PREGUNTA\*{0,2}\s*:\s*(.+?)\s*\*{0,2}RESPUESTA\*{0,2}\s*:\s*(.+)",
    re.DOTALL,
)
_TEACHER_BOILERPLATE = re.compile(r"\b(PREGUNTA|RESPUESTA)\s*:")

# Instrucción almacenada para cada tarea online: debe corresponder al formato
# que la plantilla del teacher EXIGE (ficha para resumen, JSON para
# obligaciones). Emparejar una instrucción de prosa con una respuesta en
# ficha enseñaría al modelo a ignorar el formato pedido.
_ONLINE_INSTRUCTION_POOLS: dict[str, tuple[str, ...]] = {
    "resumen": _RESUMEN_INSTRUCTIONS_ESTRUCTURADO,
    "obligaciones": (_OBLIGACIONES_JSON_INSTRUCTION,),
}


def _online_instruction(task: str, doc: RawDocument, rng: random.Random) -> str | None:
    """Instrucción coherente con el contrato de la plantilla, o None."""
    if task == "llano":
        template = _LLANO_INSTRUCTIONS[rng.randrange(len(_LLANO_INSTRUCTIONS))]
        return template.format(referencia=_short_ref(doc))
    pool = _ONLINE_INSTRUCTION_POOLS.get(task)
    if pool is None:
        return None
    return pool[rng.randrange(len(pool))]


def parse_teacher_qa(raw: str) -> tuple[str, str] | None:
    """'PREGUNTA: ... RESPUESTA: ...' -> (pregunta, respuesta) o None."""
    match = _TEACHER_QA_BLOCKS.search(raw)
    if not match:
        return None
    question = " ".join(match.group(1).split()).strip("* ")
    answer = match.group(2).strip().strip("*").strip()
    if not question or not answer or _TEACHER_BOILERPLATE.search(answer):
        return None
    return question, answer


def synthesize_online(
    docs: list[RawDocument],
    tasks: tuple[str, ...],
    *,
    model: str,
    backend: str,
    api_base: str | None,
    api_key: str | None,
    prompts_dir: str | Path,
    counter: TokenCounter | None = None,
    token_budget: int = DEFAULT_WINDOW_TOKENS,
    seed: int = 42,
) -> list[SftExample]:
    """Genera ejemplos preguntando al teacher sobre la MISMA ventana anclada.

    El teacher ve la ventana estructural (no el documento completo): así el
    ejemplo resultante respeta el presupuesto de tokens y el input guardado
    coincide exactamente con lo que el teacher leyó. Las salidas de la tarea
    ``qa`` se parsean estructuralmente (PREGUNTA:/RESPUESTA:) y las que no
    cumplen el contrato se descartan (contadas en el log).
    """
    from adapterforge.llm import LlmError, chat_complete  # perezoso

    counter = counter or TokenCounter()
    examples: list[SftExample] = []
    rejected = 0
    errors = 0
    for task in tasks:
        template = load_prompt_template(prompts_dir, task)
        for index, doc in enumerate(docs, start=1):
            _, articles = split_articles(doc.texto)
            # compose_window recibe ÍNDICES en ``articles``, no números de
            # artículo: con los números (1..N) se saltaba siempre el índice 0
            # (el artículo 1, el «Objeto») y, con numeración repetida (normas
            # que modifican otras), metía en la ventana artículos equivocados.
            needed = list(range(len(articles)))
            window = compose_window(doc, articles, needed, counter, token_budget)
            prompt = render_prompt(template, doc, texto=window)
            try:
                response = chat_complete(
                    [{"role": "user", "content": prompt}],
                    model,
                    backend=backend,
                    api_base=api_base,
                    api_key=api_key,
                    temperature=0.4,
                    max_tokens=1200,
                )
            except LlmError as exc:
                # Un fallo transitorio no debe tirar el lote entero.
                errors += 1
                print(f"[synthesize] error del teacher en {doc.id} ({exc}); se omite")
                continue
            if index % 25 == 0:
                print(f"[synthesize] {task}: {index}/{len(docs)} documentos", flush=True)
            if task == "qa":
                parsed = parse_teacher_qa(response)
                if parsed is None:
                    rejected += 1
                    continue
                instruction, response = parsed
            else:
                if _TEACHER_BOILERPLATE.search(response):
                    rejected += 1
                    continue
                rng = _example_rng(seed, doc.id, "online", task)
                instruction = _online_instruction(task, doc, rng)
                if instruction is None:
                    rejected += 1  # tarea sin instrucción coherente: se omite
                    continue
            examples.append(
                SftExample(
                    id=f"{doc.id}-{task}",
                    doc_id=doc.id,
                    task=task,
                    instruction=instruction,
                    input=window,
                    response=response,
                    source=model,
                )
            )
    if rejected or errors:
        print(
            f"[synthesize] teacher: {rejected} salidas descartadas (contrato), "
            f"{errors} errores de llamada omitidos"
        )
    return examples
