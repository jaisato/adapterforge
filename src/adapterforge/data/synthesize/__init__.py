"""Síntesis de pares instrucción->respuesta anclados al documento.

Dos modos:

- **Online** (por defecto): un modelo *teacher* (configurable vía litellm o
  endpoint OpenAI-compatible) genera la respuesta a partir de las plantillas
  versionadas en ``prompts/``. Anclar cada ejemplo al documento reduce la
  alucinación del teacher.
- **Offline** (``--offline``): generadores determinísticos que construyen la
  respuesta a partir de la estructura del propio documento (artículos, frases
  con obligaciones, plazos y sanciones). Sin API ni red: sirve para generar el
  dataset de muestra, para los tests y para entornos sin conectividad.

Principios v3 (lecciones de la autopsia de v1/v2/v2b):

- **Ventana estructural, no prefijo**: los detectores corren sobre el
  articulado completo y el input adjunta solo la cabecera + los artículos que
  contienen el dato, dentro de un presupuesto de tokens medido con el
  tokenizer real (:mod:`adapterforge.data.budget`). Nada de
  ``texto[:4500]`` cayendo en el preámbulo.
- **Anclaje garantizado**: ningún ejemplo se emite si su valor objetivo no
  aparece literalmente en la ventana que ve el modelo.
- **Etiquetas validadas**: un "plazo" debe ser una duración o fecha real
  (número + unidad temporal, o fecha), una "multa" debe llevar cuantía; las
  capturas desbordadas y los «(no se detectan)» se descartan.
- **Diversidad**: pools de instrucciones amplios muestreados por ejemplo (no
  por documento), variantes con y sin cita, y variantes JSON estilo
  benchmark («Responde únicamente con un objeto JSON con la clave X»).
- **Mezcla equilibrada**: ``--balance`` re-muestrea por tarea hacia la
  distribución del benchmark (extract/qa/resumen dominan).

Tareas soportadas (una plantilla por tarea en ``prompts/``):

- ``resumen``: prosa de un párrafo (mayoría) o ficha estructurada.
- ``qa``: preguntas/respuestas por artículo con cita [BOE-A-..., art. N].
- ``obligaciones``: extracción de obligaciones/plazos/sanciones (JSON/lista/prosa).
- ``llano``: reescritura en lenguaje llano para no juristas.
- ``extraccion``: dato puntual con respuesta corta (frase o JSON verbatim).

Mapa del paquete (una responsabilidad por módulo; este ``__init__`` reexporta
la API pública, así que ``from adapterforge.data.synthesize import X`` sigue
funcionando igual que cuando todo vivía en ``synthesize.py``):

- :mod:`.examples`: modelos ``SftExample``/``DpoPair``, ``SynthContext``,
  constantes de mezcla y presupuesto, RNG deterministas.
- :mod:`.text`: redacción en castellano (fechas, artículos gramaticales,
  glosario llano, fronteras de frase).
- :mod:`.detectors`: detectores estructurales validados (plazos y su
  propósito, obligaciones, sanciones, cuantías, tramos).
- :mod:`.window`: ventana estructural de contexto dentro del presupuesto.
- :mod:`.prompts`: pools de instrucciones y plantillas del teacher.
- :mod:`.offline`: generadores determinísticos por tarea, ``rebalance`` y
  ``build_sft_offline``.
- :mod:`.dpo`: pares chosen/rejected offline (``build_dpo_offline``).
- :mod:`.online`: teacher vía litellm/OpenAI-compatible (``synthesize_online``).
- :mod:`.cli`: ``main`` (``python -m adapterforge.data.synthesize``).

Uso CLI::

    python -m adapterforge.data.synthesize \\
        --input data/interim/documents.jsonl \\
        --output data/interim/sft_raw.jsonl --offline

TODO(jairo): variante Evol-Instruct (mutar instrucciones) y self-instruct con
semillas humanas para aumentar diversidad del dataset online.
"""

from __future__ import annotations

from adapterforge.data.synthesize.cli import main
from adapterforge.data.synthesize.detectors import (
    OBLIGATION_MARKERS,
    SANCTION_MARKERS,
    DeadlineHit,
    SentenceHit,
    TierHit,
    classify_deadline,
    deadline_purpose,
    extract_deadline_value,
    find_amount_hits,
    find_deadline_hits,
    find_deadlines,
    find_obligation_hits,
    find_obligation_sentences,
    find_sanction_hits,
    find_sanction_sentences,
    find_sanction_tiers,
    max_amount_in,
)

# Nombres privados que los tests importaban del módulo original: se mantienen
# accesibles desde el paquete para no romper ``from ... import _x``.
from adapterforge.data.synthesize.detectors import _amount_value as _amount_value
from adapterforge.data.synthesize.detectors import _future_date as _future_date
from adapterforge.data.synthesize.detectors import _trim_purpose as _trim_purpose
from adapterforge.data.synthesize.dpo import build_dpo_offline
from adapterforge.data.synthesize.examples import (
    DEFAULT_BALANCE,
    DEFAULT_WINDOW_TOKENS,
    SFT_TASKS,
    DpoPair,
    SftExample,
    SynthContext,
)
from adapterforge.data.synthesize.offline import (
    build_sft_offline,
    offline_extraccion,
    offline_llano,
    offline_obligaciones,
    offline_qa,
    offline_resumen,
    rebalance,
)
from adapterforge.data.synthesize.online import parse_teacher_qa, synthesize_online
from adapterforge.data.synthesize.prompts import load_prompt_template, render_prompt
from adapterforge.data.synthesize.text import PLAIN_GLOSSARY, format_date_es, simplify_text
from adapterforge.data.synthesize.text import _first_sentence as _first_sentence
from adapterforge.data.synthesize.window import compose_window

__all__ = [
    "DEFAULT_BALANCE",
    "DEFAULT_WINDOW_TOKENS",
    "OBLIGATION_MARKERS",
    "PLAIN_GLOSSARY",
    "SANCTION_MARKERS",
    "SFT_TASKS",
    "DeadlineHit",
    "DpoPair",
    "SentenceHit",
    "SftExample",
    "SynthContext",
    "TierHit",
    "build_dpo_offline",
    "build_sft_offline",
    "classify_deadline",
    "compose_window",
    "deadline_purpose",
    "extract_deadline_value",
    "find_amount_hits",
    "find_deadline_hits",
    "find_deadlines",
    "find_obligation_hits",
    "find_obligation_sentences",
    "find_sanction_hits",
    "find_sanction_sentences",
    "find_sanction_tiers",
    "format_date_es",
    "load_prompt_template",
    "main",
    "max_amount_in",
    "offline_extraccion",
    "offline_llano",
    "offline_obligaciones",
    "offline_qa",
    "offline_resumen",
    "parse_teacher_qa",
    "rebalance",
    "render_prompt",
    "simplify_text",
    "synthesize_online",
]
