"""Pares de preferencia (DPO) offline.

Toma como *chosen* las respuestas de los generadores offline de resumen y
obligaciones y construye el *rejected* degradándolas de tres formas
rotatorias: vaguedad, copia literal del documento y respuesta sin
estructura ni cita. ``TODO(jairo)``: pares online (teacher fuerte vs débil).
"""

from __future__ import annotations

import random

from adapterforge.data.budget import TokenCounter
from adapterforge.data.extract import RawDocument, split_articles
from adapterforge.data.synthesize.detectors import find_obligation_sentences
from adapterforge.data.synthesize.examples import (
    DEFAULT_WINDOW_TOKENS,
    DpoPair,
    SftExample,
    SynthContext,
)
from adapterforge.data.synthesize.offline import offline_obligaciones, offline_resumen
from adapterforge.data.synthesize.text import _first_sentence


def _degrade_vague(doc: RawDocument) -> str:
    return (
        "La disposición analizada regula diversas cuestiones de interés general. "
        "En términos generales, establece una serie de obligaciones y plazos para "
        "los sujetos afectados, así como un posible régimen sancionador. Para más "
        "detalle se recomienda consultar el texto completo publicado en el boletín "
        "oficial correspondiente."
    )


def _degrade_copy(doc: RawDocument) -> str:
    words = doc.texto.split()
    return " ".join(words[:60]) + " (...)"


def _degrade_unstructured(doc: RawDocument) -> str:
    obligations = find_obligation_sentences(doc.texto, limit=2)
    base = " ".join(obligations) if obligations else _first_sentence(doc.texto)
    return (
        f"Básicamente la norma dice que {base.lower()} Y también hay plazos y "
        "puede haber multas si no se cumple, aunque depende de cada caso."
    )


_DEGRADERS = (_degrade_vague, _degrade_copy, _degrade_unstructured)


def build_dpo_offline(
    docs: list[RawDocument],
    seed: int = 42,
    n_pairs: int = 15,
    counter: TokenCounter | None = None,
    token_budget: int = DEFAULT_WINDOW_TOKENS,
) -> list[DpoPair]:
    """Construye pares chosen/rejected degradando la respuesta buena."""
    counter = counter or TokenCounter()
    ctx = SynthContext(counter=counter, token_budget=token_budget, seed=seed)
    rng = random.Random(seed)
    candidates: list[SftExample] = []
    for doc in docs:
        _, articles = split_articles(doc.texto)
        candidates.extend(offline_resumen(doc, articles, ctx))
        candidates.extend(offline_obligaciones(doc, articles, ctx))
    rng.shuffle(candidates)
    pairs: list[DpoPair] = []
    docs_by_id = {doc.id: doc for doc in docs}
    for index, example in enumerate(candidates[:n_pairs]):
        degrader = _DEGRADERS[index % len(_DEGRADERS)]
        pairs.append(
            DpoPair(
                id=f"{example.id}-dpo",
                doc_id=example.doc_id,
                task=example.task,
                instruction=example.instruction,
                input=example.input,
                chosen=example.response,
                rejected=degrader(docs_by_id[example.doc_id]),
            )
        )
    return pairs
