"""Ventana estructural de contexto: qué parte del documento ve el modelo.

Cabecera (con materias y URL, anclables) + los artículos que contienen el
dato, íntegros y en el orden pedido, dentro de un presupuesto de tokens
medido con el tokenizer real (:mod:`adapterforge.data.budget`). Nada de
``texto[:4500]`` cayendo en el preámbulo. El llamador verifica después que
su target aparece literalmente en la ventana devuelta.
"""

from __future__ import annotations

from adapterforge.data.budget import TokenCounter
from adapterforge.data.extract import Article, RawDocument, split_sentences
from adapterforge.data.synthesize.examples import DEFAULT_WINDOW_TOKENS


def _fit_sentences(text: str, counter: TokenCounter, token_budget: int) -> str:
    """Prefijo del texto a nivel de frase que cabe en el presupuesto."""
    if counter.count(text) <= token_budget:
        return text
    kept: list[str] = []
    used = 0
    for sentence in split_sentences(text):
        cost = counter.count(sentence) + 1
        if used + cost > token_budget:
            break
        kept.append(sentence)
        used += cost
    return " ".join(kept)


def _article_block(article: Article) -> str:
    head = f"Artículo {article.number}." + (f" {article.heading}" if article.heading else "")
    return f"{head}\n{article.body}".strip()


def _window_header(doc: RawDocument) -> str:
    """Cabecera de la ventana: incluye materias y URL para poder anclarlos."""
    return (
        f"{doc.header()}\n"
        f"Materias: {', '.join(doc.materia)} · Texto oficial: {doc.url}"
    )


def compose_window(
    doc: RawDocument,
    articles: list[Article],
    needed: list[int],
    counter: TokenCounter,
    token_budget: int = DEFAULT_WINDOW_TOKENS,
) -> str:
    """Cabecera + artículos indicados (por ÍNDICE), dentro del presupuesto.

    ``needed`` contiene índices en ``articles`` (no números de artículo: los
    BOE que modifican otras normas repiten números). Los artículos se añaden
    ÍNTEGROS en el orden pedido mientras quepan; si ni el primero cabe, se
    recorta a nivel de frase. El llamador es responsable de verificar que su
    target aparece en la ventana devuelta (anclaje garantizado).
    """
    header = _window_header(doc)
    budget = token_budget - counter.count(header)
    valid = [i for i in dict.fromkeys(needed) if 0 <= i < len(articles)]
    blocks: list[str] = []
    for index in valid:
        block = _article_block(articles[index])
        cost = counter.count(block) + 2
        if cost <= budget:
            blocks.append(block)
            budget -= cost
    if not blocks:
        # Ni el primer artículo necesario cabe entero: recorte por frases.
        first = articles[valid[0]] if valid else (articles[0] if articles else None)
        if first is not None:
            head = f"Artículo {first.number}." + (
                f" {first.heading}" if first.heading else ""
            )
            body = _fit_sentences(
                first.body, counter, max(budget - counter.count(head) - 2, 0)
            )
            if body:
                blocks.append(f"{head}\n{body}")
    if not blocks:
        return header
    return header + "\n\n" + "\n\n".join(blocks)


def _doc_input(doc: RawDocument) -> str:
    """Bloque de documento completo (solo para compatibilidad/tests)."""
    return f"{doc.header()}\n\n{doc.texto}"
