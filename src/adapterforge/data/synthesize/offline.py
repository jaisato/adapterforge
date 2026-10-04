"""Generadores offline (determinísticos, sin LLM) y mezcla del dataset SFT.

Un generador por tarea (``resumen``, ``qa``, ``obligaciones``, ``llano``,
``extraccion``) construye la respuesta a partir de la estructura del propio
documento: detectores (:mod:`.detectors`) sobre el articulado completo,
ventana anclada (:mod:`.window`) e instrucciones muestreadas por ejemplo
(:mod:`.prompts`). :func:`build_sft_offline` orquesta las tareas, aplica
``--max-per-task`` y re-muestrea con :func:`rebalance` hacia la mezcla
objetivo. Toda la aleatoriedad sale de RNG sembrados por ejemplo: la misma
entrada produce byte a byte el mismo dataset.
"""

from __future__ import annotations

import json
import random
import re

from adapterforge.data.budget import TokenCounter
from adapterforge.data.extract import Article, RawDocument, split_articles, split_sentences
from adapterforge.data.synthesize.detectors import (
    _EMBEDDED_LIST_MARKER,
    _amount_value,
    _future_date,
    deadline_purpose,
    find_amount_hits,
    find_deadline_hits,
    find_obligation_hits,
    find_sanction_hits,
    find_sanction_tiers,
    max_amount_in,
)
from adapterforge.data.synthesize.examples import (
    DEFAULT_WINDOW_TOKENS,
    SFT_TASKS,
    SftExample,
    SynthContext,
    _example_rng,
)
from adapterforge.data.synthesize.prompts import (
    _EXTRACCION_JSON_INSTR,
    _EXTRACCION_OBLIGACION_VARIANTS,
    _EXTRACCION_PLAZO_VARIANTS,
    _EXTRACCION_SANCION_VARIANTS,
    _FECHA_STEMS,
    _LLANO_INSTRUCTIONS,
    _MULTA_GLOBAL_STEMS,
    _MULTA_GLOBAL_VARIANTS,
    _OBLIGACIONES_JSON_INSTRUCTION,
    _PLAZO_STEMS,
    _QA_INSTRUCTIONS,
    _RESUMEN_INSTRUCTIONS_ESTRUCTURADO,
    _RESUMEN_INSTRUCTIONS_PROSA,
    _disambiguated_plazo_stems,
    _disambiguated_plazo_variants,
    _tier_multa_stems,
    _tier_multa_variants,
)
from adapterforge.data.synthesize.text import (
    _RANGO_ARTICLE,
    _ends_sentence,
    _first_sentence,
    _organ_with_article,
    _short_ref,
    _valid_objeto,
    format_date_es,
    simplify_text,
)
from adapterforge.data.synthesize.window import _article_block, compose_window

# «El presente real decreto tiene por objeto ...» -> apertura en voz activa
# al estilo de las referencias del bench («El Real Decreto X tiene por
# objeto regular ...» en lugar de «X es una disposición de rango ...»).
_OBJETO_BOILERPLATE = re.compile(
    r"^(?:el|la|este|esta)\s+presente\s+[\wáéíóúñ \-]+?\s+tiene\s+por\s+objeto\s+",
    re.IGNORECASE,
)


def _resumen_opening(doc: RawDocument, objeto: str) -> str:
    """Primera frase del resumen en prosa, en voz activa y sin metadatos."""
    short = _short_ref(doc)
    short_cap = short[0].upper() + short[1:]
    stripped = _OBJETO_BOILERPLATE.sub("", objeto).strip()
    if stripped and stripped != objeto:
        return f"{short_cap} tiene por objeto {stripped}"
    if objeto:
        return f"{short_cap} regula lo siguiente: {objeto}"
    return (
        f"{short_cap} es una disposición publicada por "
        f"{_organ_with_article(doc.organo_emisor)}."
    )


def offline_resumen(
    doc: RawDocument, articles: list[Article], ctx: SynthContext
) -> list[SftExample]:
    """Resumen: prosa de un párrafo (mayoría) o ficha estructurada."""
    rng = _example_rng(ctx.seed, doc.id, "resumen")
    obligation_hits = find_obligation_hits(articles, limit=3)
    deadline_hits = find_deadline_hits(articles, limit=4)
    sanction_hits = find_sanction_hits(articles, limit=1)
    needed = sorted(
        {0}
        | {h.article for h in obligation_hits}
        | {h.article for h in deadline_hits}
        | {h.article for h in sanction_hits}
    )
    window = compose_window(doc, articles, needed, ctx.counter, ctx.token_budget)
    # Anclaje: solo se citan obligaciones/plazos/sanciones de la ventana.
    obligations = [h.sentence for h in obligation_hits if h.sentence in window]
    deadlines = [h.value for h in deadline_hits if h.value in window]
    sanctions = [h.sentence for h in sanction_hits if h.sentence in window]
    objeto = _first_sentence(articles[0].body) if articles else _first_sentence(doc.texto)
    if not _valid_objeto(objeto) or objeto not in window:
        objeto = ""

    use_prosa = rng.random() < 0.65
    if use_prosa:
        instruction, cite = _RESUMEN_INSTRUCTIONS_PROSA[
            rng.randrange(len(_RESUMEN_INSTRUCTIONS_PROSA))
        ]
        # Estilo de las referencias del bench: párrafo DENSO en voz activa
        # (apertura con la norma como sujeto, obligaciones verbatim como
        # frases propias, sanción y plazos entrelazados), sin metadatos.
        # v6: las predicciones de v4/v5 quedaban un ~20% más cortas que las
        # referencias y ROUGE-L penalizaba el recall — el target incluye
        # ahora hasta 3 obligaciones y la frase sancionadora si existe.
        opening = _resumen_opening(doc, objeto)
        sentences = [opening if _ends_sentence(opening) else f"{opening}."]
        for obligation in obligations[:3]:
            sentences.append(obligation if _ends_sentence(obligation) else f"{obligation}.")
        if sanctions and sanctions[0] not in sentences:
            sanction = sanctions[0]
            sentences.append(sanction if _ends_sentence(sanction) else f"{sanction}.")
        if deadlines:
            sentences.append(f"Los plazos principales son {'; '.join(deadlines)}.")
        if cite:
            sentences.append(f"Fuente: [{doc.id}].")
        response = " ".join(sentences)
    else:
        instruction = _RESUMEN_INSTRUCTIONS_ESTRUCTURADO[
            rng.randrange(len(_RESUMEN_INSTRUCTIONS_ESTRUCTURADO))
        ]
        included_articles = [a for a in articles if _article_block(a) in window]
        key_points = []
        for a in included_articles[1:5]:
            point = _first_sentence(a.body)
            if not _valid_objeto(point):
                continue  # apartados numerales pelados y cortes colgantes fuera
            label = a.heading.rstrip(".") if a.heading else f"Artículo {a.number}"
            key_points.append(f"- Artículo {a.number} ({label}): {point}")
        lines = [
            f"### Resumen estructurado — {doc.short_title()}",
            "",
            f"- **Norma**: {doc.titulo}",
            f"- **Rango**: {doc.rango}",
            f"- **Órgano emisor**: {doc.organo_emisor}",
            f"- **Publicación**: {doc.fecha_publicacion}",
            f"- **Materias**: {', '.join(doc.materia)}",
        ]
        if objeto:
            lines.append(f"- **Objeto**: {objeto}")
        if key_points:
            lines += ["", "**Puntos clave por artículo:**", *key_points]
        if obligations:
            lines += ["", "**Obligaciones destacadas:**"]
            lines += [f"- {sentence}" for sentence in obligations]
        if deadlines:
            lines += ["", f"**Plazos relevantes:** {'; '.join(deadlines)}."]
        lines += ["", f"Fuente: [{doc.id}] {doc.url}"]
        response = "\n".join(lines)

    if not _ends_sentence(response.splitlines()[-1]) and not response.rstrip().endswith(doc.url):
        return []
    return [
        SftExample(
            id=f"{doc.id}-resumen",
            doc_id=doc.id,
            task="resumen",
            instruction=instruction,
            input=window,
            response=response,
            source="offline",
        )
    ]


def offline_qa(
    doc: RawDocument, articles: list[Article], ctx: SynthContext, max_questions: int = 3
) -> list[SftExample]:
    """Q&A por artículo con cita en el formato del portfolio."""
    examples: list[SftExample] = []
    candidates = [(i, a) for i, a in enumerate(articles) if a.body][:max_questions]
    for index, article in candidates:
        rng = _example_rng(ctx.seed, doc.id, "qa", str(article.number))
        window = compose_window(doc, articles, [index], ctx.counter, ctx.token_budget)
        sentences = split_sentences(article.body)
        quoted = " ".join(sentences[:2])
        if quoted not in window:  # anclaje: la cita debe estar en la ventana
            quoted = _first_sentence(article.body)
            if quoted not in window:
                continue
        heading = f" ({article.heading.rstrip('.')})" if article.heading else ""
        reference = _short_ref(doc, preceded_by_de=True)
        response = (
            f"El artículo {article.number}{heading} {reference} establece lo "
            f"siguiente: «{quoted}» [{doc.id}, art. {article.number}]."
        )
        instruction = _QA_INSTRUCTIONS[rng.randrange(len(_QA_INSTRUCTIONS))].format(
            numero=article.number, referencia=reference
        )
        examples.append(
            SftExample(
                id=f"{doc.id}-qa-{article.number}",
                doc_id=doc.id,
                task="qa",
                instruction=instruction,
                input=window,
                response=response,
                source="offline",
            )
        )
    return examples


def offline_obligaciones(
    doc: RawDocument, articles: list[Article], ctx: SynthContext
) -> list[SftExample]:
    """Extracción de obligaciones/plazos/sanciones con formato variable."""
    rng = _example_rng(ctx.seed, doc.id, "obligaciones")
    obligation_hits = find_obligation_hits(articles, limit=5)
    deadline_hits = find_deadline_hits(articles, limit=5)
    sanction_hits = find_sanction_hits(articles, limit=3)
    needed = sorted(
        {h.article for h in obligation_hits}
        | {h.article for h in deadline_hits}
        | {h.article for h in sanction_hits}
    )
    if not needed:
        return []  # sin señal estructural no hay ejemplo (nunca «no se detectan»)
    window = compose_window(doc, articles, needed, ctx.counter, ctx.token_budget)
    obligations = [h.sentence for h in obligation_hits if h.sentence in window]
    deadlines = [h.value for h in deadline_hits if h.value in window]
    sanctions = [h.sentence for h in sanction_hits if h.sentence in window]
    if not (obligations or deadlines or sanctions):
        return []

    fmt = rng.randrange(3)
    if fmt == 0:  # JSON
        instruction = _OBLIGACIONES_JSON_INSTRUCTION
        payload = {
            "obligaciones": obligations,
            "plazos": deadlines,
            "sanciones": sanctions,
            "fuente": doc.id,
        }
        response = json.dumps(payload, ensure_ascii=False, indent=2)
    elif fmt == 1:  # Lista en texto plano (solo secciones con contenido)
        instruction = (
            "Enumera las obligaciones, los plazos y las sanciones que establece "
            "la siguiente disposición, en apartados con viñetas. Si algún "
            "apartado no aplica, omítelo. Cita la fuente al final."
        )
        lines: list[str] = []
        if obligations:
            lines += ["Obligaciones:"] + [f"- {s}" for s in obligations]
        if deadlines:
            lines += ["", "Plazos:"] + [f"- {d}" for d in deadlines]
        if sanctions:
            lines += ["", "Sanciones:"] + [f"- {s}" for s in sanctions]
        lines += ["", f"Fuente: [{doc.id}]"]
        response = "\n".join(line for line in lines).strip()
    else:  # Prosa breve
        instruction = (
            "Describe brevemente, en prosa, qué obligaciones impone esta "
            "disposición, con qué plazos y qué sanciones prevé. Cita la fuente."
        )
        parts = []
        if obligations:
            parts.append(f"La norma obliga, entre otras cosas, a lo siguiente: {obligations[0]}")
        if deadlines:
            parts.append(f"Fija plazos de {', '.join(deadlines[:3])}.")
        if sanctions:
            parts.append(f"En materia sancionadora: {sanctions[0]}")
        parts.append(f"[{doc.id}]")
        response = " ".join(parts)

    return [
        SftExample(
            id=f"{doc.id}-obligaciones",
            doc_id=doc.id,
            task="obligaciones",
            instruction=instruction,
            input=window,
            response=response,
            source="offline",
        )
    ]


def offline_llano(
    doc: RawDocument, articles: list[Article], ctx: SynthContext
) -> list[SftExample]:
    """Explicación en lenguaje llano, en ficha o en prosa corrida (50/50)."""
    rng = _example_rng(ctx.seed, doc.id, "llano")
    obligation_hits = find_obligation_hits(articles, limit=3)
    deadline_hits = find_deadline_hits(articles, limit=3)
    needed = sorted(
        {0} | {h.article for h in obligation_hits} | {h.article for h in deadline_hits}
    )
    window = compose_window(doc, articles, needed, ctx.counter, ctx.token_budget)
    obligations = [simplify_text(h.sentence) for h in obligation_hits if h.sentence in window]
    deadlines = [h.value for h in deadline_hits if h.value in window]
    objeto = _first_sentence(articles[0].body) if articles else ""
    if objeto and (not _valid_objeto(objeto) or objeto not in window):
        objeto = ""

    rango_phrase = _RANGO_ARTICLE.get(
        doc.rango.lower(), f"una norma de rango {doc.rango.lower()}"
    )
    instruction = _LLANO_INSTRUCTIONS[rng.randrange(len(_LLANO_INSTRUCTIONS))].format(
        referencia=_short_ref(doc)
    )

    if rng.random() < 0.5:  # ficha con encabezados
        lines = [
            f"**¿Qué es?** {doc.short_title()} es {rango_phrase} que publicó "
            f"{_organ_with_article(doc.organo_emisor)} el "
            f"{format_date_es(doc.fecha_publicacion)}.",
        ]
        if objeto:
            lines += ["", f"**¿De qué trata?** {simplify_text(objeto)}"]
        if obligations:
            lines += ["", "**¿Qué obligaciones impone?**"]
            lines += [f"- {sentence}" for sentence in obligations]
        if deadlines:
            lines += ["", "**Plazos que conviene apuntar:**"]
            lines += [f"- {deadline}" for deadline in deadlines]
        lines += [
            "",
            "**En resumen:** revisa si estás dentro del ámbito de aplicación y "
            f"anota los plazos. El texto oficial está en {doc.url} [{doc.id}].",
        ]
        response = "\n".join(lines)
    else:  # prosa corrida sin encabezados
        sentences = [
            f"{doc.short_title()} es {rango_phrase} que publicó "
            f"{_organ_with_article(doc.organo_emisor)} el "
            f"{format_date_es(doc.fecha_publicacion)}.",
        ]
        if objeto:
            sentences.append(f"En pocas palabras, {simplify_text(objeto).rstrip('.')}." )
        if obligations:
            sentences.append(
                f"Lo más importante que exige: {obligations[0].rstrip('.').lower()}."
            )
        if deadlines:
            sentences.append(
                f"Apunta estas fechas o plazos: {'; '.join(deadlines)}."
            )
        sentences.append(f"El texto oficial está en {doc.url} [{doc.id}].")
        response = " ".join(sentences)

    return [
        SftExample(
            id=f"{doc.id}-llano",
            doc_id=doc.id,
            task="llano",
            instruction=instruction,
            input=window,
            response=response,
            source="offline",
        )
    ]


def _extraccion_example(
    doc: RawDocument,
    ctx: SynthContext,
    *,
    suffix: str,
    window: str,
    value: str,
    json_field: str,
    json_desc: str,
    prose_variants: tuple[tuple[str, bool], ...],
    prose_stems: tuple[str, ...],
    require_desc: bool = False,
) -> SftExample | None:
    """Ejemplo de extracción (frase corta o JSON verbatim) con anclaje.

    ``require_desc``: con pregunta desambiguada, solo valen las plantillas
    JSON que transportan la descripción (una pregunta genérica con etiqueta
    desambiguada re-crearía la ambigüedad que se quiere eliminar).
    """
    rng = _example_rng(ctx.seed, doc.id, "extraccion", suffix)
    if value not in window:
        return None  # anclaje garantizado: el dato debe estar en la ventana
    if rng.random() < 0.5:  # variante JSON estilo benchmark
        json_pool = (
            tuple(t for t in _EXTRACCION_JSON_INSTR if "{desc}" in t)
            if require_desc
            else _EXTRACCION_JSON_INSTR
        )
        instruction = json_pool[rng.randrange(len(json_pool))].format(
            campo=json_field, desc=json_desc
        )
        response = json.dumps({json_field: value}, ensure_ascii=False)
    else:
        instruction, cite = prose_variants[rng.randrange(len(prose_variants))]
        stem = prose_stems[rng.randrange(len(prose_stems))].format(v=value)
        response = f"{stem} [{doc.id}]." if cite else f"{stem}."
    return SftExample(
        id=f"{doc.id}-extraccion-{suffix}",
        doc_id=doc.id,
        task="extraccion",
        instruction=instruction,
        input=window,
        response=response,
        source="offline",
    )


def offline_extraccion(
    doc: RawDocument, articles: list[Article], ctx: SynthContext
) -> list[SftExample]:
    """Datos puntuales con respuesta corta: SOLO el dato pedido, verbatim.

    Es la tarea que ejercita el domain_bench (extract): plazos (duración y
    fecha futura), multa máxima con cuantía y obligación principal, en
    variantes de frase única (con/sin cita) y de JSON con la clave pedida.
    """
    examples: list[SftExample] = []
    all_deadline_hits = find_deadline_hits(articles, limit=6)
    for i, hit in enumerate(all_deadline_hits[:3]):
        if hit.kind == "fecha" and not _future_date(hit.value, doc):
            continue  # fecha de referencia pre-publicación: no es un plazo
        # Con varios plazos en el documento, la ventana INCLUYE los artículos
        # de los plazos hermanos (el artículo del plazo pedido va primero y
        # sobrevive seguro al presupuesto): el modelo debe aprender a elegir
        # el plazo preguntado con el distractor delante (fallo real de v3 en
        # eb-001: respondió el plazo de justificación cuando se preguntaba
        # por el de solicitud).
        needed = [hit.article] + [
            h.article for h in all_deadline_hits if h.article != hit.article
        ]
        window = compose_window(doc, articles, needed, ctx.counter, ctx.token_budget)
        visible = [h for h in all_deadline_hits if h.value in window]
        purpose = None
        if len(visible) > 1:
            purpose = deadline_purpose(hit, articles)
            if purpose is None:
                continue  # varios plazos visibles y sin propósito: no se emite
            # El propósito debe DISCRIMINAR: si otro plazo visible deriva el
            # mismo propósito (p. ej. dos plazos del mismo artículo que caen
            # al fallback de rúbrica), la pareja instrucción->etiqueta sería
            # contradictoria entre ejemplos. No se emite.
            sibling_purposes = {
                deadline_purpose(h, articles)
                for h in visible
                if h.value != hit.value
            }
            if purpose in sibling_purposes:
                continue
        # v6: un stem VERBATIM — la frase del documento que contiene el plazo
        # («Las solicitudes se presentarán en el plazo de veinte días
        # hábiles») — enseña a responder con la fórmula normativa exacta.
        # v5 parafraseaba («en un plazo de...») y el containment castigaba
        # hasta el cambio de un artículo gramatical (fallo real en eb-142).
        verbatim_stem = None
        if _ends_sentence(hit.sentence) and len(hit.sentence.split()) <= 45:
            verbatim_stem = (
                hit.sentence.rstrip(".").replace("{", "{{").replace("}", "}}")
            )
        if hit.kind == "duracion":
            json_desc = f"el plazo {purpose}" if purpose else "el plazo"
            prose_variants = (
                _disambiguated_plazo_variants(purpose) if purpose
                else _EXTRACCION_PLAZO_VARIANTS
            )
            stems = _disambiguated_plazo_stems(purpose, "duracion") if purpose else _PLAZO_STEMS
            if verbatim_stem is not None:
                stems = stems + (verbatim_stem,)
            example = _extraccion_example(
                doc, ctx,
                suffix=f"plazo-{i}", window=window, value=hit.value,
                json_field="plazo", json_desc=json_desc,
                prose_variants=prose_variants, prose_stems=stems,
                require_desc=purpose is not None,
            )
        else:
            json_desc = f"la fecha límite {purpose}" if purpose else "la fecha límite"
            prose_variants = (
                _disambiguated_plazo_variants(purpose) if purpose
                else _EXTRACCION_PLAZO_VARIANTS
            )
            stems = _disambiguated_plazo_stems(purpose, "fecha") if purpose else _FECHA_STEMS
            if verbatim_stem is not None:
                stems = stems + (verbatim_stem,)
            example = _extraccion_example(
                doc, ctx,
                suffix=f"fecha-{i}", window=window, value=hit.value,
                json_field="fecha_limite", json_desc=json_desc,
                prose_variants=prose_variants, prose_stems=stems,
                require_desc=purpose is not None,
            )
        if example is not None:
            examples.append(example)

    # Multa máxima. Con régimen de tramos (leves/graves/muy graves) las
    # preguntas nombran el TRAMO y la etiqueta es el tope de ESE tramo, con
    # los demás tramos visibles como distractores (v4 etiquetaba siempre el
    # máximo de ventana y aprendió a responder el tope de las «muy graves»
    # aunque se preguntara por las «graves»: fallo real en eb-128). Se añade
    # además una pregunta global con fraseo explícitamente absoluto.
    tiers = find_sanction_tiers(articles)
    tier_examples_emitted = False
    if len(tiers) >= 2:
        needed = sorted({t.article for t in tiers})
        window = compose_window(doc, articles, needed, ctx.counter, ctx.token_budget)
        visible_tiers = [
            t for t in tiers if t.sentence in window and t.amount in window
        ]
        if len(visible_tiers) >= 2:
            for t in visible_tiers:
                slug = t.tier.replace(" ", "-")
                example = _extraccion_example(
                    doc, ctx,
                    suffix=f"multa-{slug}", window=window, value=t.amount,
                    json_field="multa_maxima",
                    json_desc=f"la multa máxima para las infracciones {t.tier}",
                    prose_variants=_tier_multa_variants(t.tier),
                    prose_stems=_tier_multa_stems(t.tier),
                    require_desc=True,
                )
                if example is not None:
                    examples.append(example)
                    tier_examples_emitted = True
            global_amount = max(
                (t.amount for t in visible_tiers), key=_amount_value
            )
            # Guard: si la ventana muestra CUALQUIER importe mayor que el
            # candidato (un tramo inatribuible, un rango cuyo tope no
            # capturamos), el ejemplo «máximo absoluto» mentiría a la vista
            # del propio texto. No se emite (2/15 globales del primer build
            # contradecían su ventana: hallazgo de la revisión).
            window_max = max_amount_in(window)
            if window_max is None or _amount_value(window_max) > _amount_value(global_amount):
                pass
            else:
                example = _extraccion_example(
                    doc, ctx,
                    suffix="multa-global", window=window, value=global_amount,
                    json_field="multa_maxima",
                    json_desc="la multa más alta que puede imponerse, "
                              "considerando todos los tipos de infracción",
                    prose_variants=_MULTA_GLOBAL_VARIANTS,
                    prose_stems=_MULTA_GLOBAL_STEMS,
                    require_desc=True,
                )
                if example is not None:
                    examples.append(example)

    # Sin tramos discriminables: máximo real de la ventana con pregunta
    # genérica (único importe o régimen no tramificado).
    if not tier_examples_emitted:
        amount_hits = find_amount_hits(articles, limit=6)
        if amount_hits:
            needed = sorted({h.article for h in amount_hits})
            window = compose_window(doc, articles, needed, ctx.counter, ctx.token_budget)
            visible = [h.sentence for h in amount_hits if h.sentence in window]
            amount = max_amount_in(" ".join(visible)) if visible else None
            # Guard de consistencia: la etiqueta solo vale si es el importe
            # más alto VISIBLE en la ventana. Los escalados largos que las
            # frases sancionadoras cortas no capturan (>45 palabras) siguen
            # en la ventana y desmentirían la etiqueta (4 casos del primer
            # build enseñaban un sub-máximo: hallazgo de la revisión).
            if amount is not None:
                window_max = max_amount_in(window)
                if window_max is not None and _amount_value(window_max) > _amount_value(amount):
                    amount = None
            if amount is not None:
                example = _extraccion_example(
                    doc, ctx,
                    suffix="multa-0", window=window, value=amount,
                    json_field="multa_maxima", json_desc="la multa máxima",
                    prose_variants=_EXTRACCION_SANCION_VARIANTS,
                    prose_stems=("La multa máxima prevista es de {v}",
                                 "La sanción puede alcanzar los {v}"),
                )
                if example is not None:
                    examples.append(example)

    for i, hit in enumerate(find_obligation_hits(articles, limit=2)):
        if _EMBEDDED_LIST_MARKER.search(hit.sentence):
            continue  # «a) ... b) ...» embebido: no es una respuesta autónoma
        rng = _example_rng(ctx.seed, doc.id, "extraccion", f"oblig-{i}")
        window = compose_window(doc, articles, [hit.article], ctx.counter, ctx.token_budget)
        if hit.sentence not in window:
            continue
        instruction, cite = _EXTRACCION_OBLIGACION_VARIANTS[
            rng.randrange(len(_EXTRACCION_OBLIGACION_VARIANTS))
        ]
        body = hit.sentence if _ends_sentence(hit.sentence) else f"{hit.sentence}."
        response = f"{body.rstrip('.')} [{doc.id}]." if cite else body
        examples.append(
            SftExample(
                id=f"{doc.id}-extraccion-obligacion-{i}",
                doc_id=doc.id,
                task="extraccion",
                instruction=instruction,
                input=window,
                response=response,
                source="offline",
            )
        )
        break  # una obligación por documento es suficiente

    # Órgano emisor (siempre en la cabecera: anclaje trivial).
    rng = _example_rng(ctx.seed, doc.id, "extraccion", "organo")
    if rng.random() < 0.35:
        window = compose_window(doc, articles, [0], ctx.counter, ctx.token_budget)
        instruction = _EXTRACCION_JSON_INSTR[
            rng.randrange(len(_EXTRACCION_JSON_INSTR))
        ].format(campo="organo", desc="el órgano emisor")
        examples.append(
            SftExample(
                id=f"{doc.id}-extraccion-organo",
                doc_id=doc.id,
                task="extraccion",
                instruction=instruction,
                input=window,
                response=json.dumps({"organo": doc.organo_emisor}, ensure_ascii=False),
                source="offline",
            )
        )
    return examples


_OFFLINE_GENERATORS = {
    "resumen": offline_resumen,
    "qa": offline_qa,
    "obligaciones": offline_obligaciones,
    "llano": offline_llano,
    "extraccion": offline_extraccion,
}


def rebalance(
    examples: list[SftExample],
    weights: dict[str, float],
    seed: int = 42,
) -> list[SftExample]:
    """Re-muestrea por tarea hacia los pesos objetivo (determinista).

    Escala el total al máximo alcanzable sin sobre-representar ninguna tarea:
    ``N* = min_t(count_t / w_t)``; las tareas excedentarias se muestrean.
    """
    by_task: dict[str, list[SftExample]] = {}
    for example in examples:
        by_task.setdefault(example.task, []).append(example)
    applicable = {t: w for t, w in weights.items() if t in by_task and w > 0}
    if not applicable:
        return examples
    n_star = min(len(by_task[t]) / w for t, w in applicable.items())
    rng = random.Random(seed)
    kept: list[SftExample] = []
    for task, task_examples in sorted(by_task.items()):
        weight = applicable.get(task)
        if weight is None:
            kept.extend(task_examples)  # tarea fuera de los pesos: intacta
            continue
        target = min(len(task_examples), max(1, round(weight * n_star)))
        if target >= len(task_examples):
            kept.extend(task_examples)
        else:
            indices = sorted(rng.sample(range(len(task_examples)), target))
            kept.extend(task_examples[i] for i in indices)
    return kept


def build_sft_offline(
    docs: list[RawDocument],
    tasks: tuple[str, ...] = SFT_TASKS,
    seed: int = 42,
    max_per_task: int | None = None,
    counter: TokenCounter | None = None,
    token_budget: int = DEFAULT_WINDOW_TOKENS,
    balance: dict[str, float] | None = None,
) -> list[SftExample]:
    """Genera ejemplos SFT offline con ventana estructural y anclaje."""
    counter = counter or TokenCounter()
    ctx = SynthContext(counter=counter, token_budget=token_budget, seed=seed)
    rng = random.Random(seed)
    examples: list[SftExample] = []
    split_cache: dict[str, list[Article]] = {}
    for task in tasks:
        if task not in _OFFLINE_GENERATORS:
            raise ValueError(f"Tarea desconocida: {task!r} (disponibles: {SFT_TASKS})")
        generated: list[SftExample] = []
        for doc in docs:
            if doc.id not in split_cache:
                _, split_cache[doc.id] = split_articles(doc.texto)
            generated.extend(_OFFLINE_GENERATORS[task](doc, split_cache[doc.id], ctx))
        if max_per_task is not None and len(generated) > max_per_task:
            indices = sorted(rng.sample(range(len(generated)), max_per_task))
            generated = [generated[i] for i in indices]
        examples.extend(generated)
    if balance:
        examples = rebalance(examples, balance, seed=seed)
    return examples
