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

Tareas soportadas (una plantilla por tarea en ``prompts/``):

- ``resumen``: resumen estructurado con campos fijos.
- ``qa``: preguntas/respuestas por artículo con cita [BOE-A-..., art. N].
- ``obligaciones``: extracción de obligaciones/plazos/sanciones en JSON.
- ``llano``: reescritura en lenguaje llano para no juristas.

Uso CLI::

    python -m adapterforge.data.synthesize \
        --input data/interim/documents.jsonl \
        --output data/interim/sft_raw.jsonl --offline

    python -m adapterforge.data.synthesize --input ... --output pares_dpo.jsonl \
        --offline --dpo

TODO(jairo): variante Evol-Instruct (mutar instrucciones) y self-instruct con
semillas humanas para aumentar diversidad del dataset online.
"""

from __future__ import annotations

import argparse
from pydoc import doc
import zlib
import json
import random
import re
from pathlib import Path

from annotated_types import doc
from pydantic import BaseModel, ConfigDict, Field

from adapterforge.data.extract import (
    Article,
    RawDocument,
    load_corpus,
    split_articles,
    split_sentences,
)
from adapterforge.jsonl import write_jsonl
from adapterforge.settings import get_settings

SFT_TASKS = ("resumen", "qa", "obligaciones", "llano", "extraccion")

def _variant(doc_id: str, n: int, salt: str = "") -> int:
    """Índice de variante determinista por documento (reproducible)."""
    return zlib.crc32(f"{doc_id}:{salt}".encode("utf-8")) % n

# Marcadores léxicos para detectar obligaciones/plazos/sanciones en texto
# jurídico-administrativo. Deliberadamente simples y auditables.
OBLIGATION_MARKERS = (
    "deberá",
    "deberán",
    "estarán obligad",
    "están obligad",
    "queda prohibid",
    "no podrá",
    "habrán de",
    "tendrán que",
)
SANCTION_MARKERS = ("multa", "sanción", "sancionad", "infracción", "reintegro")
_DEADLINE_PATTERNS = (
    re.compile(r"plazo(?: máximo)? de ([^,.;]+)"),
    re.compile(r"antes del (\d[^,.;]*)"),
    re.compile(r"en el término de ([^,.;]+)"),
)

# Glosario para la reescritura offline en lenguaje llano (orden importa:
# primero las expresiones más largas).
PLAIN_GLOSSARY: tuple[tuple[str, str], ...] = (
    ("estarán obligadas a", "tienen que"),
    ("estarán obligados a", "tienen que"),
    ("la presente disposición", "esta norma"),
    ("La presente resolución", "Esta resolución"),
    ("La presente norma", "Esta norma"),
    ("La presente orden", "Esta orden"),
    ("La presente ley", "Esta ley"),
    ("tiene por objeto", "sirve para"),
    ("de conformidad con", "según"),
    ("con carácter previo", "antes"),
    ("sin perjuicio de", "además de"),
    ("a tal efecto", "para ello"),
    ("en el plazo de", "en un máximo de"),
    ("No obstante", "Sin embargo"),
    ("no obstante", "sin embargo"),
    ("deberán", "tienen que"),
    ("deberá", "tiene que"),
)

_RANGO_ARTICLE = {
    "ley": "una ley",
    "real decreto": "un real decreto",
    "real decreto-ley": "un real decreto-ley",
    "orden ministerial": "una orden ministerial",
    "orden": "una orden ministerial",
    "resolución": "una resolución",
}

_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")

_MONTHS_ES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)

# Rangos gramaticalmente femeninos ("la Ley", "de la Orden"...).
_FEMININE_RANKS = {"ley", "orden", "orden ministerial", "resolución"}

# Artículo del órgano emisor según su primera palabra ("el Ministerio...").
_ORGAN_ARTICLES = {
    "ministerio": "el ",
    "jefatura": "la ",
    "servicio": "el ",
    "secretaría": "la ",
    "agencia": "la ",
    "instituto": "el ",
}


def format_date_es(iso_date: str) -> str:
    """'2026-03-18' -> '18 de marzo de 2026' (para el lenguaje llano)."""
    year, month, day = iso_date.split("-")
    return f"{int(day)} de {_MONTHS_ES[int(month) - 1]} de {int(year)}"


def _short_ref(doc: RawDocument, *, preceded_by_de: bool = False) -> str:
    """Referencia corta con artículo correcto: 'el Real Decreto X' / 'de la Ley Y'."""
    feminine = doc.rango.lower() in _FEMININE_RANKS
    if preceded_by_de:
        return ("de la " if feminine else "del ") + doc.short_title()
    return ("la " if feminine else "el ") + doc.short_title()


def _organ_with_article(organ: str) -> str:
    """Antepone el artículo al órgano emisor cuando se conoce su género."""
    first_word = organ.split()[0].lower() if organ.split() else ""
    return _ORGAN_ARTICLES.get(first_word, "") + organ


class SftExample(BaseModel):
    """Ejemplo instruccional previo al formateo chat."""

    model_config = ConfigDict(extra="forbid")

    id: str
    doc_id: str
    task: str
    instruction: str
    input: str
    response: str
    source: str = Field(description="'offline' o nombre del modelo teacher")


class DpoPair(BaseModel):
    """Par de preferencia (chosen/rejected) previo al formateo chat."""

    model_config = ConfigDict(extra="forbid")

    id: str
    doc_id: str
    task: str
    instruction: str
    input: str
    chosen: str
    rejected: str
    source: str = "offline"


# ---------------------------------------------------------------------------
# Detectores estructurales
# ---------------------------------------------------------------------------


def find_obligation_sentences(text: str, limit: int = 5) -> list[str]:
    """Frases que expresan obligaciones (marcadores deónticos).

    Se descartan "frases" de menos de 4 palabras: suelen ser rúbricas de
    artículo ("Reintegro.") y no aportan contenido.
    """
    found = [
        sentence
        for sentence in split_sentences(text)
        if len(sentence.split()) >= 4
        and any(marker in sentence.lower() for marker in OBLIGATION_MARKERS)
    ]
    return found[:limit]


def find_deadlines(text: str, limit: int = 5) -> list[str]:
    """Expresiones de plazo ("tres meses", "31 de diciembre de 2026"...)."""
    deadlines: list[str] = []
    for pattern in _DEADLINE_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1).strip()
            if value and value not in deadlines:
                deadlines.append(value)
    return deadlines[:limit]


def find_sanction_sentences(text: str, limit: int = 3) -> list[str]:
    """Frases que mencionan sanciones, multas o reintegros (>= 4 palabras)."""
    found = [
        sentence
        for sentence in split_sentences(text)
        if len(sentence.split()) >= 4
        and any(marker in sentence.lower() for marker in SANCTION_MARKERS)
    ]
    return found[:limit]


def simplify_text(text: str) -> str:
    """Aplica el glosario de lenguaje llano (sustitución determinística)."""
    result = text
    for formal, plain in PLAIN_GLOSSARY:
        result = result.replace(formal, plain)
    return result


# ---------------------------------------------------------------------------
# Generadores offline (determinísticos, sin LLM)
# ---------------------------------------------------------------------------


def _doc_input(doc: RawDocument) -> str:
    """Bloque de documento que acompaña a la instrucción (anclaje)."""
    return f"{doc.header()}\n\n{doc.texto}"


def _first_sentence(text: str) -> str:
    sentences = split_sentences(text)
    return sentences[0] if sentences else text.strip()


_RESUMEN_INSTRUCTIONS_ESTRUCTURADO = (
    "Elabora un resumen estructurado de la siguiente disposición con los "
    "campos: norma, rango, órgano emisor, publicación, materias, objeto, "
    "puntos clave por artículo, obligaciones y plazos. Cita la fuente.",
    "Resume la disposición siguiente en formato de ficha con estos campos: "
    "norma, rango, órgano emisor, publicación, materias, objeto, puntos "
    "clave, obligaciones y plazos. Incluye la cita de la fuente.",
    "Prepara una ficha-resumen de esta norma (norma, rango, órgano emisor, "
    "publicación, materias, objeto, puntos clave por artículo, obligaciones "
    "y plazos), citando la fuente al final.",
)

_RESUMEN_INSTRUCTIONS_PROSA = (
    "Resume en un párrafo de 4 a 6 frases la siguiente disposición: qué "
    "norma es, quién la publica, cuál es su objeto y qué obligaciones y "
    "plazos principales establece. Cita la fuente al final.",
    "Escribe un resumen breve, en prosa y sin listas, de esta disposición "
    "indicando norma, órgano emisor, objeto y los plazos más relevantes. "
    "Termina citando la fuente.",
    "¿De qué trata la siguiente disposición? Contesta en un único párrafo "
    "mencionando la norma, su objeto y sus plazos clave, y cita la fuente.",
)


def offline_resumen(doc: RawDocument) -> SftExample:
    """Resumen con formato variable: ficha estructurada o prosa (50/50)."""
    _, articles = split_articles(doc.texto)
    objeto = _first_sentence(articles[0].body) if articles else _first_sentence(doc.texto)
    obligations = find_obligation_sentences(doc.texto, limit=2)
    deadlines = find_deadlines(doc.texto, limit=4)

    use_prosa = _variant(doc.id, 2, salt="resumen-formato") == 1

    if not use_prosa:
        key_points = []
        for article in articles[1:5]:
            label = article.heading.rstrip(".") if article.heading else f"Artículo {article.number}"
            key_points.append(
                f"- Artículo {article.number} ({label}): {_first_sentence(article.body)}"
            )
        lines = [
            f"### Resumen estructurado — {doc.short_title()}",
            "",
            f"- **Norma**: {doc.titulo}",
            f"- **Rango**: {doc.rango}",
            f"- **Órgano emisor**: {doc.organo_emisor}",
            f"- **Publicación**: {doc.fecha_publicacion}",
            f"- **Materias**: {', '.join(doc.materia)}",
            f"- **Objeto**: {objeto}",
        ]
        if key_points:
            lines += ["", "**Puntos clave por artículo:**", *key_points]
        if obligations:
            lines += ["", "**Obligaciones destacadas:**"]
            lines += [f"- {sentence}" for sentence in obligations]
        if deadlines:
            lines += ["", f"**Plazos relevantes:** {'; '.join(deadlines)}."]
        lines += ["", f"Fuente: [{doc.id}] {doc.url}"]
        response = "\n".join(lines)
        pool = _RESUMEN_INSTRUCTIONS_ESTRUCTURADO
    else:
        sentences = [
            f"{doc.short_title()} es una disposición de rango {doc.rango.lower()} "
            f"publicada por {_organ_with_article(doc.organo_emisor)} el "
            f"{format_date_es(doc.fecha_publicacion)}.",
            f"Su objeto es el siguiente: {objeto}",
        ]
        if obligations:
            sentences.append(f"Entre sus obligaciones destaca que {obligations[0].rstrip('.').lower()}.")
        if deadlines:
            sentences.append(f"Los plazos más relevantes son: {'; '.join(deadlines)}.")
        sentences.append(f"Fuente: [{doc.id}] {doc.url}.")
        response = " ".join(sentences)
        pool = _RESUMEN_INSTRUCTIONS_PROSA

    instruction = pool[_variant(doc.id, len(pool), salt="resumen-instr")]
    return SftExample(
        id=f"{doc.id}-resumen",
        doc_id=doc.id,
        task="resumen",
        instruction=instruction,
        input=_doc_input(doc),
        response=response,
        source="offline",
    )

_QA_INSTRUCTIONS = (
    "Según el texto de la disposición, ¿qué establece el artículo {numero} "
    "{referencia}? Cita la fuente entre corchetes.",
    "¿Qué dispone el artículo {numero} {referencia}? Responde citando la "
    "fuente entre corchetes.",
    "Explica el contenido del artículo {numero} {referencia} y cita la "
    "fuente entre corchetes al final.",
)

def offline_qa(doc: RawDocument, max_questions: int = 2) -> list[SftExample]:
    """Q&A por artículo con cita en el formato del portfolio."""
    _, articles = split_articles(doc.texto)
    examples: list[SftExample] = []
    candidates: list[Article] = [article for article in articles if article.body][:max_questions]
    for article in candidates:
        sentences = split_sentences(article.body)
        quoted = " ".join(sentences[:2])
        heading = f" ({article.heading.rstrip('.')})" if article.heading else ""
        reference = _short_ref(doc, preceded_by_de=True)
        response = (
            f"El artículo {article.number}{heading} {reference} establece lo "
            f"siguiente: «{quoted}» [{doc.id}, art. {article.number}]."
        )
        examples.append(
            SftExample(
                id=f"{doc.id}-qa-{article.number}",
                doc_id=doc.id,
                task="qa",
                instruction=(
                    _QA_INSTRUCTIONS[_variant(doc.id, len(_QA_INSTRUCTIONS), salt=f"qa-{article.number}")]
                    .format(numero=article.number, referencia=reference)
                ),
                input=_doc_input(doc),
                response=response,
                source="offline",
            )
        )
    return examples


def offline_obligaciones(doc: RawDocument) -> SftExample:
    """Extracción de obligaciones/plazos/sanciones con formato variable."""
    _, articles = split_articles(doc.texto)
    articulado = "\n".join(a.body for a in articles) or doc.texto  # fallback
    obligations = find_obligation_sentences(articulado, limit=5)
    deadlines = find_deadlines(articulado, limit=5)
    sanctions = find_sanction_sentences(articulado, limit=3)

    fmt = _variant(doc.id, 3, salt="obligaciones-formato")

    if fmt == 0:  # JSON (formato v1)
        instruction = (
            "Extrae las obligaciones, los plazos y las sanciones de la siguiente "
            'disposición. Responde únicamente con un objeto JSON con las claves '
            '"obligaciones", "plazos", "sanciones" y "fuente".'
        )
        payload = {
            "obligaciones": obligations,
            "plazos": deadlines,
            "sanciones": sanctions,
            "fuente": doc.id,
        }
        response = json.dumps(payload, ensure_ascii=False, indent=2)
    elif fmt == 1:  # Lista en texto plano
        instruction = (
            "Enumera las obligaciones, los plazos y las sanciones que establece "
            "la siguiente disposición, en tres apartados con viñetas. Cita la "
            "fuente al final."
        )
        lines = ["Obligaciones:"]
        lines += [f"- {s}" for s in obligations] or ["- (no se detectan)"]
        lines += ["", "Plazos:"]
        lines += [f"- {d}" for d in deadlines] or ["- (no se detectan)"]
        lines += ["", "Sanciones:"]
        lines += [f"- {s}" for s in sanctions] or ["- (no se detectan)"]
        lines += ["", f"Fuente: [{doc.id}]"]
        response = "\n".join(lines)
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
        if not parts:
            parts.append("No se detectan obligaciones, plazos ni sanciones explícitas en el texto.")
        parts.append(f"[{doc.id}]")
        response = " ".join(parts)

    return SftExample(
        id=f"{doc.id}-obligaciones",
        doc_id=doc.id,
        task="obligaciones",
        instruction=instruction,
        input=_doc_input(doc),
        response=response,
        source="offline",
    )

_LLANO_INSTRUCTIONS = (
    "Explica en lenguaje llano, para una persona sin formación jurídica, qué "
    "dice {referencia} y qué obligaciones y plazos le afectan.",
    "Explícame como si no supiera nada de derecho qué establece {referencia} "
    "y qué tengo que hacer o tener en cuenta.",
    "Traduce a lenguaje sencillo el contenido de {referencia}: qué regula, a "
    "quién obliga y con qué plazos.",
)

def offline_llano(doc: RawDocument) -> SftExample:
    """Explicación en lenguaje llano para personas sin formación jurídica."""
    _, articles = split_articles(doc.texto)
    objeto = _first_sentence(articles[0].body) if articles else _first_sentence(doc.texto)
    rango_key = doc.rango.lower()
    rango_phrase = _RANGO_ARTICLE.get(rango_key, f"una norma de rango {doc.rango.lower()}")
    obligations = [simplify_text(s) for s in find_obligation_sentences(doc.texto, limit=3)]
    deadlines = find_deadlines(doc.texto, limit=3)
    lines = [
        f"**¿Qué es?** {doc.short_title()} es {rango_phrase} que publicó "
        f"{_organ_with_article(doc.organo_emisor)} el {format_date_es(doc.fecha_publicacion)}.",
        "",
        f"**¿De qué trata?** {simplify_text(objeto)}",
    ]
    if obligations:
        lines += ["", "**¿Qué obligaciones impone?**"]
        lines += [f"- {sentence}" for sentence in obligations]
    if deadlines:
        lines += ["", "**Plazos que conviene apuntar:**"]
        lines += [f"- {deadline}" for deadline in deadlines]
    lines += [
        "",
        "**En resumen:** revisa si estás dentro del ámbito de aplicación y anota "
        f"los plazos. El texto oficial está disponible en {doc.url} [{doc.id}].",
    ]
    return SftExample(
        id=f"{doc.id}-llano",
        doc_id=doc.id,
        task="llano",
        instruction=_LLANO_INSTRUCTIONS[_variant(doc.id, len(_LLANO_INSTRUCTIONS), salt="llano")].format(referencia=_short_ref(doc)),
        input=_doc_input(doc),
        response="\n".join(lines),
        source="offline",
    )

_EXTRACCION_PLAZO_INSTR = (
    "Según la disposición siguiente, ¿qué plazo se establece? Responde en una "
    "sola frase y cita la fuente entre corchetes.",
    "Indica de forma breve el plazo que fija esta disposición, citando la "
    "fuente entre corchetes.",
    "¿De cuánto tiempo se dispone según esta norma? Contesta con una frase y "
    "la cita de la fuente.",
)

_EXTRACCION_SANCION_INSTR = (
    "¿Qué sanción o multa prevé la siguiente disposición? Responde de forma "
    "breve citando la fuente entre corchetes.",
    "Indica brevemente el régimen sancionador que establece esta norma, con "
    "cita de la fuente entre corchetes.",
)

_EXTRACCION_OBLIGACION_INSTR = (
    "¿Qué obligación principal impone la siguiente disposición y a quién? "
    "Responde de forma breve citando la fuente.",
    "Señala en una o dos frases la obligación más relevante que establece "
    "esta norma, con la cita de la fuente.",
)


def offline_extraccion(doc: RawDocument) -> list[SftExample]:
    """Preguntas puntuales con respuesta corta: SOLO el dato pedido + cita.

    Es la tarea que ejercita el domain_bench (extract) y que la v1 no cubría:
    enseña al modelo a responder escueto sin imponer plantillas.
    """
    examples: list[SftExample] = []
    _, articles = split_articles(doc.texto)
    articulado = "\n".join(a.body for a in articles) or doc.texto  # fallback
    for i, deadline in enumerate(find_deadlines(articulado, limit=2)):
        instr = _EXTRACCION_PLAZO_INSTR[_variant(doc.id, len(_EXTRACCION_PLAZO_INSTR), salt=f"ext-p{i}")]
        examples.append(
            SftExample(
                id=f"{doc.id}-extraccion-plazo-{i}",
                doc_id=doc.id,
                task="extraccion",
                instruction=instr,
                input=_doc_input(doc),
                response=f"El plazo establecido es de {deadline} [{doc.id}].",
                source="offline",
            )
        )

    sanctions = find_sanction_sentences(articulado, limit=1)
    if sanctions:
        instr = _EXTRACCION_SANCION_INSTR[_variant(doc.id, len(_EXTRACCION_SANCION_INSTR), salt="ext-s")]
        examples.append(
            SftExample(
                id=f"{doc.id}-extraccion-sancion",
                doc_id=doc.id,
                task="extraccion",
                instruction=instr,
                input=_doc_input(doc),
                response=f"{sanctions[0]} [{doc.id}].",
                source="offline",
            )
        )

    obligations = find_obligation_sentences(articulado, limit=1)
    if obligations:
        instr = _EXTRACCION_OBLIGACION_INSTR[_variant(doc.id, len(_EXTRACCION_OBLIGACION_INSTR), salt="ext-o")]
        examples.append(
            SftExample(
                id=f"{doc.id}-extraccion-obligacion",
                doc_id=doc.id,
                task="extraccion",
                instruction=instr,
                input=_doc_input(doc),
                response=f"{obligations[0]} [{doc.id}].",
                source="offline",
            )
        )
    return examples

_OFFLINE_GENERATORS = {
    "resumen": lambda doc: [offline_resumen(doc)],
    "qa": offline_qa,
    "obligaciones": lambda doc: [offline_obligaciones(doc)],
    "llano": lambda doc: [offline_llano(doc)],
    "extraccion": offline_extraccion,
}

CONTEXT_CHAR_BUDGET = 4500

def build_sft_offline(
    docs: list[RawDocument],
    tasks: tuple[str, ...] = SFT_TASKS,
    seed: int = 42,
    max_per_task: int | None = None,
) -> list[SftExample]:
    """Genera ejemplos SFT offline, con muestreo determinístico opcional."""
    rng = random.Random(seed)
    examples: list[SftExample] = []
    for task in tasks:
        if task not in _OFFLINE_GENERATORS:
            raise ValueError(f"Tarea desconocida: {task!r} (disponibles: {SFT_TASKS})")
        generated: list[SftExample] = []
        for doc in docs:
            if len(doc.texto) > CONTEXT_CHAR_BUDGET:
                doc = doc.model_copy(update={"texto": doc.texto[:CONTEXT_CHAR_BUDGET]})
            generated.extend(_OFFLINE_GENERATORS[task](doc))
        if max_per_task is not None and len(generated) > max_per_task:
            indices = sorted(rng.sample(range(len(generated)), max_per_task))
            generated = [generated[i] for i in indices]
        examples.extend(generated)
    return examples


# ---------------------------------------------------------------------------
# Pares de preferencia (DPO) offline
# ---------------------------------------------------------------------------


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
) -> list[DpoPair]:
    """Construye pares chosen/rejected degradando la respuesta buena.

    ``chosen`` es la salida offline estructurada (resumen u obligaciones);
    ``rejected`` es una degradación realista: vaguedad sin anclaje, copia
    literal truncada o respuesta desestructurada sin formato ni cita.
    """
    rng = random.Random(seed)
    candidates: list[SftExample] = []
    for doc in docs:
        candidates.append(offline_resumen(doc))
        candidates.append(offline_obligaciones(doc))
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


# ---------------------------------------------------------------------------
# Síntesis online (teacher vía litellm / endpoint OpenAI-compatible)
# ---------------------------------------------------------------------------


def render_prompt(template_text: str, doc: RawDocument) -> str:
    """Sustituye placeholders ``{{campo}}`` del template con datos del doc."""
    mapping = {
        "id": doc.id,
        "rango": doc.rango,
        "titulo": doc.titulo,
        "titulo_corto": doc.short_title(),
        "organo_emisor": doc.organo_emisor,
        "fecha_publicacion": doc.fecha_publicacion,
        "materias": ", ".join(doc.materia),
        "url": doc.url,
        "texto": doc.texto,
    }

    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in mapping:
            missing.append(key)
            return match.group(0)
        return mapping[key]

    rendered = _PLACEHOLDER.sub(_replace, template_text)
    if missing:
        raise ValueError(f"Placeholders desconocidos en la plantilla: {sorted(set(missing))}")
    return rendered


def load_prompt_template(prompts_dir: str | Path, task: str) -> str:
    """Carga la plantilla de una tarea desde ``prompts/<task>.md``."""
    path = Path(prompts_dir) / f"{task}.md"
    if not path.exists():
        raise FileNotFoundError(f"No existe la plantilla {path} para la tarea {task!r}")
    return path.read_text(encoding="utf-8")


def synthesize_online(
    docs: list[RawDocument],
    tasks: tuple[str, ...],
    *,
    model: str,
    backend: str,
    api_base: str | None,
    api_key: str | None,
    prompts_dir: str | Path,
) -> list[SftExample]:
    """Genera ejemplos preguntando al teacher (import perezoso del cliente)."""
    from adapterforge.llm import chat_complete  # perezoso: evita litellm en offline

    examples: list[SftExample] = []
    for task in tasks:
        template = load_prompt_template(prompts_dir, task)
        for doc in docs:
            prompt = render_prompt(template, doc)
            response = chat_complete(
                [{"role": "user", "content": prompt}],
                model,
                backend=backend,
                api_base=api_base,
                api_key=api_key,
                temperature=0.4,
                max_tokens=1200,
            )
            reference = _OFFLINE_GENERATORS[task](doc)
            instruction = reference[0].instruction if reference else prompt[:200]
            examples.append(
                SftExample(
                    id=f"{doc.id}-{task}",
                    doc_id=doc.id,
                    task=task,
                    instruction=instruction,
                    input=_doc_input(doc),
                    response=response,
                    source=model,
                )
            )
    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Síntesis de dataset instruccional")
    parser.add_argument("--input", required=True, help="Corpus JSONL (salida de extract)")
    parser.add_argument("--output", required=True, help="JSONL de ejemplos generados")
    parser.add_argument(
        "--offline", action="store_true", help="Sin LLM: plantillas determinísticas"
    )
    parser.add_argument("--dpo", action="store_true", help="Genera pares chosen/rejected (offline)")
    parser.add_argument("--tasks", default=",".join(SFT_TASKS), help="Tareas separadas por comas")
    parser.add_argument("--max-per-task", type=int, default=None)
    parser.add_argument("--n-pairs", type=int, default=15, help="Nº de pares DPO")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, help="Teacher (default: AF_TEACHER_MODEL)")
    parser.add_argument("--backend", default="litellm", choices=["litellm", "openai"])
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompts-dir", default="prompts")
    args = parser.parse_args(argv)

    settings = get_settings()
    docs = load_corpus(args.input)
    if args.max_docs is not None:
        docs = docs[: args.max_docs]
    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())

    if args.dpo:
        if not args.offline:
            raise SystemExit(
                "La generación DPO online no está en el alcance base; usa --offline. "
                "TODO(jairo): pares de preferencia online (teacher fuerte vs débil)."
            )
        pairs = build_dpo_offline(docs, seed=args.seed, n_pairs=args.n_pairs)
        count = write_jsonl(args.output, (pair.model_dump() for pair in pairs))
        print(f"[synthesize] {count} pares DPO offline -> {args.output}")
        return 0

    if args.offline:
        examples = build_sft_offline(
            docs, tasks=tasks, seed=args.seed, max_per_task=args.max_per_task
        )
    else:
        examples = synthesize_online(
            docs,
            tasks,
            model=args.model or settings.teacher_model,
            backend=args.backend,
            api_base=args.api_base or settings.api_base,
            api_key=args.api_key or settings.api_key,
            prompts_dir=args.prompts_dir,
        )
    count = write_jsonl(args.output, (example.model_dump() for example in examples))
    mode = "offline" if args.offline else f"online ({args.model or settings.teacher_model})"
    print(f"[synthesize] {count} ejemplos SFT {mode} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
