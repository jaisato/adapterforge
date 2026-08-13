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
import json
import random
import re
from pathlib import Path

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

SFT_TASKS = ("resumen", "qa", "obligaciones", "llano")

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


def offline_resumen(doc: RawDocument) -> SftExample:
    """Resumen estructurado con campos fijos y cita de la fuente."""
    _, articles = split_articles(doc.texto)
    objeto = _first_sentence(articles[0].body) if articles else _first_sentence(doc.texto)
    key_points = []
    for article in articles[1:5]:
        label = article.heading.rstrip(".") if article.heading else f"Artículo {article.number}"
        key_points.append(f"- Artículo {article.number} ({label}): {_first_sentence(article.body)}")
    obligations = find_obligation_sentences(doc.texto, limit=2)
    deadlines = find_deadlines(doc.texto, limit=4)
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
    return SftExample(
        id=f"{doc.id}-resumen",
        doc_id=doc.id,
        task="resumen",
        instruction=(
            "Elabora un resumen estructurado de la siguiente disposición con los "
            "campos: norma, rango, órgano emisor, publicación, materias, objeto, "
            "puntos clave por artículo, obligaciones y plazos. Cita la fuente."
        ),
        input=_doc_input(doc),
        response="\n".join(lines),
        source="offline",
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
                    f"Según el texto de la disposición, ¿qué establece el artículo "
                    f"{article.number} {reference}? Cita la fuente entre corchetes."
                ),
                input=_doc_input(doc),
                response=response,
                source="offline",
            )
        )
    return examples


def offline_obligaciones(doc: RawDocument) -> SftExample:
    """Extracción estructurada en JSON de obligaciones, plazos y sanciones."""
    payload = {
        "obligaciones": find_obligation_sentences(doc.texto, limit=5),
        "plazos": find_deadlines(doc.texto, limit=5),
        "sanciones": find_sanction_sentences(doc.texto, limit=3),
        "fuente": doc.id,
    }
    return SftExample(
        id=f"{doc.id}-obligaciones",
        doc_id=doc.id,
        task="obligaciones",
        instruction=(
            "Extrae las obligaciones, los plazos y las sanciones de la siguiente "
            'disposición. Responde únicamente con un objeto JSON con las claves '
            '"obligaciones", "plazos", "sanciones" y "fuente".'
        ),
        input=_doc_input(doc),
        response=json.dumps(payload, ensure_ascii=False, indent=2),
        source="offline",
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
        instruction=(
            f"Explica en lenguaje llano, para una persona sin formación jurídica, "
            f"qué dice {_short_ref(doc)} y qué obligaciones y plazos le afectan."
        ),
        input=_doc_input(doc),
        response="\n".join(lines),
        source="offline",
    )


_OFFLINE_GENERATORS = {
    "resumen": lambda doc: [offline_resumen(doc)],
    "qa": offline_qa,
    "obligaciones": lambda doc: [offline_obligaciones(doc)],
    "llano": lambda doc: [offline_llano(doc)],
}


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
