"""Instrucciones y plantillas: lo que el modelo lee como pregunta.

- Pools de instrucciones por tarea (muestreados POR EJEMPLO por los
  generadores offline, con variantes con y sin cita y variantes JSON estilo
  benchmark), más los *stems* de respuesta de extracción y las variantes
  desambiguadas (propósito del plazo, tramo sancionador).
- Plantillas del teacher online: ``prompts/<tarea>.md`` con placeholders
  ``{{campo}}`` (:func:`load_prompt_template`, :func:`render_prompt`).

Cambiar un texto de este módulo cambia el dataset generado: las plantillas
se consideran versionadas con el código.
"""

from __future__ import annotations

import re
from pathlib import Path

from adapterforge.data.extract import RawDocument

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
    # Espejo del estilo de instrucción del benchmark (sin cita).
    ("Resume en un párrafo la disposición, indicando objeto, obligaciones y "
     "plazos principales.", False),
    ("Resume en un párrafo la disposición, indicando objeto y plazos.", False),
    ("Resume en un párrafo la disposición, indicando obligaciones, plazos y "
     "sanciones si las hay.", False),
    ("Resume esta disposición en un solo párrafo: qué regula, a quién "
     "obliga y qué plazos marca.", False),
    # Variantes con cita (minoritarias: el estilo del producto).
    ("Escribe un resumen breve, en prosa y sin listas, de esta disposición "
     "indicando objeto, obligaciones y los plazos más relevantes. Termina "
     "citando la fuente.", True),
    ("¿De qué trata la siguiente disposición? Contesta en un único párrafo "
     "mencionando su objeto y sus plazos clave, y cita la fuente.", True),
)


_QA_INSTRUCTIONS = (
    "Según el texto de la disposición, ¿qué establece el artículo {numero} "
    "{referencia}? Cita la fuente entre corchetes.",
    "¿Qué dispone el artículo {numero} {referencia}? Responde citando la "
    "fuente entre corchetes.",
    "Explica el contenido del artículo {numero} {referencia} y cita la "
    "fuente entre corchetes al final.",
    "Resume qué regula el artículo {numero} {referencia}, con la cita de la "
    "fuente entre corchetes.",
    "¿Qué dice exactamente el artículo {numero} {referencia}? Apoya tu "
    "respuesta con la cita de la fuente.",
)

_LLANO_INSTRUCTIONS = (
    "Explica en lenguaje llano, para una persona sin formación jurídica, qué "
    "dice {referencia} y qué obligaciones y plazos le afectan.",
    "Explícame como si no supiera nada de derecho qué establece {referencia} "
    "y qué tengo que hacer o tener en cuenta.",
    "Traduce a lenguaje sencillo el contenido de {referencia}: qué regula, a "
    "quién obliga y con qué plazos.",
    "¿Me afecta {referencia}? Cuéntamelo en lenguaje claro: qué regula, qué "
    "obligaciones impone y qué fechas conviene apuntar.",
)

_EXTRACCION_PLAZO_VARIANTS = (
    ("Según la disposición siguiente, ¿qué plazo se establece? Responde en "
     "una sola frase y cita la fuente entre corchetes.", True),
    ("Indica de forma breve el plazo que fija esta disposición, citando la "
     "fuente entre corchetes.", True),
    ("¿De cuánto tiempo se dispone según esta norma? Contesta con una única "
     "frase que incluya la cita de la fuente.", True),
    ("¿Qué plazo establece esta disposición? Responde en una sola frase, "
     "sin añadir nada más.", False),
    ("Dime únicamente el plazo que fija la norma siguiente, en una frase.", False),
)

_PLAZO_STEMS = (
    "El plazo establecido es de {v}",
    "El plazo fijado es de {v}",
    "La norma establece un plazo de {v}",
)


def _disambiguated_plazo_variants(purpose: str) -> tuple[tuple[str, bool], ...]:
    """Variantes de pregunta que especifican QUÉ plazo se pide."""
    return (
        (f"Según la disposición siguiente, ¿qué plazo se establece {purpose}? "
         "Responde en una sola frase y cita la fuente entre corchetes.", True),
        (f"Indica brevemente el plazo fijado {purpose}, citando la fuente "
         "entre corchetes.", True),
        (f"¿Qué plazo fija esta norma {purpose}? Responde en una sola frase, "
         "sin añadir nada más.", False),
        (f"Extrae el plazo previsto {purpose}. Responde en una sola frase.", False),
        (f"¿De cuánto tiempo se dispone {purpose} según esta norma? Contesta "
         "con una única frase que incluya la cita de la fuente.", True),
    )


def _disambiguated_plazo_stems(purpose: str, kind: str) -> tuple[str, ...]:
    """Stems de respuesta que ECO el propósito preguntado.

    v4 respondía «El plazo establecido es de X» sin repetir el propósito: la
    loss nunca ligaba pregunta y valor, y el modelo podía aprender «responde
    un plazo cualquiera». Repetir el propósito en la respuesta hace que la
    elección correcta forme parte de la señal de entrenamiento.
    """
    if kind == "fecha":
        return (
            f"La fecha límite {purpose} es el {{v}}",
            f"El plazo {purpose} vence el {{v}}",
        )
    return (
        f"El plazo {purpose} es de {{v}}",
        f"El plazo establecido {purpose} es de {{v}}",
    )


def _tier_multa_variants(tier: str) -> tuple[tuple[str, bool], ...]:
    """Preguntas de multa que nombran el TRAMO cuyo tope se pide."""
    return (
        (f"¿Qué multa máxima se prevé para las infracciones {tier}? Responde "
         "de forma breve citando la fuente entre corchetes.", True),
        (f"Indica la multa máxima aplicable a las infracciones {tier}, "
         "citando la fuente entre corchetes.", True),
        (f"¿Con qué multa máxima se sancionan las infracciones {tier} según "
         "esta disposición? Contesta en una frase.", False),
    )


def _tier_multa_stems(tier: str) -> tuple[str, ...]:
    return (
        f"La multa máxima para las infracciones {tier} es de {{v}}",
        f"Las infracciones {tier} se sancionan con multa de hasta {{v}}",
    )


_MULTA_GLOBAL_VARIANTS = (
    ("¿Cuál es la multa más alta que puede llegar a imponerse según esta "
     "disposición, considerando todos los tipos de infracción? Responde de "
     "forma breve citando la fuente entre corchetes.", True),
    ("Indica el importe máximo absoluto de multa que contempla esta norma, "
     "sea cual sea el tipo de infracción, citando la fuente.", True),
)

_MULTA_GLOBAL_STEMS = (
    "La multa más alta que contempla la norma es de {v}",
    "El importe máximo absoluto de multa asciende a {v}",
)

_FECHA_STEMS = (
    "La fecha límite es el {v}",
    "El plazo vence el {v}",
)

_EXTRACCION_SANCION_VARIANTS = (
    ("¿Qué sanción o multa prevé la siguiente disposición? Responde de forma "
     "breve citando la fuente entre corchetes.", True),
    ("Indica brevemente el régimen sancionador que establece esta norma, con "
     "cita de la fuente entre corchetes.", True),
    ("¿Qué multa puede imponerse según esta disposición? Contesta en una "
     "frase.", False),
)

_EXTRACCION_OBLIGACION_VARIANTS = (
    ("¿Qué obligación principal impone la siguiente disposición y a quién? "
     "Responde de forma breve citando la fuente.", True),
    ("Señala en una o dos frases la obligación más relevante que establece "
     "esta norma, con la cita de la fuente.", True),
    ("Resume en una frase la obligación central de esta norma.", False),
)

_OBLIGACIONES_JSON_INSTRUCTION = (
    "Extrae las obligaciones, los plazos y las sanciones de la siguiente "
    'disposición. Responde únicamente con un objeto JSON con las claves '
    '"obligaciones", "plazos", "sanciones" y "fuente".'
)

# Variantes estilo benchmark: JSON con clave pedida y valor VERBATIM.
_EXTRACCION_JSON_INSTR = (
    "Extrae {desc} de la siguiente disposición. Responde únicamente con un "
    'objeto JSON con la clave "{campo}".',
    "Lee la disposición y devuelve solo un objeto JSON con la clave "
    '"{campo}" y el valor exacto tal como aparece en el texto.',
    "Devuelve un objeto JSON, sin texto adicional, con la clave "
    '"{campo}" cuyo valor sea {desc} según esta norma.',
)


_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")


def render_prompt(template_text: str, doc: RawDocument, texto: str | None = None) -> str:
    """Sustituye placeholders ``{{campo}}`` del template con datos del doc.

    ``texto`` permite pasar la ventana estructural en lugar del documento
    completo (obligatorio para que el ejemplo sobreviva al presupuesto).
    """
    mapping = {
        "id": doc.id,
        "rango": doc.rango,
        "titulo": doc.titulo,
        "titulo_corto": doc.short_title(),
        "organo_emisor": doc.organo_emisor,
        "fecha_publicacion": doc.fecha_publicacion,
        "materias": ", ".join(doc.materia),
        "url": doc.url,
        "texto": texto if texto is not None else doc.texto,
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
