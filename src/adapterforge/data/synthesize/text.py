"""Utilidades de redacción en castellano para los targets sintetizados.

Fechas en formato largo, artículos gramaticales de rangos y órganos
emisores, glosario de lenguaje llano, fronteras de frase y primera frase
con contenido de un bloque. Sin dependencias del resto del paquete: lo usan
tanto los detectores como los generadores.
"""

from __future__ import annotations

import re

from adapterforge.data.extract import RawDocument, split_sentences

_MONTHS_ES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
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

_TERMINAL_CHARS = (".", "!", "?", "»", "]", ")")


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


def _ends_sentence(text: str) -> bool:
    """Comprueba que un target de prosa termina en frontera de frase."""
    stripped = text.rstrip()
    return bool(stripped) and stripped.endswith(_TERMINAL_CHARS)


def simplify_text(text: str) -> str:
    """Aplica el glosario de lenguaje llano (sustitución determinística)."""
    result = text
    for formal, plain in PLAIN_GLOSSARY:
        result = result.replace(formal, plain)
    return result


def _first_sentence(text: str) -> str:
    """Primera frase CON CONTENIDO del texto.

    Los cuerpos de artículo suelen empezar por apartados numerados («1. El
    presente real decreto...»); el separador de frases parte «1.» como
    "frase" propia, así que se salta cualquier fragmento de menos de 4
    palabras hasta dar con una frase real.
    """
    stripped = re.sub(r"^\s*\d+\.\s+", "", text.strip())
    for sentence in split_sentences(stripped):
        if len(sentence.split()) >= 4:
            return sentence
    return stripped.strip()


def _valid_objeto(text: str) -> bool:
    """Un objeto usable: frase real y sin cortes colgantes («...redacción: «»)."""
    stripped = text.rstrip()
    return len(stripped.split()) >= 4 and not stripped.endswith(("«", ":", ";", ","))
