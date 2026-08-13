"""Extracción y normalización del corpus crudo (JSONL estilo BOE).

Cada documento del corpus tiene la estructura de una disposición publicada en
un boletín oficial: identificador, rango normativo, título, órgano emisor,
fecha, materias, URL y texto articulado ("Artículo 1. ... Artículo 2. ...").

Este módulo valida el corpus con pydantic, normaliza espacios y ofrece
utilidades estructurales (troceo por artículos, troceo por frases) que
reutilizan la síntesis offline y el filtrado.

Uso CLI::

    python -m adapterforge.data.extract \
        --input data/raw/sample_corpus.jsonl \
        --output data/interim/documents.jsonl

TODO(jairo): conector real al corpus de NormaRAG (Postgres/MinIO) y mini
conector BOE standalone (API datos abiertos) como fuentes alternativas.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adapterforge.jsonl import read_jsonl, write_jsonl

_ARTICLE_SPLIT = re.compile(r"(?=Artículo\s+\d+\.)")
_ARTICLE_HEAD = re.compile(r"^Artículo\s+(\d+)\.\s*(.*)$", re.DOTALL)
# Frontera de frase: puntuación fuerte seguida de mayúscula, comilla o dígito.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?»])\s+(?=[A-ZÁÉÍÓÚÑ«\d])")
_MULTI_SPACE = re.compile(r"[ \t]+")


class RawDocument(BaseModel):
    """Documento normativo crudo tal y como entra al pipeline."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=5, max_length=40, pattern=r"^[A-Z0-9\-]+$")
    rango: str = Field(min_length=3, max_length=60)
    titulo: str = Field(min_length=10, max_length=400)
    organo_emisor: str = Field(min_length=3, max_length=200)
    fecha_publicacion: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    materia: list[str] = Field(min_length=1, max_length=10)
    url: str = Field(min_length=10, max_length=300)
    texto: str = Field(min_length=200)

    @field_validator("texto")
    @classmethod
    def _texto_has_articles(cls, value: str) -> str:
        if "Artículo 1." not in value:
            raise ValueError("el texto debe contener al menos 'Artículo 1.'")
        return value

    def short_title(self) -> str:
        """Denominación corta de la norma: 'Real Decreto 214/2026'."""
        return self.titulo.split(",")[0].strip()

    def header(self) -> str:
        """Cabecera de contexto que se antepone al texto en los prompts."""
        return (
            f"{self.titulo}\n"
            f"{self.rango} · {self.organo_emisor} · "
            f"Publicado el {self.fecha_publicacion} · {self.id}"
        )


class Article(BaseModel):
    """Artículo individual de una disposición."""

    number: int
    heading: str  # rúbrica corta ("Objeto.") o cadena vacía
    body: str


def normalize_whitespace(text: str) -> str:
    """Colapsa espacios múltiples y recorta extremos, preservando párrafos."""
    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def split_sentences(text: str) -> list[str]:
    """Divide en frases con una heurística conservadora para texto jurídico."""
    return [chunk.strip() for chunk in _SENTENCE_BOUNDARY.split(text) if chunk.strip()]


def split_articles(texto: str) -> tuple[str, list[Article]]:
    """Separa preámbulo y artículos de un texto articulado."""
    segments = [seg.strip() for seg in _ARTICLE_SPLIT.split(texto) if seg.strip()]
    if not segments:
        return "", []
    preamble = "" if segments[0].startswith("Artículo") else segments[0]
    articles: list[Article] = []
    for segment in segments:
        match = _ARTICLE_HEAD.match(segment)
        if not match:
            continue
        number = int(match.group(1))
        rest = match.group(2).strip()
        heading = ""
        body = rest
        # La rúbrica es la primera "frase" si es corta ("Objeto.", "Ámbito de
        # aplicación."). El resto es el cuerpo del artículo.
        first_period = rest.find(".")
        if 0 < first_period <= 60:
            candidate = rest[: first_period + 1].strip()
            if len(candidate.split()) <= 7:
                heading = candidate
                body = rest[first_period + 1 :].strip()
        articles.append(Article(number=number, heading=heading, body=body))
    return preamble, articles


def load_corpus(path: str | Path) -> list[RawDocument]:
    """Lee y valida el corpus completo; falla con contexto si algo no cuadra."""
    documents: list[RawDocument] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(read_jsonl(path), start=1):
        try:
            document = RawDocument.model_validate(record)
        except Exception as exc:
            raise ValueError(f"Documento #{index} inválido en {path}: {exc}") from exc
        if document.id in seen_ids:
            raise ValueError(f"Identificador duplicado en el corpus: {document.id}")
        seen_ids.add(document.id)
        documents.append(document)
    return documents


def main(argv: list[str] | None = None) -> int:
    """CLI: valida el corpus, normaliza el texto y escribe el resultado."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help="Corpus crudo JSONL")
    parser.add_argument("--output", required=True, help="Salida JSONL normalizada")
    args = parser.parse_args(argv)

    documents = load_corpus(args.input)
    normalized = []
    total_articles = 0
    for document in documents:
        clean = document.model_copy(update={"texto": normalize_whitespace(document.texto)})
        _, articles = split_articles(clean.texto)
        total_articles += len(articles)
        normalized.append(clean.model_dump())
    count = write_jsonl(args.output, normalized)
    print(
        f"[extract] {count} documentos validados -> {args.output} "
        f"({total_articles} artículos detectados)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
