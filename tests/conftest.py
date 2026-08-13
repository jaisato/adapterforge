"""Fixtures compartidos: rutas del repo y corpus de muestra validado."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapterforge.data.extract import RawDocument, load_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def corpus(repo_root: Path) -> list[RawDocument]:
    """Corpus de 12 documentos estilo BOE incluidos en el repositorio."""
    return load_corpus(repo_root / "data" / "raw" / "sample_corpus.jsonl")
