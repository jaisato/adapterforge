"""Descarga los datasets de JurisTune desde el dataset privado de HF Hub.

Los JSONL generados (corpus real del BOE y splits ``juristune-v*``) superan
el límite de 100 MB de GitHub y están ignorados por git: viven en el dataset
privado ``Jaisato/juristune-datasets`` con la MISMA disposición de carpetas
que este repositorio (``data/raw/...``, ``data/processed/...``), de modo que
la descarga deja cada fichero en el sitio que esperan el Makefile y las
configs de entrenamiento.

Uso::

    python -m adapterforge.data.pull                  # corpus BOE + splits v6
    python -m adapterforge.data.pull --version v3     # otra versión congelada
    python -m adapterforge.data.pull --version all    # todo (~1,3 GB)
    python -m adapterforge.data.pull --list           # ficheros del dataset
    make data-pull [DATA_VERSION=v5]

Requiere un token con acceso de lectura al dataset: ``HF_TOKEN`` o
``AF_HF_TOKEN`` en el entorno/``.env``, o la sesión de ``hf auth login``.
``huggingface_hub`` se importa de forma perezosa (extra ``hub`` de
``pyproject.toml``).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from adapterforge.settings import get_settings

DATASET_REPO_ID = "Jaisato/juristune-datasets"
CORPUS_FILE = "data/raw/corpus_boe.jsonl"

# Versiones congeladas del dataset instruccional publicadas en el Hub. ``v1``
# es el prefijo sin versión (``juristune_{train,val,test,dpo}.jsonl``); el
# resto siguen ``juristune-<versión>_{train,val,test}.jsonl``.
KNOWN_VERSIONS: tuple[str, ...] = ("v1", "v2", "v2b", "v3", "v4", "v5", "v6")
DEFAULT_VERSION = "v6"
ALL = "all"


def allow_patterns(version: str, *, corpus: bool = True) -> list[str]:
    """Patrones (fnmatch) de los ficheros del dataset a descargar.

    ``version`` es una de :data:`KNOWN_VERSIONS` o ``"all"``; ``corpus``
    añade el corpus BOE del que se derivan todos los splits.
    """
    if version == ALL:
        return ["data/raw/*.jsonl", "data/processed/*.jsonl"]
    if version not in KNOWN_VERSIONS:
        raise ValueError(f"Versión desconocida {version!r} (válidas: {KNOWN_VERSIONS + (ALL,)})")
    prefix = "juristune_" if version == "v1" else f"juristune-{version}_"
    patterns = [f"data/processed/{prefix}*.jsonl"]
    if corpus:
        patterns.insert(0, CORPUS_FILE)
    return patterns


def resolve_token(explicit: str | None = None) -> str | None:
    """Token de lectura: ``--token`` > ``HF_TOKEN`` > ``AF_HF_TOKEN`` > sesión."""
    return explicit or os.environ.get("HF_TOKEN") or get_settings().hf_token or None


def pull(
    version: str = DEFAULT_VERSION,
    dest: str | Path = ".",
    *,
    repo_id: str = DATASET_REPO_ID,
    token: str | None = None,
    corpus: bool = True,
    revision: str | None = None,
    downloader: Callable[..., Any] | None = None,
) -> list[Path]:
    """Descarga los ficheros de ``version`` bajo ``dest`` y devuelve sus rutas.

    ``downloader`` permite inyectar una función con la firma de
    ``huggingface_hub.snapshot_download`` (tests sin red); por defecto se
    importa de forma perezosa. ``snapshot_download`` deja sus metadatos en
    ``<dest>/.cache/huggingface/`` (ignorado por git).
    """
    patterns = allow_patterns(version, corpus=corpus)
    if downloader is None:
        from huggingface_hub import snapshot_download  # perezoso

        downloader = snapshot_download
    dest_path = Path(dest)
    downloader(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
        local_dir=str(dest_path),
        token=resolve_token(token),
    )
    downloaded = sorted({path for pattern in patterns for path in dest_path.glob(pattern)})
    return downloaded


def list_remote(repo_id: str = DATASET_REPO_ID, token: str | None = None) -> list[tuple[str, int]]:
    """``[(ruta, bytes)]`` de los ficheros del dataset en el Hub."""
    from huggingface_hub import HfApi  # perezoso

    api = HfApi(token=resolve_token(token))
    entries = api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
    return sorted(
        (entry.path, int(getattr(entry, "size", 0) or 0))
        for entry in entries
        if getattr(entry, "size", None) is not None
    )


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descarga de datasets JurisTune desde HF Hub")
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        choices=KNOWN_VERSIONS + (ALL,),
        help=f"Versión del dataset instruccional (por defecto {DEFAULT_VERSION})",
    )
    parser.add_argument("--repo-id", default=DATASET_REPO_ID)
    parser.add_argument("--dest", default=".", help="Raíz del repositorio (por defecto, cwd)")
    parser.add_argument("--revision", default=None, help="Commit/rama del dataset en el Hub")
    parser.add_argument("--token", default=None, help="Default: HF_TOKEN / AF_HF_TOKEN / sesión")
    parser.add_argument("--no-corpus", action="store_true", help="Omite data/raw/corpus_boe.jsonl")
    parser.add_argument("--list", action="store_true", help="Solo lista los ficheros remotos")
    args = parser.parse_args(argv)

    if args.list:
        files = list_remote(args.repo_id, token=args.token)
        for path, size in files:
            print(f"{_human(size):>10}  {path}")
        print(f"[pull] {len(files)} ficheros en {args.repo_id}")
        return 0

    downloaded = pull(
        args.version,
        args.dest,
        repo_id=args.repo_id,
        token=args.token,
        corpus=not args.no_corpus,
        revision=args.revision,
    )
    if not downloaded:
        print(f"[pull] FALLO: ningún fichero coincide con la versión {args.version!r}")
        return 1
    for path in downloaded:
        print(f"{_human(path.stat().st_size):>10}  {path}")
    print(f"[pull] {len(downloaded)} ficheros de {args.repo_id} ({args.version}) en {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
