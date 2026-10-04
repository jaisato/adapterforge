"""Descarga los datasets de JurisTune desde el dataset privado de HF Hub.

Los JSONL generados (corpus real del BOE y splits ``juristune-v*``) superan
el límite de 100 MB de GitHub y están ignorados por git: viven en el dataset
privado ``Jaisato/juristune-datasets`` con la MISMA disposición de carpetas
que este repositorio (``data/raw/...``, ``data/processed/...``), de modo que
la descarga deja cada fichero en el sitio que esperan el Makefile y las
configs de entrenamiento.

Uso (desde el repositorio; el ``.env`` se lee del directorio actual)::

    python -m adapterforge.data.pull                  # corpus BOE + splits v6
    python -m adapterforge.data.pull --version v3     # otra versión congelada
    python -m adapterforge.data.pull --version all    # todo (~1,3 GB)
    python -m adapterforge.data.pull --list           # ficheros del dataset
    make data-pull [DATA_VERSION=v5] [FORCE=1]

Salvaguardas:

- El destino por defecto es la raíz del repositorio git del directorio
  actual, no el propio directorio actual, y la descarga se aborta antes de
  empezar si git no ignoraría alguno de los ficheros en su destino: un
  ``--dest`` equivocado no deja cientos de MB listos para ``git add``.
- Solo se descargan ficheros que casan con los patrones de ``.gitignore``
  (``data/raw/corpus_boe.jsonl``, ``data/processed/juristune*.jsonl``):
  nunca se toca un fichero versionado.
- Los ficheros que ya existen en local se omiten salvo ``--force``: unos
  ``juristune-v6_*.jsonl`` regenerados con ``make data-v6-teacher`` no se
  sustituyen en silencio por los del Hub.

Requiere un token con acceso de lectura al dataset: ``AF_HF_TOKEN`` o
``HF_TOKEN`` en el entorno o en el ``.env``, o la sesión de
``hf auth login`` (ver :func:`adapterforge.settings.resolve_hf_token`).
``huggingface_hub`` se importa de forma perezosa (extra ``hub`` de
``pyproject.toml``).
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from adapterforge.settings import resolve_hf_token

DATASET_REPO_ID = "Jaisato/juristune-datasets"
# Los mismos patrones que ignora .gitignore: lo que se descarga nunca coincide
# con un fichero versionado (sample_corpus.jsonl, sample_*.jsonl, sft_*.jsonl).
CORPUS_FILE = "data/raw/corpus_boe.jsonl"
DATASETS_GLOB = "data/processed/juristune*.jsonl"

# Versiones congeladas del dataset instruccional publicadas en el Hub. ``v1``
# es el prefijo sin versión (``juristune_{train,val,test,dpo}.jsonl``); el
# resto siguen ``juristune-<versión>_{train,val,test}.jsonl``.
KNOWN_VERSIONS: tuple[str, ...] = ("v1", "v2", "v2b", "v3", "v4", "v5", "v6")
DEFAULT_VERSION = "v6"
ALL = "all"


class PullError(RuntimeError):
    """La descarga no puede hacerse (o no de forma segura); el CLI sale con 1."""


@dataclass
class PullResult:
    """Rutas locales descargadas y omitidas (ya existían; ``force`` las sustituye)."""

    downloaded: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)


def allow_patterns(version: str, *, corpus: bool = True) -> list[str]:
    """Patrones (fnmatch) de los ficheros del dataset a descargar.

    ``version`` es una de :data:`KNOWN_VERSIONS` o ``"all"`` (todos los
    ``juristune*``); ``corpus`` añade el corpus BOE del que se derivan todos
    los splits, también con ``"all"``.
    """
    if version == ALL:
        datasets = DATASETS_GLOB
    elif version in KNOWN_VERSIONS:
        prefix = "juristune_" if version == "v1" else f"juristune-{version}_"
        datasets = f"data/processed/{prefix}*.jsonl"
    else:
        raise ValueError(f"Versión desconocida {version!r} (válidas: {KNOWN_VERSIONS + (ALL,)})")
    return [CORPUS_FILE, datasets] if corpus else [datasets]


def default_dest(cwd: Path | None = None) -> Path:
    """Raíz del repositorio git que contiene ``cwd`` (o ``cwd`` si no hay repo)."""
    cwd = Path.cwd() if cwd is None else cwd
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # git no instalado
        return cwd
    toplevel = result.stdout.strip()
    return Path(toplevel) if result.returncode == 0 and toplevel else cwd


def not_ignored(paths: list[Path]) -> list[Path] | None:
    """Rutas de ``paths`` que git NO ignora (las versionaría un ``git add``).

    Las rutas pueden no existir todavía. Devuelve ``None`` si no están dentro
    de un repositorio git (o no hay git): fuera de un repo no hay riesgo de
    subirlas por error. Un fichero versionado cuenta como no ignorado.
    """
    if not paths:
        return []
    anchor = paths[0].parent
    while not anchor.is_dir():
        anchor = anchor.parent
    try:
        result = subprocess.run(
            # -v -n: un registro (origen, línea, patrón, ruta) por cada ruta,
            # vacío si ninguna regla la ignora; -z evita el escapado de rutas.
            ["git", "-C", str(anchor), "check-ignore", "--stdin", "-z", "-v", "-n"],
            input="".join(f"{path}\0" for path in paths),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # git no instalado
        return None
    if result.returncode not in (0, 1):  # 128: fuera de un repositorio git
        return None
    fields = result.stdout.split("\0")
    records = [fields[i : i + 4] for i in range(0, len(fields) - 1, 4)]
    # Sin origen: ninguna regla la ignora; patrón con "!": una regla la des-ignora.
    return [
        Path(path)
        for source, _line, pattern, path in records
        if not source or pattern.startswith("!")
    ]


def list_remote(
    repo_id: str = DATASET_REPO_ID, token: str | None = None, revision: str | None = None
) -> list[tuple[str, int]]:
    """``[(ruta, bytes)]`` de los ficheros del dataset en el Hub."""
    from huggingface_hub import HfApi  # perezoso
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi(token=token)
    try:
        entries = list(
            api.list_repo_tree(repo_id, repo_type="dataset", recursive=True, revision=revision)
        )
    except RepositoryNotFoundError as exc:
        raise PullError(
            f"{repo_id} no existe o el token no tiene acceso de lectura: define HF_TOKEN o "
            "AF_HF_TOKEN (entorno o .env) o inicia sesión con `hf auth login`"
        ) from exc
    return sorted(
        (entry.path, int(getattr(entry, "size", 0) or 0))
        for entry in entries
        if getattr(entry, "size", None) is not None
    )


def pull(
    version: str = DEFAULT_VERSION,
    dest: str | Path | None = None,
    *,
    repo_id: str = DATASET_REPO_ID,
    token: str | None = None,
    corpus: bool = True,
    revision: str | None = None,
    force: bool = False,
    lister: Callable[..., list[tuple[str, int]]] | None = None,
    downloader: Callable[..., Any] | None = None,
) -> PullResult:
    """Descarga los ficheros de ``version`` bajo ``dest`` (por defecto, la raíz del repo).

    1. Lista el dataset en el Hub y se queda con lo que casa con
       :func:`allow_patterns`; :class:`PullError` si no hay ningún split de
       la versión (el corpus solo no basta).
    2. :class:`PullError` también si git no ignoraría alguno de esos
       ficheros en su destino (``dest`` que no es la raíz del repositorio).
    3. Omite los que ya existen en local salvo ``force``.
    4. Descarga el resto con ``snapshot_download``, que deja sus metadatos en
       ``<dest>/.cache/huggingface/`` (ignorado por git).

    ``lister`` y ``downloader`` permiten inyectar :func:`list_remote` y una
    función con la firma de ``huggingface_hub.snapshot_download`` (tests sin
    red); por defecto se usan esas dos.
    """
    patterns = allow_patterns(version, corpus=corpus)
    token = resolve_hf_token(token)
    dest_path = (default_dest() if dest is None else Path(dest)).resolve()

    remote = (lister or list_remote)(repo_id, token=token, revision=revision)
    selected = sorted(
        path for path, _size in remote if any(fnmatchcase(path, p) for p in patterns)
    )
    # El último patrón es el de los splits: el corpus solo no cuenta como la versión.
    if not any(fnmatchcase(path, patterns[-1]) for path in selected):
        raise PullError(
            f"ningún fichero de {repo_id} coincide con la versión {version!r} ({patterns[-1]})"
        )

    unignored = not_ignored([dest_path / path for path in selected])
    if unignored:
        listed = ", ".join(str(path) for path in unignored[:3])
        raise PullError(
            f"git no ignoraría {len(unignored)} fichero(s) en {dest_path} ({listed}...): "
            "--dest debe ser la raíz del repositorio (o un directorio fuera de git)"
        )

    result = PullResult()
    to_download = []
    for path in selected:
        if (dest_path / path).exists() and not force:
            result.skipped.append(dest_path / path)
        else:
            to_download.append(path)
    if not to_download:
        return result

    if downloader is None:
        from huggingface_hub import snapshot_download  # perezoso

        downloader = snapshot_download
    downloader(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        # Rutas exactas: ni los omitidos ni nada fuera de lo listado.
        allow_patterns=to_download,
        local_dir=str(dest_path),
        token=token,
    )
    result.downloaded = [dest_path / path for path in to_download]
    missing = [path for path in result.downloaded if not path.is_file()]
    if missing:
        raise PullError(f"la descarga terminó sin {', '.join(map(str, missing))}")
    return result


def _human(size: int) -> str:
    """Tamaño en unidades decimales (1 MB = 10⁶ bytes), las mismas que el Hub."""
    value = float(size)
    for unit in ("B", "kB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            break
        value /= 1000
    return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descarga de datasets JurisTune desde HF Hub")
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        choices=KNOWN_VERSIONS + (ALL,),
        help=f"Versión del dataset instruccional (por defecto {DEFAULT_VERSION})",
    )
    parser.add_argument("--repo-id", default=DATASET_REPO_ID)
    parser.add_argument(
        "--dest",
        default=None,
        help="Raíz del repositorio donde dejar data/ (por defecto, la del repo git actual)",
    )
    parser.add_argument("--revision", default=None, help="Commit/rama del dataset en el Hub")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sustituye los ficheros que ya existen en local (por defecto se omiten)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Desaconsejado (queda en el historial y en `ps`): mejor HF_TOKEN o "
        "AF_HF_TOKEN en el entorno o en .env, o `hf auth login`",
    )
    parser.add_argument(
        "--no-corpus",
        action="store_true",
        help="Omite data/raw/corpus_boe.jsonl (también con --version all)",
    )
    parser.add_argument("--list", action="store_true", help="Solo lista los ficheros remotos")
    args = parser.parse_args(argv)

    try:
        if args.list:
            files = list_remote(
                args.repo_id, token=resolve_hf_token(args.token), revision=args.revision
            )
            for path, size in files:
                print(f"{_human(size):>10}  {path}")
            print(f"[pull] {len(files)} ficheros en {args.repo_id}")
            return 0
        result = pull(
            args.version,
            args.dest,
            repo_id=args.repo_id,
            token=args.token,
            corpus=not args.no_corpus,
            revision=args.revision,
            force=args.force,
        )
    except PullError as exc:
        print(f"[pull] FALLO: {exc}")
        return 1
    except ModuleNotFoundError as exc:
        if exc.name != "huggingface_hub":
            raise
        print('[pull] FALLO: falta huggingface_hub; instala el extra: pip install -e ".[hub]"')
        return 1

    for path in result.downloaded:
        print(f"{_human(path.stat().st_size):>10}  {path}")
    for path in result.skipped:
        print(f"{'omitido':>10}  {path} (ya existe; --force para sustituirlo)")
    print(
        f"[pull] {len(result.downloaded)} descargado(s) y {len(result.skipped)} omitido(s) "
        f"de {args.repo_id} ({args.version})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
