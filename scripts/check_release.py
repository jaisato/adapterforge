"""Comprueba que la versión del paquete cuenta la misma historia en todas partes.

Fuentes que deben coincidir (solo stdlib, sin instalar el paquete):

- ``pyproject.toml`` -> ``[project] version``.
- ``src/adapterforge/__init__.py`` -> ``__version__``.
- ``CHANGELOG.md`` -> la primera sección de versión (``## [X.Y.Z] - AAAA-MM-DD``)
  debe ser esa misma versión, con fecha, y debajo de ``## [Unreleased]``.
- Opcionalmente el tag git que dispara la release (``--tag vX.Y.Z``): debe ser
  exactamente ``v`` + versión.

Lo ejecutan ``tests/test_release_coherence.py`` en cada CI y el job ``verify``
de ``.github/workflows/release-hub.yml`` con el tag que llega. Uso local::

    python scripts/check_release.py            # coherencia de ficheros
    python scripts/check_release.py --tag v0.1.0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYPROJECT = ROOT / "pyproject.toml"
DEFAULT_INIT = ROOT / "src" / "adapterforge" / "__init__.py"
DEFAULT_CHANGELOG = ROOT / "CHANGELOG.md"

_SEMVER = r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?"
_PYPROJECT_VERSION = re.compile(rf'^version\s*=\s*"({_SEMVER})"\s*$', re.MULTILINE)
_INIT_VERSION = re.compile(rf'^__version__\s*=\s*"({_SEMVER})"\s*$', re.MULTILINE)
_CHANGELOG_HEADING = re.compile(r"^## \[([^\]]+)\](?:\s*-\s*(\S+))?\s*$", re.MULTILINE)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_pyproject_version(text: str) -> str | None:
    match = _PYPROJECT_VERSION.search(text)
    return match.group(1) if match else None


def read_init_version(text: str) -> str | None:
    match = _INIT_VERSION.search(text)
    return match.group(1) if match else None


def read_changelog_release(text: str) -> tuple[str, str | None] | None:
    """Primera sección de versión del CHANGELOG: ``(versión, fecha)`` o None.

    ``[Unreleased]`` se salta (puede ir primero, es lo esperado); la primera
    sección con otro nombre es la última release documentada.
    """
    for match in _CHANGELOG_HEADING.finditer(text):
        name, date = match.group(1), match.group(2)
        if name.lower() == "unreleased":
            continue
        return name, date
    return None


def check_release(
    pyproject_text: str,
    init_text: str,
    changelog_text: str,
    tag: str | None = None,
) -> list[str]:
    """Devuelve la lista de incoherencias (vacía si todo cuadra)."""
    problems: list[str] = []
    version = read_pyproject_version(pyproject_text)
    if version is None:
        problems.append("pyproject.toml: no se encuentra `version = \"X.Y.Z\"` en [project]")
    init_version = read_init_version(init_text)
    if init_version is None:
        problems.append("src/adapterforge/__init__.py: no se encuentra `__version__`")
    elif version is not None and init_version != version:
        problems.append(
            f"__version__ ({init_version}) no coincide con pyproject.toml ({version})"
        )
    release = read_changelog_release(changelog_text)
    if release is None:
        problems.append("CHANGELOG.md: no hay ninguna sección `## [X.Y.Z] - AAAA-MM-DD`")
    else:
        name, date = release
        if version is not None and name != version:
            problems.append(
                f"CHANGELOG.md: la última versión documentada es [{name}] "
                f"pero pyproject.toml declara {version}"
            )
        if date is None or not _DATE.match(date):
            problems.append(f"CHANGELOG.md: la sección [{name}] necesita fecha AAAA-MM-DD")
    if "## [Unreleased]" not in changelog_text:
        problems.append("CHANGELOG.md: falta la sección `## [Unreleased]`")
    if tag is not None and version is not None:
        expected = f"v{version}"
        if tag != expected:
            problems.append(
                f"el tag {tag} no corresponde a la versión declarada "
                f"({expected}): sube la versión en pyproject.toml, __init__.py "
                f"y CHANGELOG.md antes de etiquetar"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coherencia de versión para la release")
    parser.add_argument("--tag", default=None, help="Tag git a validar (vX.Y.Z)")
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--changelog", type=Path, default=DEFAULT_CHANGELOG)
    args = parser.parse_args(argv)

    texts: list[str] = []
    for path in (args.pyproject, args.init, args.changelog):
        if not path.exists():
            print(f"[check_release] FALLO: no existe {path}")
            return 1
        texts.append(path.read_text(encoding="utf-8"))
    problems = check_release(*texts, tag=args.tag)
    if problems:
        for problem in problems:
            print(f"[check_release] FALLO: {problem}")
        return 1
    version = read_pyproject_version(texts[0])
    suffix = f", tag {args.tag}" if args.tag else ""
    print(
        f"[check_release] OK: versión {version} coherente en pyproject, __init__ "
        f"y CHANGELOG{suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
