"""Comprueba que la versión del paquete cuenta la misma historia en todas partes.

Fuentes que deben coincidir (solo stdlib, sin instalar el paquete):

- ``pyproject.toml`` -> ``[project] version`` (leído con ``tomllib``).
- ``src/adapterforge/__init__.py`` -> ``__version__``.
- ``CHANGELOG.md`` -> la primera sección de versión (``## [X.Y.Z] - fecha``)
  debe ser esa misma versión y estar debajo de ``## [Unreleased]``. La fecha
  es la del día del tag: mientras la release se prepara puede quedar el
  marcador ``YYYY-MM-DD``.
- Opcionalmente el tag git que dispara la release (``--tag vX.Y.Z``): debe ser
  exactamente ``v`` + versión y, además, la sección necesita una fecha real
  (ni el marcador ni posterior a mañana) y ``[Unreleased]`` tiene que estar
  vacío (sus entradas van en la sección de la versión).

Lo ejecutan ``tests/test_release_coherence.py`` en cada CI (sin ``--tag``) y
el job ``verify`` de ``.github/workflows/release-hub.yml`` con el tag que
llega. Uso local::

    python scripts/check_release.py            # coherencia de ficheros
    python scripts/check_release.py --tag v0.1.0
"""

from __future__ import annotations

import argparse
import re
import tomllib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYPROJECT = ROOT / "pyproject.toml"
DEFAULT_INIT = ROOT / "src" / "adapterforge" / "__init__.py"
DEFAULT_CHANGELOG = ROOT / "CHANGELOG.md"

PENDING_DATE = "YYYY-MM-DD"
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?$")
_INIT_VERSION = re.compile(r'^__version__\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
_CHANGELOG_HEADING = re.compile(r"^## \[([^\]]+)\](?:\s*-\s*(\S+))?\s*$", re.MULTILINE)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Lo que no cuenta como entrada de [Unreleased]: comentarios HTML y las
# definiciones de enlaces del final ([Unreleased]: https://...).
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_LINK_DEFINITION = re.compile(r"^\[[^\]]+\]:[^\n]*$", re.MULTILINE)


def read_pyproject_version(text: str) -> str | None:
    """``[project] version`` si es SemVer; None si falta o el TOML no es válido."""
    try:
        version = tomllib.loads(text).get("project", {}).get("version")
    except tomllib.TOMLDecodeError:
        return None
    return version if isinstance(version, str) and _SEMVER.match(version) else None


def read_init_version(text: str) -> str | None:
    match = _INIT_VERSION.search(text)
    return match.group(1) if match else None


def read_changelog_release(text: str) -> tuple[str, str | None] | None:
    """Primera sección de versión del CHANGELOG: ``(versión, fecha)`` o None.

    ``[Unreleased]`` se salta (puede ir primero, es lo esperado); la primera
    sección con otro nombre es la última release documentada.
    """
    for match in _CHANGELOG_HEADING.finditer(text):
        name, released = match.group(1), match.group(2)
        if name.lower() == "unreleased":
            continue
        return name, released
    return None


def read_unreleased_entries(text: str) -> str | None:
    """Contenido de ``## [Unreleased]`` hasta la siguiente sección, o None si falta."""
    headings = list(_CHANGELOG_HEADING.finditer(text))
    for index, match in enumerate(headings):
        if match.group(1).lower() != "unreleased":
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = _HTML_COMMENT.sub("", text[match.end() : end])
        return _LINK_DEFINITION.sub("", body).strip()
    return None


def _parse_date(value: str) -> date | None:
    if not _DATE.match(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def check_release(
    pyproject_text: str,
    init_text: str,
    changelog_text: str,
    tag: str | None = None,
    *,
    today: date | None = None,
) -> list[str]:
    """Devuelve la lista de incoherencias (vacía si todo cuadra).

    ``today`` (por defecto, hoy en UTC) solo se usa con ``tag``: la fecha de
    la sección no puede ser posterior a mañana (margen de zona horaria).
    """
    problems: list[str] = []
    version = read_pyproject_version(pyproject_text)
    if version is None:
        problems.append("pyproject.toml: falta `version` SemVer (X.Y.Z) en [project]")
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
        name, released = release
        if version is not None and name != version:
            problems.append(
                f"CHANGELOG.md: la última versión documentada es [{name}] "
                f"pero pyproject.toml declara {version}"
            )
        if released == PENDING_DATE:
            if tag is not None:
                problems.append(
                    f"CHANGELOG.md: la sección [{name}] tiene la fecha pendiente "
                    f"({PENDING_DATE}): pon la del día del tag antes de etiquetar"
                )
        else:
            parsed = _parse_date(released or "")
            if parsed is None:
                problems.append(
                    f"CHANGELOG.md: la sección [{name}] necesita fecha AAAA-MM-DD "
                    f"({PENDING_DATE} mientras no se etiquete)"
                )
            elif tag is not None:
                limit = (today or datetime.now(UTC).date()) + timedelta(days=1)
                if parsed > limit:
                    problems.append(
                        f"CHANGELOG.md: la fecha de [{name}] ({released}) es futura: "
                        "debe ser la del día del tag"
                    )

    entries = read_unreleased_entries(changelog_text)
    if entries is None:
        problems.append("CHANGELOG.md: falta la sección `## [Unreleased]`")
    elif tag is not None and entries:
        problems.append(
            "CHANGELOG.md: [Unreleased] tiene entradas: muévelas a la sección de la "
            "versión antes de etiquetar"
        )

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
