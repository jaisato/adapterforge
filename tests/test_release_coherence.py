"""La versión del paquete cuenta lo mismo en pyproject, __version__, CHANGELOG y tag.

Ejecuta ``scripts/check_release.py`` (solo stdlib) sobre los ficheros reales
del repositorio y comprueba, con textos sintéticos, que detecta cada
incoherencia que el job ``verify`` de release-hub.yml debe parar.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import adapterforge

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_release.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


check_release_mod = _load_script()

PYPROJECT_OK = '[project]\nname = "adapterforge"\nversion = "0.1.0"\n'
INIT_OK = '"""doc"""\n\n__version__ = "0.1.0"\n'
CHANGELOG_OK = (
    "# Changelog\n\n## [Unreleased]\n\n- nada\n\n"
    "## [0.1.0] - 2026-10-04\n\n### Añadido\n\n- todo\n"
)


def test_repository_files_are_coherent():
    texts = [
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        (ROOT / "src" / "adapterforge" / "__init__.py").read_text(encoding="utf-8"),
        (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
    ]
    assert check_release_mod.check_release(*texts) == []
    version = check_release_mod.read_pyproject_version(texts[0])
    assert version == adapterforge.__version__
    assert check_release_mod.check_release(*texts, tag=f"v{version}") == []


def test_synthetic_inputs_pass():
    assert check_release_mod.check_release(PYPROJECT_OK, INIT_OK, CHANGELOG_OK, tag="v0.1.0") == []


def test_detects_init_version_mismatch():
    problems = check_release_mod.check_release(
        PYPROJECT_OK, INIT_OK.replace("0.1.0", "0.2.0"), CHANGELOG_OK
    )
    assert len(problems) == 1 and "__version__ (0.2.0)" in problems[0]


def test_detects_changelog_without_release_section_or_date():
    no_release = "# Changelog\n\n## [Unreleased]\n\n- pendiente\n"
    problems = check_release_mod.check_release(PYPROJECT_OK, INIT_OK, no_release)
    assert any("ninguna sección" in p for p in problems)

    undated = CHANGELOG_OK.replace("## [0.1.0] - 2026-10-04", "## [0.1.0]")
    problems = check_release_mod.check_release(PYPROJECT_OK, INIT_OK, undated)
    assert any("necesita fecha" in p for p in problems)

    stale = CHANGELOG_OK.replace("## [0.1.0] - 2026-10-04", "## [0.0.9] - 2026-01-01")
    problems = check_release_mod.check_release(PYPROJECT_OK, INIT_OK, stale)
    assert any("[0.0.9]" in p and "0.1.0" in p for p in problems)


def test_detects_tag_that_does_not_match_the_declared_version():
    problems = check_release_mod.check_release(PYPROJECT_OK, INIT_OK, CHANGELOG_OK, tag="v0.2.0")
    assert len(problems) == 1 and "v0.2.0" in problems[0] and "v0.1.0" in problems[0]
    problems = check_release_mod.check_release(PYPROJECT_OK, INIT_OK, CHANGELOG_OK, tag="0.1.0")
    assert len(problems) == 1  # sin el prefijo v tampoco vale


def test_requires_unreleased_section():
    without = CHANGELOG_OK.replace("## [Unreleased]\n\n- nada\n\n", "")
    problems = check_release_mod.check_release(PYPROJECT_OK, INIT_OK, without)
    assert problems == ["CHANGELOG.md: falta la sección `## [Unreleased]`"]


def test_cli_exit_codes(tmp_path):
    ok = subprocess.run(
        [sys.executable, str(SCRIPT), "--tag", f"v{adapterforge.__version__}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "[check_release] OK" in ok.stdout

    bad = subprocess.run(
        [sys.executable, str(SCRIPT), "--tag", "v9.9.9"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert bad.returncode == 1
    assert "[check_release] FALLO" in bad.stdout

    missing = subprocess.run(
        [sys.executable, str(SCRIPT), "--changelog", str(tmp_path / "no-existe.md")],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert missing.returncode == 1
