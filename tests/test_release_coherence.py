"""La versión del paquete cuenta lo mismo en pyproject, __version__, CHANGELOG y tag.

Ejecuta ``scripts/check_release.py`` (solo stdlib) sobre los ficheros reales
del repositorio y comprueba, con textos sintéticos, que detecta cada
incoherencia que el job ``verify`` de release-hub.yml debe parar.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import date
from pathlib import Path

import adapterforge

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_release.py"
TODAY = date(2026, 10, 4)


def _load_script():
    spec = importlib.util.spec_from_file_location("check_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


check_release_mod = _load_script()

PYPROJECT_OK = '[project]\nname = "adapterforge"\nversion = "0.1.0"\n'
INIT_OK = '"""doc"""\n\n__version__ = "0.1.0"\n'
# Listo para etiquetar: [Unreleased] vacío y la sección de la versión con fecha.
CHANGELOG_OK = (
    "# Changelog\n\n## [Unreleased]\n\n"
    "## [0.1.0] - 2026-10-04\n\n### Añadido\n\n- todo\n\n"
    "[Unreleased]: https://example.org/compare/v0.1.0...HEAD\n"
    "[0.1.0]: https://example.org/releases/tag/v0.1.0\n"
)
PENDING = CHANGELOG_OK.replace("## [0.1.0] - 2026-10-04", "## [0.1.0] - YYYY-MM-DD")


def _check(changelog: str = CHANGELOG_OK, tag: str | None = None, **texts: str) -> list[str]:
    return check_release_mod.check_release(
        texts.get("pyproject", PYPROJECT_OK),
        texts.get("init", INIT_OK),
        changelog,
        tag=tag,
        today=TODAY,
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
    # Con el tag, lo único que puede faltar antes del commit de release es la
    # fecha (marcador YYYY-MM-DD); nunca entradas sueltas en [Unreleased].
    problems = check_release_mod.check_release(*texts, tag=f"v{version}")
    assert all("fecha pendiente" in problem for problem in problems), problems


def test_synthetic_inputs_pass():
    assert _check(tag="v0.1.0") == []


def test_version_is_read_from_the_project_table():
    # Otra tabla con `version` antes de [project] no la confunde (tomllib).
    pyproject = '[tool.bumpversion]\nversion = "9.9.9"\n\n' + PYPROJECT_OK
    assert check_release_mod.read_pyproject_version(pyproject) == "0.1.0"
    assert _check(pyproject=pyproject) == []
    assert check_release_mod.read_pyproject_version("[project\nversion = 1") is None
    problems = _check(pyproject='[project]\nname = "adapterforge"\n')
    assert any("falta `version`" in p for p in problems)


def test_detects_init_version_mismatch():
    problems = _check(init=INIT_OK.replace("0.1.0", "0.2.0"))
    assert len(problems) == 1 and "__version__ (0.2.0)" in problems[0]


def test_detects_changelog_without_release_section_or_date():
    no_release = "# Changelog\n\n## [Unreleased]\n\n- pendiente\n"
    assert any("ninguna sección" in p for p in _check(no_release))

    undated = CHANGELOG_OK.replace("## [0.1.0] - 2026-10-04", "## [0.1.0]")
    assert any("necesita fecha" in p for p in _check(undated))

    impossible = CHANGELOG_OK.replace("2026-10-04", "2026-13-45")
    assert any("necesita fecha" in p for p in _check(impossible))

    stale = CHANGELOG_OK.replace("## [0.1.0] - 2026-10-04", "## [0.0.9] - 2026-01-01")
    assert any("[0.0.9]" in p and "0.1.0" in p for p in _check(stale))


def test_pending_date_is_fine_until_the_tag():
    assert _check(PENDING) == []
    problems = _check(PENDING, tag="v0.1.0")
    assert len(problems) == 1 and "fecha pendiente" in problems[0]


def test_tag_rejects_a_future_date():
    tomorrow = CHANGELOG_OK.replace("2026-10-04", "2026-10-05")
    assert _check(tomorrow, tag="v0.1.0") == []  # margen de zona horaria
    later = CHANGELOG_OK.replace("2026-10-04", "2026-10-09")
    assert _check(later) == []
    problems = _check(later, tag="v0.1.0")
    assert len(problems) == 1 and "futura" in problems[0]


def test_tag_requires_an_empty_unreleased_section():
    with_entries = CHANGELOG_OK.replace(
        "## [Unreleased]\n\n", "## [Unreleased]\n\n### Añadido\n\n- algo nuevo\n\n"
    )
    assert _check(with_entries) == []
    problems = _check(with_entries, tag="v0.1.0")
    assert len(problems) == 1 and "[Unreleased] tiene entradas" in problems[0]
    # Un comentario HTML no es una entrada.
    commented = CHANGELOG_OK.replace("## [Unreleased]\n\n", "## [Unreleased]\n\n<!-- -->\n\n")
    assert _check(commented, tag="v0.1.0") == []


def test_detects_tag_that_does_not_match_the_declared_version():
    problems = _check(tag="v0.2.0")
    assert len(problems) == 1 and "v0.2.0" in problems[0] and "v0.1.0" in problems[0]
    problems = _check(tag="0.1.0")
    assert len(problems) == 1  # sin el prefijo v tampoco vale


def test_requires_unreleased_section():
    without = CHANGELOG_OK.replace("## [Unreleased]\n\n", "")
    assert _check(without) == ["CHANGELOG.md: falta la sección `## [Unreleased]`"]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_exit_codes(tmp_path):
    files = {
        "pyproject": ("pyproject.toml", PYPROJECT_OK),
        "init": ("__init__.py", INIT_OK),
        "changelog": ("CHANGELOG.md", PENDING),
    }
    paths = []
    for flag, (name, text) in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
        paths.append(f"--{flag}={tmp_path / name}")

    ok = _run(*paths)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "[check_release] OK" in ok.stdout

    pending = _run(*paths, "--tag", "v0.1.0")
    assert pending.returncode == 1
    assert "fecha pendiente" in pending.stdout

    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG_OK, encoding="utf-8")
    tagged = _run(*paths, "--tag", "v0.1.0")
    assert tagged.returncode == 0, tagged.stdout + tagged.stderr
    assert "tag v0.1.0" in tagged.stdout

    bad = _run(*paths, "--tag", "v9.9.9")
    assert bad.returncode == 1
    assert "[check_release] FALLO" in bad.stdout

    missing = _run("--changelog", str(tmp_path / "no-existe.md"))
    assert missing.returncode == 1
