"""Los dos sitios que fijan la versión de Gradio del Space tienen que coincidir.

Hugging Face construye el Space instalando a la vez ``space/requirements.txt``
y ``gradio[oauth,mcp]==<sdk_version>``, con el ``sdk_version`` de la cabecera
de ``space/README.md``. Si las dos versiones difieren, pip no puede resolver la
instalación y el Space queda en ``BUILD_ERROR`` sin llegar a arrancar: pasó en
el primer despliegue, con gradio 6.16.0 en requirements.txt y 6.15.1 en el
README.
"""

from __future__ import annotations

import re
from pathlib import Path

SPACE = Path(__file__).resolve().parent.parent / "space"


def _sdk_version() -> str:
    cabecera = (SPACE / "README.md").read_text(encoding="utf-8").split("---")[1]
    coincidencia = re.search(r'^sdk_version:\s*"?([^"\s]+)"?\s*$', cabecera, flags=re.MULTILINE)
    assert coincidencia, "space/README.md no declara sdk_version en su cabecera"
    return coincidencia.group(1)


def _gradio_fijado() -> str:
    requisitos = (SPACE / "requirements.txt").read_text(encoding="utf-8")
    coincidencia = re.search(r"^gradio==(\S+)\s*$", requisitos, flags=re.MULTILINE)
    assert coincidencia, "space/requirements.txt no fija gradio con =="
    return coincidencia.group(1)


def test_el_sdk_version_del_space_es_el_gradio_fijado_en_requirements():
    assert _sdk_version() == _gradio_fijado()


def test_llama_cpp_python_es_una_rueda_precompilada_fijada_por_url_y_sha256():
    """El constructor del Space no puede compilar llama-cpp-python: se queda sin
    memoria («OOMKilled»). PyPI solo publica el código fuente, así que tiene que
    instalarse la rueda de CPU que publica el propio proyecto en sus releases,
    con su sha256 para que pip compruebe lo que descarga."""
    requisitos = (SPACE / "requirements.txt").read_text(encoding="utf-8")
    lineas = [linea for linea in requisitos.splitlines() if linea.startswith("llama-cpp-python")]
    assert len(lineas) == 1, "llama-cpp-python debe aparecer una sola vez"

    coincidencia = re.fullmatch(
        r"llama-cpp-python @ https://github\.com/abetlen/llama-cpp-python/releases/download/"
        r"v(?P<etiqueta>[0-9.]+)/llama_cpp_python-(?P<version>[0-9.]+)-py3-none-"
        r"(?P<plataforma>[A-Za-z0-9_.]+)\.whl#sha256=(?P<sha256>[0-9a-f]{64})",
        lineas[0],
    )
    assert coincidencia, f"se esperaba la rueda precompilada con su sha256, no {lineas[0]!r}"
    assert coincidencia["etiqueta"] == coincidencia["version"]
    # El Space corre en Linux x86_64 con glibc: ni musl ni otra arquitectura.
    assert "manylinux" in coincidencia["plataforma"]
    assert coincidencia["plataforma"].endswith("x86_64")
