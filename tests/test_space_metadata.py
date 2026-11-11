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
