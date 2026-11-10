"""Recorte del historial de la demo del Space (space/app.py) a la ventana del modelo.

El módulo importa gradio, huggingface_hub y llama_cpp y descarga el GGUF al
importarse: aquí se sustituyen por dobles para probar ``_fit_context`` sin red.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SPACE_APP = Path(__file__).resolve().parent.parent / "space" / "app.py"


@pytest.fixture
def space_app(monkeypatch):
    class _Llama:
        def __init__(self, *args, **kwargs):
            pass

        def tokenize(self, data, add_bos=False, special=True):
            return data.split()

    gradio = types.ModuleType("gradio")
    gradio.ChatInterface = lambda **kwargs: types.SimpleNamespace(launch=lambda: None)
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kwargs: "/tmp/modelo.gguf"
    llama_cpp = types.ModuleType("llama_cpp")
    llama_cpp.Llama = _Llama
    for name, module in {"gradio": gradio, "huggingface_hub": hub, "llama_cpp": llama_cpp}.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("space_app_bajo_prueba", SPACE_APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _turnos(n: int, palabras: int = 10) -> list[dict]:
    texto = " ".join(["palabra"] * palabras)
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": texto} for i in range(n)]


def test_descarta_los_turnos_antiguos_y_empieza_por_el_usuario(space_app):
    system = {"role": "system", "content": "sistema"}
    mensajes = [system, *_turnos(6), {"role": "user", "content": "pregunta final"}]

    def contar(texto: str) -> int:
        return len(texto.split())

    # Cada turno cuesta 10 + 8; el system 1 + 8 y el último 2 + 8.
    recortado = space_app._fit_context(mensajes, contar, budget=60)
    assert recortado[0] is system
    assert recortado[-1]["content"] == "pregunta final"
    assert recortado[1]["role"] == "user"
    total = sum(contar(m["content"]) + space_app._PER_MESSAGE_OVERHEAD for m in recortado)
    assert total <= 60
    # Sin recorte si ya cabe
    assert space_app._fit_context(mensajes, contar, budget=10_000) == mensajes


def test_el_ultimo_mensaje_se_conserva_aunque_no_quepa(space_app):
    system = {"role": "system", "content": "sistema"}
    largo = {"role": "user", "content": " ".join(["x"] * 500)}
    recortado = space_app._fit_context([system, *_turnos(4), largo], lambda t: len(t.split()), 50)
    assert recortado == [system, largo]


def test_cada_mensaje_se_tokeniza_una_sola_vez(space_app):
    """El historial lo controla el cliente del endpoint público: el recorte es lineal."""
    llamadas = 0

    def contar(texto: str) -> int:
        nonlocal llamadas
        llamadas += 1
        return len(texto.split())

    mensajes = [{"role": "system", "content": "sistema"}, *_turnos(2000)]
    recortado = space_app._fit_context(mensajes, contar, budget=200)
    assert llamadas == len(mensajes)
    assert len(recortado) < len(mensajes)
