"""Demo del Space (space/app.py): recorte del historial y uso del modelo.

El módulo importa gradio, huggingface_hub y llama_cpp y descarga el GGUF al
importarse: aquí se sustituyen por dobles para probarlo sin red ni modelo.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
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
    gradio.ChatInterface = lambda **kwargs: types.SimpleNamespace(
        launch=lambda: None, kwargs=kwargs
    )
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


def test_las_respuestas_simultaneas_entran_de_una_en_una_en_el_modelo(space_app):
    """llama.cpp aborta el proceso si dos hilos usan a la vez el mismo contexto
    (``ggml_abort`` en ``set_rows``). Pasó en el Space: Gradio llama a ``respond``
    desde varios hilos (el chat, los ejemplos, la API pública)."""
    dentro = 0
    maximo = 0
    contador = threading.Lock()

    def usar_el_modelo() -> None:
        nonlocal dentro, maximo
        with contador:
            dentro += 1
            maximo = max(maximo, dentro)
        time.sleep(0.02)
        with contador:
            dentro -= 1

    def tokenizar(data, add_bos=False, special=True):
        usar_el_modelo()
        return data.split()

    def generar(**kwargs):
        usar_el_modelo()
        return {"choices": [{"message": {"content": "respuesta"}}]}

    space_app.llm.tokenize = tokenizar
    space_app.llm.create_chat_completion = generar
    respuestas: list[str] = []
    hilos = [
        threading.Thread(
            target=lambda i=i: respuestas.append(space_app.respond(f"pregunta {i}", []))
        )
        for i in range(6)
    ]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join()

    assert respuestas == ["respuesta"] * 6
    assert maximo == 1


def test_un_fallo_al_generar_no_deja_el_modelo_bloqueado(space_app):
    def falla(**kwargs):
        raise RuntimeError("fallo del modelo")

    space_app.llm.create_chat_completion = falla
    with pytest.raises(RuntimeError):
        space_app.respond("pregunta", [])

    space_app.llm.create_chat_completion = lambda **kwargs: {
        "choices": [{"message": {"content": "respuesta"}}]
    }
    assert space_app.respond("otra pregunta", []) == "respuesta"


def test_los_ejemplos_no_se_calculan_todos_al_arrancar(space_app):
    """En Spaces, Gradio calcula por defecto todos los ejemplos al arrancar: tres
    generaciones seguidas en la CPU gratuita antes de atender a nadie. Con
    ``lazy`` cada ejemplo se calcula la primera vez que alguien lo usa."""
    assert space_app.demo.kwargs["cache_examples"] is True
    assert space_app.demo.kwargs["cache_mode"] == "lazy"
