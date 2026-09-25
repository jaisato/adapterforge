"""Tests del cliente LLM OpenAI-compatible con respx (sin red real)."""

from __future__ import annotations

import httpx
import pytest
import respx

from adapterforge.llm import LlmError, chat_complete

API_BASE = "http://gateway.test/v1"


@respx.mock
def test_chat_complete_openai_backend_happy_path():
    route = respx.post(f"{API_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "El plazo es de tres meses."}}
                ]
            },
        )
    )
    content = chat_complete(
        [{"role": "user", "content": "¿Plazo?"}],
        "juristune",
        backend="openai",
        api_base=API_BASE,
        api_key="secreto",
        temperature=0.0,
    )
    assert content == "El plazo es de tres meses."
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer secreto"
    import json

    payload = json.loads(request.content)
    assert payload["model"] == "juristune"
    assert payload["messages"][0]["content"] == "¿Plazo?"


@respx.mock
def test_chat_complete_raises_on_http_error_and_malformed():
    respx.post(f"{API_BASE}/chat/completions").mock(return_value=httpx.Response(500))
    with pytest.raises(LlmError, match="Fallo HTTP"):
        chat_complete(
            [{"role": "user", "content": "x"}], "m", backend="openai", api_base=API_BASE
        )
    respx.post(f"{API_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    with pytest.raises(LlmError, match="malformada"):
        chat_complete(
            [{"role": "user", "content": "x"}], "m", backend="openai", api_base=API_BASE
        )


def test_openai_backend_requires_api_base_and_valid_backend():
    with pytest.raises(ValueError, match="requiere api_base"):
        chat_complete([{"role": "user", "content": "x"}], "m", backend="openai")
    with pytest.raises(ValueError, match="Backend LLM desconocido"):
        chat_complete([{"role": "user", "content": "x"}], "m", backend="magico")


@pytest.mark.parametrize("body", [b"", b"<html>upstream error</html>", b"{invalid"])
@respx.mock
def test_success_response_with_invalid_json_raises_llm_error(body):
    respx.post(f"{API_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, content=body)
    )
    with pytest.raises(LlmError, match="JSON"):
        chat_complete(
            [{"role": "user", "content": "x"}], "m", backend="openai", api_base=API_BASE
        )
