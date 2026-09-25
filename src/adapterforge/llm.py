"""Cliente LLM unificado con dos backends.

- ``openai``: cliente OpenAI-compatible mínimo sobre ``httpx`` (dep. core).
  Sirve para vLLM, Ollama (``/v1``) o cualquier gateway compatible, y es el
  path que se testea con ``respx`` sin red.
- ``litellm``: multiproveedor (OpenAI, Groq, Mistral, ollama/...). Se importa
  de forma perezosa porque su import es costoso.

Los módulos de síntesis, filtrado LLM-judge y evaluación usan esta única
función; así el teacher/juez es configurable sin tocar código.
"""

from __future__ import annotations

from typing import Any

import httpx

Message = dict[str, str]

DEFAULT_TIMEOUT_S = 120.0


class LlmError(RuntimeError):
    """Error de invocación del LLM (transporte o respuesta malformada)."""


def chat_complete(
    messages: list[Message],
    model: str,
    *,
    backend: str = "litellm",
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Devuelve el contenido del primer choice de una chat completion.

    ``backend="openai"`` exige ``api_base`` (endpoint OpenAI-compatible);
    ``backend="litellm"`` delega la resolución de proveedor en litellm.
    """
    if backend == "openai":
        if not api_base:
            raise ValueError("backend='openai' requiere api_base (endpoint OpenAI-compatible)")
        return _chat_complete_openai_compatible(
            messages,
            model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    if backend == "litellm":
        return _chat_complete_litellm(
            messages,
            model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    raise ValueError(f"Backend LLM desconocido: {backend!r} (usa 'openai' o 'litellm')")


def _chat_complete_openai_compatible(
    messages: list[Message],
    model: str,
    *,
    api_base: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> str:
    """POST directo a ``{api_base}/chat/completions`` vía httpx."""
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout_s)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LlmError(f"Fallo HTTP contra {url}: {exc}") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise LlmError(f"Respuesta malformada de {url}: JSON inválido") from exc
    return _extract_content(payload, source=url)


def _chat_complete_litellm(
    messages: list[Message],
    model: str,
    *,
    api_base: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> str:
    """Invoca ``litellm.completion`` (import perezoso)."""
    import litellm  # import perezoso: es lento y solo se necesita aquí

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_s,
        )
    except Exception as exc:  # litellm expone jerarquías propias por proveedor
        raise LlmError(f"Fallo de litellm con el modelo {model!r}: {exc}") from exc
    payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    return _extract_content(payload, source=f"litellm:{model}")


def _extract_content(payload: dict[str, Any], *, source: str) -> str:
    """Extrae ``choices[0].message.content`` con validación defensiva."""
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError(f"Respuesta malformada de {source}: sin choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise LlmError(f"Respuesta vacía de {source}")
    return content.strip()
