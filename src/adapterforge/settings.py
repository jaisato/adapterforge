"""Configuración global vía variables de entorno (prefijo ``AF_``).

Se apoya en ``pydantic-settings``: cualquier campo puede fijarse con
``AF_<CAMPO>`` o en un fichero ``.env`` del directorio de trabajo (ver
``.env.example``). La excepción es el token de Hugging Face, que acepta
también ``HF_TOKEN``, el nombre estándar del ecosistema. La configuración de
cada experimento de entrenamiento NO vive aquí sino en ``configs/*.yaml``
(ver :mod:`adapterforge.config`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Ajustes de proceso: endpoints, modelos teacher/juez, rutas y tokens."""

    model_config = SettingsConfigDict(
        env_prefix="AF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Modelos remotos (formato litellm "proveedor/modelo").
    teacher_model: str = "openai/gpt-4o-mini"
    judge_model: str = "openai/gpt-4o-mini"

    # Endpoint OpenAI-compatible (vLLM, Ollama, InferaGate...). Desde un
    # contenedor, el Ollama del host se alcanza vía host.docker.internal
    # (ver extra_hosts en compose.yaml).
    api_base: str | None = "http://host.docker.internal:11434/v1"
    api_key: str | None = None

    # Publicación en Hugging Face Hub (usuario del portfolio: Jaisato).
    hf_repo_id: str = "Jaisato/juristune-qwen3-4b-lora"
    # Token de HF: ``AF_HF_TOKEN`` o ``HF_TOKEN``, del entorno o del ``.env``.
    # Con un alias explícito pydantic-settings no aplica el prefijo AF_, así
    # que ``HF_TOKEN`` también se lee del ``.env`` (el Makefile no lo carga).
    # Orden: entorno antes que ``.env``; dentro de cada uno, AF_HF_TOKEN gana.
    hf_token: str | None = Field(
        default=None, validation_alias=AliasChoices("AF_HF_TOKEN", "HF_TOKEN")
    )

    # Tracking de experimentos (servicio mlflow de compose.yaml).
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Rutas base del repositorio.
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve el singleton de ajustes (cacheado por proceso)."""
    return Settings()


def resolve_hf_token(explicit: str | None = None) -> str | None:
    """Token de Hugging Face: el explícito o el de :class:`Settings`.

    Orden: ``explicit`` (``--token``) > ``AF_HF_TOKEN`` > ``HF_TOKEN`` del
    entorno > los mismos en ``.env``. ``None`` (también si el valor está
    vacío, como en ``.env.example``) deja que ``huggingface_hub`` use la
    sesión de ``hf auth login``.
    """
    return explicit or get_settings().hf_token or None
