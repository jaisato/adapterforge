"""Configuración global vía variables de entorno (prefijo ``AF_``).

Se apoya en ``pydantic-settings``: cualquier campo puede fijarse con
``AF_<CAMPO>`` o en un fichero ``.env`` (ver ``.env.example``). La
configuración de cada experimento de entrenamiento NO vive aquí sino en
``configs/*.yaml`` (ver :mod:`adapterforge.config`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
    hf_token: str | None = None

    # Tracking de experimentos (servicio mlflow de compose.yaml).
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Rutas base del repositorio.
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve el singleton de ajustes (cacheado por proceso)."""
    return Settings()
