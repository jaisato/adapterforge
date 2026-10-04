"""Token de Hugging Face: HF_TOKEN/AF_HF_TOKEN del entorno o del .env (sin red).

``make data-pull`` no carga el ``.env`` en el entorno: lo lee pydantic-settings
desde el directorio de trabajo, y ``HF_TOKEN`` (sin prefijo AF_) tiene que
llegar igual que ``AF_HF_TOKEN``.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from adapterforge.data import pull
from adapterforge.serve import push_to_hub
from adapterforge.settings import Settings, get_settings, resolve_hf_token


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Directorio de trabajo vacío, sin tokens en el entorno ni ajustes cacheados."""
    monkeypatch.chdir(tmp_path)
    for name in ("HF_TOKEN", "AF_HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def _dotenv(directory: Path, text: str) -> None:
    (directory / ".env").write_text(text, encoding="utf-8")


def test_hf_token_is_read_from_dotenv(workdir: Path) -> None:
    _dotenv(workdir, "HF_TOKEN=hf-dotenv\n")
    assert Settings().hf_token == "hf-dotenv"
    _dotenv(workdir, "AF_HF_TOKEN=af-dotenv\n")
    assert Settings().hf_token == "af-dotenv"


def test_hf_token_precedence(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # En el mismo origen gana AF_HF_TOKEN...
    _dotenv(workdir, "HF_TOKEN=hf-dotenv\nAF_HF_TOKEN=af-dotenv\n")
    assert Settings().hf_token == "af-dotenv"
    # ...y el entorno gana al .env.
    monkeypatch.setenv("HF_TOKEN", "hf-env")
    assert Settings().hf_token == "hf-env"
    monkeypatch.setenv("AF_HF_TOKEN", "af-env")
    assert Settings().hf_token == "af-env"


def test_resolve_hf_token(workdir: Path) -> None:
    assert resolve_hf_token() is None  # sin token: sesión de `hf auth login`
    _dotenv(workdir, "HF_TOKEN=\n")  # vacío, como en .env.example
    get_settings.cache_clear()
    assert resolve_hf_token() is None
    _dotenv(workdir, "HF_TOKEN=hf-dotenv\n")
    get_settings.cache_clear()
    assert resolve_hf_token() == "hf-dotenv"
    assert resolve_hf_token("explicito") == "explicito"


def test_data_pull_uses_the_dotenv_token(workdir: Path) -> None:
    _dotenv(workdir, "HF_TOKEN=hf-dotenv\n")
    seen: dict[str, object] = {}

    def fake_snapshot_download(**kwargs) -> str:
        seen["download_token"] = kwargs["token"]
        return kwargs["local_dir"]

    pull.pull("v6", workdir / "out", downloader=fake_snapshot_download)
    assert seen == {"download_token": "hf-dotenv"}


def test_push_to_hub_uses_the_dotenv_token(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    _dotenv(workdir, "HF_TOKEN=hf-dotenv\n")
    tokens: list[str] = []

    class FakeApi:
        def __init__(self, token: str) -> None:
            tokens.append(token)

        def create_repo(self, **kwargs) -> None:
            pass

        def upload_folder(self, **kwargs) -> None:
            pass

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    adapter = workdir / "adapter"
    adapter.mkdir()
    metrics = workdir / "metrics.json"
    metrics.write_text(json.dumps({"experiment": "t"}), encoding="utf-8")

    push_to_hub.push(
        adapter,
        "user/repo",
        metrics_path=metrics,
        template_path=repo_root / "serve" / "templates" / "model_card.md",
    )
    assert tokens == ["hf-dotenv"]
