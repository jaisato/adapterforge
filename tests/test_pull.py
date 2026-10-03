"""data.pull: patrones por versión, resolución del token y descarga (sin red)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from adapterforge.data import pull


def test_allow_patterns_per_version() -> None:
    assert pull.allow_patterns("v6") == [
        "data/raw/corpus_boe.jsonl",
        "data/processed/juristune-v6_*.jsonl",
    ]
    # v1 es el prefijo sin versión (train/val/test/dpo).
    assert pull.allow_patterns("v1", corpus=False) == ["data/processed/juristune_*.jsonl"]
    assert pull.allow_patterns("all") == ["data/raw/*.jsonl", "data/processed/*.jsonl"]
    with pytest.raises(ValueError, match="Versión desconocida"):
        pull.allow_patterns("v99")


def test_resolve_token_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    # Settings falsos: el test no debe depender de un .env local.
    settings = SimpleNamespace(hf_token=None)
    monkeypatch.setattr(pull, "get_settings", lambda: settings)
    monkeypatch.setenv("HF_TOKEN", "hf-env")
    assert pull.resolve_token("explicito") == "explicito"
    assert pull.resolve_token() == "hf-env"
    monkeypatch.delenv("HF_TOKEN")
    settings.hf_token = "af-env"
    assert pull.resolve_token() == "af-env"
    settings.hf_token = None
    # Sin token explícito, huggingface_hub usa la sesión de `hf auth login`.
    assert pull.resolve_token() is None


def test_pull_downloads_into_repo_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs) -> str:
        calls.append(kwargs)
        # El Hub replica la disposición del repo: data/raw y data/processed.
        for relative in (
            "data/raw/corpus_boe.jsonl",
            "data/processed/juristune-v6_train.jsonl",
            "data/processed/juristune-v6_val.jsonl",
            "data/processed/juristune-v6_test.jsonl",
            "data/processed/juristune-v5_train.jsonl",  # otra versión: no cuenta
        ):
            target = Path(kwargs["local_dir"]) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}\n", encoding="utf-8")
        return kwargs["local_dir"]

    downloaded = pull.pull("v6", tmp_path, token="tok", downloader=fake_snapshot_download)

    (call,) = calls
    assert call["repo_id"] == pull.DATASET_REPO_ID
    assert call["repo_type"] == "dataset"
    assert call["token"] == "tok"
    assert call["allow_patterns"] == pull.allow_patterns("v6")
    assert [p.relative_to(tmp_path).as_posix() for p in downloaded] == [
        "data/processed/juristune-v6_test.jsonl",
        "data/processed/juristune-v6_train.jsonl",
        "data/processed/juristune-v6_val.jsonl",
        "data/raw/corpus_boe.jsonl",
    ]


def test_main_fails_when_nothing_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pull, "pull", lambda *args, **kwargs: [])
    assert pull.main(["--dest", str(tmp_path)]) == 1
