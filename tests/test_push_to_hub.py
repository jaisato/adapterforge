"""push_to_hub no sube artefactos de entrenamiento (sin red: HfApi falso)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from adapterforge.serve import push_to_hub


class _FakeApi:
    calls: list[dict] = []

    def __init__(self, token: str) -> None:
        self.token = token

    def create_repo(self, **kwargs) -> None:
        pass

    def upload_folder(self, **kwargs) -> None:
        _FakeApi.calls.append(kwargs)


def test_training_args_pickle_is_never_uploaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    _FakeApi.calls.clear()

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    (adapter / "training_args.bin").write_bytes(b"pickle")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"experiment": "t", "steps": 1}), encoding="utf-8")

    push_to_hub.push(
        adapter,
        "user/repo",
        metrics_path=metrics,
        template_path=repo_root / "serve" / "templates" / "model_card.md",
        token="x",
    )

    (call,) = _FakeApi.calls
    assert "training_args.bin" in call["ignore_patterns"]
    assert "checkpoint-*" in call["ignore_patterns"]
