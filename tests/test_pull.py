"""data.pull: patrones por versión, destino seguro, omisión de existentes (sin red)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from adapterforge.data import pull

# Inventario con la forma del dataset del Hub (ruta, bytes).
REMOTE = [
    (".gitattributes", 3459),
    ("data/processed/juristune-v5_train.jsonl", 10),
    ("data/processed/juristune-v6_test.jsonl", 10),
    ("data/processed/juristune-v6_train.jsonl", 10),
    ("data/processed/juristune-v6_val.jsonl", 10),
    ("data/processed/juristune_dpo.jsonl", 10),
    ("data/processed/juristune_train.jsonl", 10),
    ("data/raw/corpus_boe.jsonl", 10),
]

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="requiere git")


def _lister(calls: list[dict] | None = None):
    def fake_list_remote(repo_id: str, **kwargs) -> list[tuple[str, int]]:
        if calls is not None:
            calls.append({"repo_id": repo_id, **kwargs})
        return REMOTE

    return fake_list_remote


def _downloader(calls: list[dict]):
    """snapshot_download falso: escribe bajo local_dir lo que pidan allow_patterns."""

    def fake_snapshot_download(**kwargs) -> str:
        calls.append(kwargs)
        for relative in kwargs["allow_patterns"]:
            target = Path(kwargs["local_dir"]) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("hub\n", encoding="utf-8")
        return kwargs["local_dir"]

    return fake_snapshot_download


def _git_repo(path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    # Las dos reglas del .gitignore del repositorio que cubren los datasets.
    (path / ".gitignore").write_text(
        f"{pull.CORPUS_FILE}\n{pull.DATASETS_GLOB}\n", encoding="utf-8"
    )
    return path


def test_allow_patterns_per_version() -> None:
    assert pull.allow_patterns("v6") == [
        "data/raw/corpus_boe.jsonl",
        "data/processed/juristune-v6_*.jsonl",
    ]
    # v1 es el prefijo sin versión (train/val/test/dpo).
    assert pull.allow_patterns("v1", corpus=False) == ["data/processed/juristune_*.jsonl"]
    assert pull.allow_patterns("all") == [
        "data/raw/corpus_boe.jsonl",
        "data/processed/juristune*.jsonl",
    ]
    # --no-corpus también vale con --version all.
    assert pull.allow_patterns("all", corpus=False) == ["data/processed/juristune*.jsonl"]
    with pytest.raises(ValueError, match="Versión desconocida"):
        pull.allow_patterns("v99")


@needs_git
def test_every_pattern_targets_files_ignored_by_the_repository(repo_root: Path) -> None:
    if not (repo_root / ".git").exists():
        pytest.skip("no es un checkout de git")
    # Lo que puede descargar cualquier versión cae en reglas del .gitignore real.
    samples = {
        pattern.replace("*", "train")
        for version in (*pull.KNOWN_VERSIONS, pull.ALL)
        for pattern in pull.allow_patterns(version)
    }
    assert pull.not_ignored([repo_root / sample for sample in sorted(samples)]) == []
    # Y nunca un fichero versionado del repositorio.
    assert pull.not_ignored([repo_root / "data/raw/sample_corpus.jsonl"]) == [
        repo_root / "data/raw/sample_corpus.jsonl"
    ]


def test_pull_downloads_only_the_version_files_into_repo_layout(tmp_path: Path) -> None:
    list_calls: list[dict] = []
    calls: list[dict] = []

    result = pull.pull(
        "v6",
        tmp_path,
        token="tok",
        lister=_lister(list_calls),
        downloader=_downloader(calls),
    )

    assert list_calls == [{"repo_id": pull.DATASET_REPO_ID, "token": "tok", "revision": None}]
    (call,) = calls
    assert call["repo_id"] == pull.DATASET_REPO_ID
    assert call["repo_type"] == "dataset"
    assert call["token"] == "tok"
    assert call["local_dir"] == str(tmp_path.resolve())
    # Rutas exactas de lo que hay en el Hub para v6 (+ corpus), nada más.
    assert call["allow_patterns"] == [
        "data/processed/juristune-v6_test.jsonl",
        "data/processed/juristune-v6_train.jsonl",
        "data/processed/juristune-v6_val.jsonl",
        "data/raw/corpus_boe.jsonl",
    ]
    assert [p.relative_to(tmp_path.resolve()).as_posix() for p in result.downloaded] == call[
        "allow_patterns"
    ]
    assert result.skipped == []


def test_pull_skips_existing_local_files_unless_forced(tmp_path: Path) -> None:
    regenerated = tmp_path / "data/processed/juristune-v6_train.jsonl"
    regenerated.parent.mkdir(parents=True)
    regenerated.write_text("regenerado en local\n", encoding="utf-8")
    calls: list[dict] = []

    result = pull.pull("v6", tmp_path, token="t", lister=_lister(), downloader=_downloader(calls))

    assert regenerated.read_text(encoding="utf-8") == "regenerado en local\n"
    assert result.skipped == [regenerated.resolve()]
    assert "data/processed/juristune-v6_train.jsonl" not in calls[0]["allow_patterns"]
    assert len(result.downloaded) == 3

    calls.clear()
    result = pull.pull(
        "v6", tmp_path, token="t", force=True, lister=_lister(), downloader=_downloader(calls)
    )
    assert regenerated.read_text(encoding="utf-8") == "hub\n"
    assert result.skipped == []
    assert len(calls[0]["allow_patterns"]) == 4


def test_pull_with_everything_present_does_not_call_the_downloader(tmp_path: Path) -> None:
    for relative, _size in REMOTE:
        if relative.endswith(".jsonl"):
            (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / relative).write_text("local\n", encoding="utf-8")
    calls: list[dict] = []

    result = pull.pull("all", tmp_path, token="t", lister=_lister(), downloader=_downloader(calls))

    assert calls == []
    assert result.downloaded == []
    assert len(result.skipped) == 7  # todo menos .gitattributes


def test_pull_fails_when_the_version_is_not_in_the_hub(tmp_path: Path) -> None:
    calls: list[dict] = []
    # El corpus sí está, pero sin splits v3 no es la versión pedida.
    with pytest.raises(pull.PullError, match="ningún fichero"):
        pull.pull("v3", tmp_path, token="t", lister=_lister(), downloader=_downloader(calls))
    assert calls == []


@needs_git
def test_pull_aborts_when_git_would_not_ignore_the_destination(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    calls: list[dict] = []

    # --dest data desde la raíz: dejaría data/data/raw/corpus_boe.jsonl sin ignorar.
    with pytest.raises(pull.PullError, match="git no ignoraría"):
        pull.pull("v6", repo / "data", token="t", lister=_lister(), downloader=_downloader(calls))
    assert calls == []
    assert not (repo / "data").exists()

    # En la raíz del repositorio todo cae en el .gitignore: se descarga.
    result = pull.pull("v6", repo, token="t", lister=_lister(), downloader=_downloader(calls))
    assert len(result.downloaded) == 4


@needs_git
def test_default_dest_is_the_git_toplevel(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    subdir = repo / "src" / "pkg"
    subdir.mkdir(parents=True)
    assert pull.default_dest(subdir).resolve() == repo.resolve()
    outside = tmp_path / "fuera"
    outside.mkdir()
    assert pull.default_dest(outside) == outside


@pytest.mark.parametrize(
    ("size", "expected"),
    [(999, "999 B"), (1_500, "1.5 kB"), (78_796_122, "78.8 MB"), (1_295_270_000, "1.3 GB")],
)
def test_human_uses_decimal_units_like_the_hub(size: int, expected: str) -> None:
    assert pull._human(size) == expected


def test_main_reports_downloaded_and_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    downloaded = tmp_path / "data/raw/corpus_boe.jsonl"
    downloaded.parent.mkdir(parents=True)
    downloaded.write_bytes(b"x" * 2_000)
    skipped = tmp_path / "data/processed/juristune-v6_train.jsonl"
    result = pull.PullResult(downloaded=[downloaded], skipped=[skipped])
    monkeypatch.setattr(pull, "pull", lambda *args, **kwargs: result)

    assert pull.main(["--dest", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "2.0 kB" in out and str(downloaded) in out
    assert f"omitido  {skipped}" in out
    assert "1 descargado(s) y 1 omitido(s)" in out


def test_main_fails_with_a_message_on_pull_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_pull(*args, **kwargs):
        raise pull.PullError("ningún fichero de x coincide")

    monkeypatch.setattr(pull, "pull", failing_pull)
    assert pull.main([]) == 1
    assert "[pull] FALLO: ningún fichero" in capsys.readouterr().out
