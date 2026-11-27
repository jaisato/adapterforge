"""Tests de las salvaguardas de entrenamiento (template, NaN-abort, config v3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from adapterforge.config import load_experiment_config
from adapterforge.train.common import (
    OFFICIAL_CHAT_TEMPLATE_PATH,
    git_sha,
    load_official_chat_template,
)
from adapterforge.train.sft import _assert_clean_render


def test_official_template_is_versioned_and_clean():
    assert OFFICIAL_CHAT_TEMPLATE_PATH.exists()
    template = load_official_chat_template()
    assert "<think>" not in template
    assert "<|im_start|>" in template
    assert len(template) > 2000  # el oficial de 2507 tiene ~2630 chars


def test_assert_clean_render_catches_think_contamination():
    good = "<|im_start|>system\nS<|im_end|>\n<|im_start|>assistant\nR<|im_end|>\n"
    _assert_clean_render([good])  # no lanza
    bad = good.replace("assistant\nR", "assistant\n<think>\n\n</think>\n\nR")
    with pytest.raises(SystemExit):
        _assert_clean_render([bad])
    with pytest.raises(SystemExit):
        _assert_clean_render(["render sin turno de assistant"])


class _State:
    global_step = 341


class _Control:
    should_training_stop = False
    should_save = True


def test_abort_on_non_finite_loss_flags_and_stops():
    # El callback hereda de transformers.TrainerCallback, que llega con el extra
    # `train`: sin transformers este test se salta, en vez de romper la recogida
    # de todo el fichero.
    pytest.importorskip("transformers")
    from adapterforge.train.callbacks import AbortOnNonFiniteLoss

    callback = AbortOnNonFiniteLoss()
    control = _Control()
    callback.on_log(None, _State(), control, logs={"loss": 1.5})
    assert not callback.diverged
    callback.on_log(None, _State(), control, logs={"loss": float("nan")})
    assert callback.diverged
    assert callback.bad_step == 341
    assert control.should_training_stop is True
    assert control.should_save is False


def test_abort_on_non_finite_loss_turns_off_the_trainer_nan_filter():
    # Sin torch (el CI): lo que hace el callback al empezar el entrenamiento.
    pytest.importorskip("transformers")
    from adapterforge.train.callbacks import AbortOnNonFiniteLoss

    args = SimpleNamespace(logging_nan_inf_filter=True)
    AbortOnNonFiniteLoss().on_train_begin(args, _State(), _Control())
    assert args.logging_nan_inf_filter is False


def test_abort_on_non_finite_loss_sees_a_nan_through_the_real_trainer_loop(tmp_path):
    """TrainingArguments (y SFTConfig/DPOConfig) traen logging_nan_inf_filter=True: el
    bucle de Trainer cambia una loss NaN por la media de la ventana ANTES de on_log, así
    que el callback nunca la veía y el adapter divergido se guardaba."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from adapterforge.train.callbacks import AbortOnNonFiniteLoss

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            loss = (self.w * x).sum()
            return {"loss": loss * float("nan") if self.calls >= 3 else loss}

    class Rows(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, i):
            return {"x": torch.ones(1)}

    args = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=6,
        logging_steps=1,
        per_device_train_batch_size=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        disable_tqdm=True,
        dataloader_pin_memory=False,
    )
    assert args.logging_nan_inf_filter is True  # el valor por defecto que tapaba la NaN
    guard = AbortOnNonFiniteLoss(label="dpo")
    trainer = transformers.Trainer(
        model=Tiny(), args=args, train_dataset=Rows(), callbacks=[guard]
    )
    trainer.train()

    assert guard.diverged
    assert guard.bad_step == 3
    assert trainer.state.global_step == 3  # se para en el primer paso con NaN


def test_git_sha_available_in_repo():
    sha = git_sha()
    assert sha is None or (isinstance(sha, str) and len(sha) >= 7)


def test_v3_config_loads_with_validation_split():
    cfg = load_experiment_config("configs/qlora_qwen3-4b_v3.yaml")
    assert cfg.val_dataset_path is not None
    assert "juristune-v3_val" in str(cfg.val_dataset_path)
    assert cfg.lora.dropout == 0.05
    assert cfg.seq_len == 2048


# ---------------------------------------------------------------------------
# DPO: mismas salvaguardas que SFT (template oficial, render limpio, NaN-abort)
# con torch/datasets/trl simulados: sin GPU ni descargas.
# ---------------------------------------------------------------------------

_THINKING_TEMPLATE = "{{ '<think>' }} template thinking del mirror de Unsloth"


class _FakeTokenizer:
    """Renderiza ChatML; con un template thinking inyecta <think> como el mirror."""

    def __init__(self) -> None:
        self.chat_template = _THINKING_TEMPLATE
        self.saved_to: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        think = "<think>\n\n</think>\n\n" if "<think>" in self.chat_template else ""
        return "".join(
            f"<|im_start|>{m['role']}\n{think if m['role'] == 'assistant' else ''}"
            f"{m['content']}<|im_end|>\n"
            for m in messages
        )

    def save_pretrained(self, path):
        self.saved_to.append(path)


class _FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = sorted(rows[0])

    def remove_columns(self, columns):
        return _FakeDataset([{k: v for k, v in r.items() if k not in columns} for r in self.rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _install_fake_dpo_stack(monkeypatch, losses):
    """Instala torch/datasets/trl simulados y devuelve el registro del trainer."""
    import dataclasses
    import sys
    import types

    pytest.importorskip("transformers")  # AbortOnNonFiniteLoss hereda de TrainerCallback
    import adapterforge.train.callbacks  # noqa: F401 — antes de simular torch
    import adapterforge.train.runtime as runtime

    record: dict = {}
    tokenizer = _FakeTokenizer()
    record["tokenizer"] = tokenizer
    monkeypatch.setattr(
        runtime, "load_model_and_tokenizer", lambda cfg: (object(), tokenizer, False)
    )

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    rows = [
        {
            "prompt": [
                {"role": "system", "content": "Eres JurisTune."},
                {"role": "user", "content": f"Pregunta {i}"},
            ],
            "chosen": [{"role": "assistant", "content": "Respuesta anclada."}],
            "rejected": [{"role": "assistant", "content": "Respuesta inventada."}],
            "meta": {"id": i},
        }
        for i in range(3)
    ]
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: _FakeDataset(rows)
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    @dataclasses.dataclass
    class DPOConfig:
        output_dir: str = ""
        beta: float = 0.1
        max_length: int = 1024

    class DPOTrainer:
        def __init__(self, *, model, args, train_dataset, processing_class, callbacks):
            record["template_seen"] = processing_class.chat_template
            record["callbacks"] = callbacks
            record["saved"] = []
            self.state = types.SimpleNamespace(global_step=0, log_history=[])

        def train(self):
            control = types.SimpleNamespace(should_training_stop=False, should_save=True)
            for step, loss in enumerate(losses, start=1):
                self.state.global_step = step
                self.state.log_history.append({"loss": loss})
                for callback in record["callbacks"]:
                    callback.on_log(None, self.state, control, logs={"loss": loss})
                if control.should_training_stop:
                    break

        def save_model(self, path):
            record["saved"].append(path)

    trl = types.ModuleType("trl")
    trl.DPOConfig, trl.DPOTrainer = DPOConfig, DPOTrainer
    monkeypatch.setitem(sys.modules, "trl", trl)
    return record


def _dpo_config(tmp_path):
    from adapterforge.config import ExperimentConfig

    return ExperimentConfig(
        name="dpo-test",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        dataset_path=tmp_path / "dpo.jsonl",
        output_dir=tmp_path / "out",
    )


def test_dpo_forces_official_template_and_saves(tmp_path, monkeypatch):
    import json

    from adapterforge.train.dpo import run_dpo

    record = _install_fake_dpo_stack(monkeypatch, losses=[0.69, 0.55])
    metrics = run_dpo(_dpo_config(tmp_path))

    assert record["template_seen"] == load_official_chat_template()
    assert record["saved"] == [str(tmp_path / "out" / "adapter")]
    assert metrics["phase"] == "dpo" and metrics["train_loss_final"] == 0.55
    saved = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert "diverged" not in saved


def test_dpo_aborts_on_non_finite_loss_before_saving(tmp_path, monkeypatch):
    import json

    from adapterforge.train.dpo import run_dpo

    record = _install_fake_dpo_stack(monkeypatch, losses=[0.69, float("nan"), 0.5])
    with pytest.raises(SystemExit) as excinfo:
        run_dpo(_dpo_config(tmp_path))

    assert excinfo.value.code == 1
    assert record["saved"] == []  # ningún adapter divergido llega a guardarse
    saved = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["diverged"] is True and saved["diverged_step"] == 2


def test_dpo_aborts_when_render_is_contaminated(tmp_path, monkeypatch):
    import adapterforge.train.dpo as dpo

    record = _install_fake_dpo_stack(monkeypatch, losses=[0.69])
    # Un template oficial «roto» (con <think>) debe parar el run antes de entrenar.
    monkeypatch.setattr(dpo, "load_official_chat_template", lambda: _THINKING_TEMPLATE)
    with pytest.raises(SystemExit, match=r"\[dpo\] ABORTADO"):
        dpo.run_dpo(_dpo_config(tmp_path))
    assert "template_seen" not in record
