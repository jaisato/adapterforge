"""Tests de las salvaguardas de entrenamiento (template, NaN-abort, config v3)."""

from __future__ import annotations

import pytest

from adapterforge.config import load_experiment_config
from adapterforge.train.callbacks import AbortOnNonFiniteLoss
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
    callback = AbortOnNonFiniteLoss()
    control = _Control()
    callback.on_log(None, _State(), control, logs={"loss": 1.5})
    assert not callback.diverged
    callback.on_log(None, _State(), control, logs={"loss": float("nan")})
    assert callback.diverged
    assert callback.bad_step == 341
    assert control.should_training_stop is True
    assert control.should_save is False


def test_git_sha_available_in_repo():
    sha = git_sha()
    assert sha is None or (isinstance(sha, str) and len(sha) >= 7)


def test_v3_config_loads_with_validation_split():
    cfg = load_experiment_config("configs/qlora_qwen3-4b_v3.yaml")
    assert cfg.val_dataset_path is not None
    assert "juristune-v3_val" in str(cfg.val_dataset_path)
    assert cfg.lora.dropout == 0.05
    assert cfg.seq_len == 2048
