"""Callbacks de entrenamiento (testables sin GPU)."""

from __future__ import annotations

import math

from transformers import TrainerCallback


class AbortOnNonFiniteLoss(TrainerCallback):
    """Detiene el entrenamiento en la PRIMERA loss no finita.

    Lección de v2: divergió a NaN en el paso 341 y aun así el adapter se
    guardó, mergeó, cuantizó y evaluó, porque el guard corría después de
    ``save_model``. Este callback corta en cuanto aparece una loss (de train
    o de eval) no finita y deja el run marcado como divergido ANTES de que
    nada se guarde.

    Necesita ver la loss cruda: con ``logging_nan_inf_filter=True`` (el valor
    por defecto de TrainingArguments, SFTConfig y DPOConfig) el bucle de
    Trainer cambia una loss NaN/inf por la media de la ventana antes de
    ``on_log``, y el callback no la veía nunca. Por eso lo apaga al empezar.
    """

    def __init__(self, label: str = "sft") -> None:
        self.label = label  # prefijo del log: sft o dpo
        self.diverged = False
        self.bad_step: int | None = None

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        # El bucle lee este mismo objeto args en cada paso.
        args.logging_nan_inf_filter = False
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return control
        for key in ("loss", "eval_loss"):
            value = logs.get(key)
            if value is not None and not math.isfinite(float(value)):
                self.diverged = True
                self.bad_step = int(state.global_step)
                control.should_training_stop = True
                control.should_save = False
                print(
                    f"[{self.label}] ABORTADO: {key} no finita ({value}) en el paso "
                    f"{state.global_step}; el run queda marcado como divergido"
                )
        return control
