"""Entrenamiento SFT/DPO con TRL + PEFT sobre base 4-bit (NF4).

Los imports pesados (torch, transformers, trl, peft, datasets, unsloth,
mlflow) son SIEMPRE perezosos: importar este subpaquete no debe costar nada.
"""
