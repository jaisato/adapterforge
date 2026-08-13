"""AdapterForge: fábrica reproducible de adapters LoRA/QLoRA en GPU de 8 GB.

El paquete se divide en cuatro subpaquetes:

- ``data``: pipeline de dataset instruccional (extracción, síntesis, filtrado,
  formateo chat + splits).
- ``train``: SFT y DPO con TRL/PEFT sobre base 4-bit (imports pesados lazy).
- ``eval``: benchmark de dominio propio e informes before/after.
- ``serve``: exportación (GGUF/Ollama/vLLM-LoRA) y publicación en HF Hub.

Regla de oro: importar ``adapterforge`` o cualquier módulo del núcleo NUNCA
debe arrastrar torch/transformers/mlflow; esas dependencias se importan de
forma perezosa dentro de las funciones que las usan.
"""

__version__ = "0.1.0"
