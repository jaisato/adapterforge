"""Serving y publicación del adapter.

Módulos Python (merge, push a HF Hub, render de Modelfile) con imports
pesados perezosos. Los assets no-Python (export_gguf.sh, Modelfile.j2,
snippets de vLLM multi-LoRA, plantilla de model card) viven en ``serve/``
en la raíz del repositorio.
"""
