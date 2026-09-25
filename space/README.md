---
title: JurisTune Demo (AdapterForge)
emoji: "⚖️"
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: "6.15.1"
app_file: app.py
pinned: false
license: mit
short_description: Adapter QLoRA jurídico-administrativo servido en CPU (GGUF)
---

# JurisTune — demo pública (CPU)

Demo del adapter **JurisTune** entrenado con
[AdapterForge](https://github.com/jairosarabia/adapterforge): resúmenes
estructurados, Q&A con cita y extracción de obligaciones/plazos sobre
normativa española.

- Modelo: GGUF Q4_K_M (merge base+adapter) descargado de
  [Jaisato en Hugging Face](https://huggingface.co/Jaisato) y servido con
  `llama-cpp-python` en el tier CPU gratuito de Spaces.
- Variables del Space: `AF_GGUF_REPO` y `AF_GGUF_FILE` permiten apuntar a otro
  GGUF sin tocar código.
- La primera respuesta tarda más (carga del modelo); después responde en
  segundos con contextos cortos.
