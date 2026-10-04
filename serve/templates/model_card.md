---
language:
  - es
license: mit
base_model: {{ base_model }}
tags:
  - lora
  - qlora
  - peft
  - trl
  - legal
  - boe
  - spanish
pipeline_tag: text-generation
---

# JurisTune — adapter {{ experiment }}

Adapter LoRA que especializa `{{ base_model }}` en **redacción y comprensión
jurídico-administrativa española**: resúmenes estructurados de disposiciones,
Q&A con cita verificable, extracción de obligaciones/plazos/sanciones en JSON
y reescritura en lenguaje llano.

Entrenado con [AdapterForge](https://github.com/jaisato/adapterforge)
(pipeline reproducible: dataset sintético anclado al BOE -> QLoRA -> DPO ->
evaluación before/after) en una GPU de consumo de 8 GB.

## Métricas

| Métrica | Valor |
|---|---|
| Fase | {{ phase }} |
| Pasos de entrenamiento | {{ steps }} |
| Loss inicial -> final | {{ train_loss_initial }} -> {{ train_loss_final }} |
| Benchmark de dominio (overall) | {{ bench_overall }} |
| — extract (EM por campos) | {{ bench_extract }} |
| — qa (containment/ROUGE-L) | {{ bench_qa }} |
| — resumen (ROUGE-L) | {{ bench_resumen }} |

Configuración: LoRA r={{ lora_r }}, seq_len={{ seq_len }}, base 4-bit NF4,
unsloth={{ used_unsloth }}. Generado el {{ timestamp }}.

## Uso

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{{ base_model }}")
model = PeftModel.from_pretrained(base, "Jaisato/juristune-qwen3-4b-lora")
tokenizer = AutoTokenizer.from_pretrained("Jaisato/juristune-qwen3-4b-lora")
```

Con vLLM (multi-LoRA): `--enable-lora --lora-modules juristune=<ruta>` y
`"model": "juristune"` en cada petición.

## Datos y licencia

Dataset sintético generado por destilación sobre textos normativos públicos
(el BOE es reutilizable con cita de la fuente). Los ejemplos están anclados al
documento para minimizar la alucinación del teacher. El corpus y los splits
`juristune-v*` están en el dataset privado
[Jaisato/juristune-datasets](https://huggingface.co/datasets/Jaisato/juristune-datasets)
(`make data-pull` en el repositorio). Licencia del adapter: MIT.

## Límites de uso

- No es asesoramiento jurídico; verifica siempre contra el texto oficial.
- Entrenado sobre normativa española: fuera de ese dominio rinde como la base.
- La cita [BOE-A-XXXX-YYYY, art. N] debe validarse contra la fuente enlazada.
