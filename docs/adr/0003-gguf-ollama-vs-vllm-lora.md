# ADR 0003 — Serving dual: GGUF/Ollama (edge/CPU) y vLLM multi-LoRA (GPU)

- **Estado**: aceptada
- **Fecha**: 2026-08-01
- **Decisores**: Jairo Sarabia

## Contexto

El adapter debe servirse en tres escenarios con restricciones distintas:

| Escenario | Hardware | Necesidad |
|---|---|---|
| Demo pública (HF Space) y VPS | CPU, poca RAM | Un binario autocontenido y barato |
| Estación local / producción GPU | GPU 8 GB | Varios "modelos" especializados a la vez |
| Distribución | — | Artefactos estándar en HF Hub |

Un único formato no cubre los tres: GGUF no aprovecha multi-adapter en GPU;
vLLM no corre en el tier CPU gratuito de Spaces.

## Decisión

**Mantener las dos rutas como artefactos de release, cada una con su caso de
uso, en lugar de elegir una:**

1. **Ruta edge/CPU — GGUF Q4_K_M + Ollama** (`serve/export_gguf.sh`):
   merge del adapter -> conversión GGUF -> cuantización Q4_K_M con el
   contenedor `ghcr.io/ggml-org/llama.cpp` -> `Modelfile` renderizado
   (`serve/ollama/Modelfile.j2`). Es lo que corre en el VPS sin GPU y en el
   Space (llama-cpp-python). Coste: el merge congela el adapter (un GGUF por
   variante) y Q4_K_M pierde algo de calidad, medible en el benchmark.
2. **Ruta GPU — vLLM `--enable-lora`** (perfil `serve-vllm` de compose +
   `serve/vllm_lora/`): una única base residente y N adapters seleccionables
   por petición vía el campo `model` de la API OpenAI-compatible. Es el patrón
   multi-tenant real de producción y el que usa la evaluación before/after
   (base y adapter en el mismo endpoint, mismas condiciones).

Criterio de elección documentado: **¿hay GPU y se necesitan variantes o
comparativas? -> vLLM-LoRA. ¿CPU, edge o distribución sencilla? -> GGUF.**

## Alternativas consideradas

- **Solo GGUF**: simple, pero pierde el serving multi-adapter (gran señal de
  senioridad) y duplica artefactos por cada variante del adapter.
- **Solo vLLM-LoRA**: inviable en VPS/Space sin GPU.
- **TGI**: soporta LoRA pero con menos madurez multi-adapter que vLLM en el
  momento de la decisión; una sola pila GPU es más mantenible.
- **AWQ/GPTQ de la base fusionada**: interesante para vLLM en 8 GB;
  documentado como extensión (`TODO(jairo)` en `serve/export_gguf.sh`).

## Consecuencias

- (+) Cada release publica adapter (safetensors) + GGUF, y cualquier revisor
  puede probarlo con `ollama create` o con `docker compose --profile
  serve-vllm up`.
- (−) Dos rutas que mantener y testear; se mitiga con scripts documentados y
  el mismo benchmark de dominio para ambas.
