# ADR 0001 — QLoRA (+ Unsloth opcional) frente a full fine-tuning

- **Estado**: aceptada
- **Fecha**: 2026-08-01
- **Decisores**: Jairo Sarabia

## Contexto

El hardware objetivo es una GPU de consumo con **8 GB de VRAM** (más 64 GB de
RAM). Un full fine-tuning de un modelo de 3-4B en FP16/BF16 exige, solo en
pesos + optimizador Adam + gradientes, del orden de 16x el número de
parámetros en bytes (~48-64 GB para 4B): imposible en este equipo. Además el
objetivo del proyecto es producir **adapters** intercambiables (multi-tenant
LoRA en vLLM), no modelos monolíticos.

## Decisión

1. **QLoRA como técnica base**: modelo congelado cuantizado a **4-bit NF4 con
   double quantization** (bitsandbytes) + adapters **LoRA r=16, alpha=32,
   dropout=0.05 sobre todas las proyecciones lineales**, optimizador
   `paged_adamw_8bit` y gradient checkpointing. Presupuesto: ~3.7-5 GB para un
   4B con seq_len 1024 y batch 1 + grad_accum 16.
2. **Unsloth como acelerador opcional (try-import)**: si está instalado, sus
   kernels fusionados reducen VRAM y aceleran ~1.5-2x, y son los que hacen
   viable el *stretch goal* de 7-8B en 8 GB (seq corta). El código funciona
   igual sin él (`src/adapterforge/train/runtime.py`), porque la instalación
   de Unsloth está acoplada a versiones exactas de torch/CUDA y no queremos
   que el pipeline dependa de ella.
3. **Validación previa al hardware**: `ExperimentConfig` rechaza en duro
   combinaciones `seq_len*batch` inviables y emite avisos suaves con una
   estimación heurística de VRAM, con matriz de fallback OOM documentada en el
   README.

## Alternativas consideradas

- **Full fine-tuning**: descartado por VRAM (ver contexto) y porque destruye
  la historia multi-adapter del serving.
- **LoRA sobre base FP16 (sin cuantizar)**: un 4B en FP16 son ~8 GB solo de
  pesos; no deja sitio a activaciones. Descartado.
- **Solo Unsloth (obligatorio)**: acopla el repo a una dependencia de
  evolución rápida; preferimos TRL/PEFT estándar con Unsloth como mejora.

## Consecuencias

- (+) Entrenamiento reproducible en hardware de consumo; adapter de ~50-100 MB
  publicable y combinable.
- (+) La misma config YAML sirve con y sin Unsloth (flag en métricas).
- (−) La cuantización NF4 introduce una pequeña pérdida de calidad frente a
  FP16; se mide en el benchmark de dominio (before/after) en lugar de asumirse.
- (−) Dos caminos de carga de modelo que mantener (unsloth/transformers).
