# ADR 0002 — DPO como fase de alineamiento (ORPO como extensión)

- **Estado**: aceptada
- **Fecha**: 2026-08-01
- **Decisores**: Jairo Sarabia

## Contexto

Tras el SFT, el adapter debe mejorar **formato y estilo** (estructura fija de
resúmenes, JSON válido en extracción, cita obligatoria) usando pares de
preferencia chosen/rejected. Candidatos: **DPO** (Direct Preference
Optimization), **ORPO** (Odds Ratio Preference Optimization) y RLHF clásico
(PPO).

Restricción dura: 8 GB de VRAM. El DPO "de libro" necesita un modelo de
referencia congelado además del entrenado (2x memoria de pesos), pero **con
PEFT, TRL computa la referencia desactivando el adapter** sobre la misma base
cuantizada: una sola copia del modelo en memoria.

## Decisión

**DPO con TRL (`DPOTrainer`) sobre el adapter SFT**, con `beta=0.1`,
`lr=5e-6` y los pares en formato conversacional
(`prompt`/`chosen`/`rejected`, ver `data/processed/sample_dpo.jsonl`).

Razones frente a ORPO:

1. **Madurez y trazabilidad**: DPO es el método más asentado en TRL, con
   hiperparámetros bien documentados y métricas (margins, rewards) fáciles de
   interpretar en el informe before/after.
2. **El argumento de memoria de ORPO se diluye con PEFT**: la ventaja clásica
   de ORPO (sin modelo de referencia) importa menos cuando la referencia es
   "la misma base con el adapter desactivado" (coste extra ≈ una pasada
   forward, no una copia de pesos).
3. **Pipeline en dos fases (SFT -> DPO)** = dos artefactos comparables en la
   evaluación (base vs SFT vs SFT+DPO), que es exactamente la tabla que el
   portfolio quiere enseñar.

## Alternativas consideradas

- **ORPO**: integra SFT+preferencias en una sola fase y ahorra la pasada de
  referencia. Queda como extensión documentada — `TODO(jairo)` en
  `train/dpo.py` — para comparar coste/calidad con el mismo dataset.
- **PPO/RLHF completo**: requiere modelo de recompensa y mucho más cómputo;
  desproporcionado para un adapter de dominio.

## Consecuencias

- (+) Cabe en 8 GB (una base 4-bit + adapter, sin referencia separada).
- (+) Mejora medible de formato/estilo con ~15-4.000 pares; el repo incluye 15
  pares de muestra generados offline para el smoke.
- (−) DPO exige pares de calidad; el modo offline degrada respuestas buenas
  (vaguedad, copia, desestructura) como aproximación honesta hasta tener
  preferencias reales del juez/humanas.
