# Informe de evaluación — before/after

- **Before (base)**: `qwen3-base-2507-eval`
- **After (adapter)**: `juristune-v6-eval`
- **Casos**: 111
- **Generación**: temp 0.0 · 3 repetición(es)/caso · modelos intercalados en una sesión · 2026-08-29T09:21:09+00:00

## Resumen por tipo de tarea

| Tipo | Before | After | Delta |
|---|---:|---:|---:|
| extract | 1.000 | 1.000 | +0.000 |
| qa | 0.827 | 0.922 | +0.096 |
| resumen | 0.628 | 0.726 | +0.098 |
| **overall** | **0.840** | **0.898** | **+0.058** |

## Incertidumbre (deltas pareados por caso)

- Delta medio: +0.058 · SD 0.243 · SE 0.023 (n=111)
- IC 95 %: (+0.013, +0.104)
- El IC excluye 0: el adapter mejora de forma significativa.

## Regresiones por caso

| Caso | Before | After | ¿Robusta? |
|---|---:|---:|---|
| eb-019-resumen-impulso | 0.646 | 0.533 | sí |
| eb-022-resumen-bono | 0.548 | 0.458 | sí |
| eb-023-resumen-ciber | 0.459 | 0.449 | sí |
| eb-025-resumen-menores | 0.421 | 0.302 | sí |
| eb-137-apoderamientos-registro | 1.000 | 0.286 | sí |
| eb-150-anticipo-garantia | 0.850 | 0.524 | sí |
| eb-161-informe-calidad | 1.000 | 0.121 | sí |
| eb-166-resumen-barreras-comercio | 0.741 | 0.620 | sí |
| eb-168-resumen-registro-voluntariado | 0.562 | 0.427 | sí |
| eb-173-resumen-alumbrado | 0.719 | 0.447 | sí |
| eb-185-resumen-salud-mental | 0.599 | 0.492 | sí |
| eb-186-resumen-referente-bienestar | 0.690 | 0.442 | sí |

> Generado con `python -m adapterforge.eval.report`. Métricas: exact-match por campos (extract), containment/ROUGE-L (qa) y ROUGE-L (resumen), con normalización numérica (separadores de miles y numerales en castellano) y limpieza de artefactos `<think>`/`tool_call` antes de puntuar.
