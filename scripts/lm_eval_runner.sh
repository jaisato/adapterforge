#!/usr/bin/env bash
# =============================================================================
# Evaluación de capacidades generales con lm-evaluation-harness (grupo "eval").
#
# Objetivo: comprobar que el fine-tuning NO degrada capacidades generales en
# español (olvido catastrófico). Se ejecuta sobre la base, el adapter SFT y el
# adapter SFT+DPO, y la tabla comparada va al README.
#
# Instalación (opcional, pesada — por eso vive en un grupo aparte):
#   pip install -e ".[eval]"        # instala lm-eval
#
# Requiere el stack train instalado (transformers/peft) para cargar el adapter.
# =============================================================================
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
ADAPTER_DIR="${ADAPTER_DIR:-artifacts/qlora_qwen3-4b/adapter}"
# Tareas en español razonables para un 4B; ajusta según la versión de lm-eval:
#   lm-eval --tasks list | grep -i -E "belebele|mmlu"
TASKS="${TASKS:-belebele_spa_Latn}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/lm_eval}"

mkdir -p "${OUTPUT_DIR}"

echo "==> [1/2] Base: ${BASE_MODEL}"
lm_eval --model hf \
    --model_args "pretrained=${BASE_MODEL},load_in_4bit=True" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --output_path "${OUTPUT_DIR}/base"

echo "==> [2/2] Base + adapter: ${ADAPTER_DIR}"
lm_eval --model hf \
    --model_args "pretrained=${BASE_MODEL},peft=${ADAPTER_DIR},load_in_4bit=True" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --output_path "${OUTPUT_DIR}/adapter"

echo ""
echo "Resultados en ${OUTPUT_DIR}/{base,adapter}. Criterio de aceptación del"
echo "portfolio: degradación < 2 puntos frente a la base."
echo ""
echo "TODO(jairo): integrar lm-eval-harness completo (suite MMLU-es + gate CI)."
