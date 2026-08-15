#!/usr/bin/env bash
# =============================================================================
# Exportación del adapter a GGUF Q4_K_M usando el contenedor oficial de
# llama.cpp (ghcr.io/ggml-org/llama.cpp). Ruta "edge/CPU": el resultado corre
# en Ollama o llama.cpp sin GPU (VPS, Space...).
#
# Pasos que ejecuta este script (documentados uno a uno):
#   1) MERGE   : fusiona base + adapter LoRA en un modelo FP16/BF16 (en CPU).
#   2) CONVERT : HF safetensors -> GGUF F16 con convert_hf_to_gguf.py.
#   3) QUANT   : GGUF F16 -> Q4_K_M con llama-quantize.
#   4) MODELFILE: renderiza el Modelfile de Ollama listo para `ollama create`.
#
# Uso:
#   bash serve/export_gguf.sh [BASE_MODEL] [ADAPTER_DIR] [OUTPUT_DIR]
# Defaults pensados para el experimento principal (qlora_qwen3-4b).
# =============================================================================
set -euo pipefail

BASE_MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
ADAPTER_DIR="${2:-artifacts/qlora_qwen3-4b/adapter}"
OUTPUT_DIR="${3:-artifacts/gguf}"
MODEL_NAME="juristune"
LLAMACPP_IMAGE="ghcr.io/ggml-org/llama.cpp:full"

MERGED_DIR="${OUTPUT_DIR}/merged"
GGUF_F16="${OUTPUT_DIR}/${MODEL_NAME}-f16.gguf"
GGUF_Q4="${OUTPUT_DIR}/${MODEL_NAME}-q4_k_m.gguf"

mkdir -p "${OUTPUT_DIR}"

echo "==> [1/4] Merge del adapter con la base (CPU, usa RAM del host)"
python -m adapterforge.serve.merge_adapter \
    --base "${BASE_MODEL}" \
    --adapter "${ADAPTER_DIR}" \
    --output "${MERGED_DIR}"

echo "==> [2/4] Conversión HF -> GGUF F16 (contenedor llama.cpp)"
# El contenedor 'full' incluye los scripts de conversión de llama.cpp.
docker run --rm -v "$(pwd):/workdir" "${LLAMACPP_IMAGE}" \
    --convert "/workdir/${MERGED_DIR}" \
    --outfile "/workdir/${GGUF_F16}" \
    --outtype f16

echo "==> [3/4] Cuantización Q4_K_M (~4 bits, la mejor relación calidad/tamaño)"
docker run --rm -v "$(pwd):/workdir" "${LLAMACPP_IMAGE}" \
    --quantize "/workdir/${GGUF_F16}" "/workdir/${GGUF_Q4}" "Q4_K_M"

echo "==> [4/4] Modelfile de Ollama"
python -m adapterforge.serve.render_modelfile \
    --gguf "${GGUF_Q4}" \
    --output "${OUTPUT_DIR}/Modelfile"

echo ""
echo "Exportación completada:"
echo "  GGUF F16   : ${GGUF_F16} (puede borrarse tras cuantizar)"
echo "  GGUF Q4_K_M: ${GGUF_Q4}"
echo "  Modelfile  : ${OUTPUT_DIR}/Modelfile"
echo ""
echo "Siguientes pasos:"
echo "  ollama create ${MODEL_NAME} -f ${OUTPUT_DIR}/Modelfile"
echo "  ollama run ${MODEL_NAME} 'Resume el Real Decreto 214/2026'"
echo ""
echo "TODO(jairo): cuantización AWQ para la ruta GPU (vLLM) como alternativa."
