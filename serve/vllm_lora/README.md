# vLLM multi-LoRA: una base residente, N adapters

Patrón de producción para GPU: **una única copia del modelo base en VRAM** y
múltiples adapters LoRA que se aplican por petición (multi-tenant LoRA). En una
GPU de 8 GB es la única forma sensata de servir varios "modelos" especializados
a la vez.

## Arranque

El perfil `serve-vllm` de `compose.yaml` ya lo deja listo (monta el adapter
final, `artifacts/qlora_qwen3-4b-v6/adapter`; otro run con la variable
`ADAPTER_DIR`, ruta absoluta o relativa a la raíz del repo):

```bash
docker compose --profile serve-vllm up vllm
ADAPTER_DIR=artifacts/qlora_qwen3-4b/adapter docker compose --profile serve-vllm up vllm
ADAPTER_DIR=/ruta/absoluta/al/adapter docker compose --profile serve-vllm up vllm
```

Equivalente manual (para ver todos los flags):

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v ./artifacts/qlora_qwen3-4b-v6/adapter:/adapters/juristune:ro \
  vllm/vllm-openai:v0.8.5 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules juristune=/adapters/juristune
```

Claves:

- `--enable-lora` activa el runtime multi-adapter.
- `--lora-modules nombre=/ruta` registra cada adapter con un nombre de modelo
  propio; puede repetirse N veces (ver `compose.multi-adapter.yaml`).
- `--max-lora-rank` debe ser >= al `r` del adapter (aquí r=16, margen a 32).
- En 8 GB: base 4B + `--max-model-len 4096` + `gpu-memory-utilization 0.90`
  deja hueco para los adapters y el KV cache. Si hay OOM: baja `max-model-len`.

## Consumo (API OpenAI-compatible)

El campo `model` selecciona base o adapter **por petición**:

```bash
# Modelo base (control / comparativa before-after)
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "messages": [{"role": "user", "content": "Resume el Real Decreto 214/2026"}]
}'

# Adapter JurisTune
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "juristune",
  "messages": [{"role": "user", "content": "Resume el Real Decreto 214/2026"}]
}'
```

Esto es exactamente lo que usa la evaluación before/after:

```bash
python -m adapterforge.eval.domain_bench --api-base http://localhost:8000/v1 \
  --model Qwen/Qwen3-4B-Instruct-2507 --output artifacts/eval/base.json
python -m adapterforge.eval.domain_bench --api-base http://localhost:8000/v1 \
  --model juristune --output artifacts/eval/adapter.json
python -m adapterforge.eval.report --before artifacts/eval/base.json \
  --after artifacts/eval/adapter.json --output artifacts/eval/report.md
```

## Varios adapters a la vez

`compose.multi-adapter.yaml` (snippet de este directorio) muestra cómo montar
y registrar varios adapters (p. ej. `juristune` SFT y `juristune-dpo`):

```bash
docker compose -f compose.yaml -f serve/vllm_lora/compose.multi-adapter.yaml \
  --profile serve-vllm up vllm
```

TODO(jairo): carga dinámica de adapters en caliente (endpoint
`/v1/load_lora_adapter` de vLLM) y autenticación por API key delante (InferaGate).
