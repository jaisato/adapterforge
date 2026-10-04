# AdapterForge

> **Fábrica reproducible de adapters LoRA/QLoRA en una GPU de consumo de 8 GB.**
> Dataset instruccional sintético anclado a normativa española → SFT QLoRA (TRL/PEFT,
> Unsloth opcional) → alineamiento DPO → evaluación before/after con benchmark propio →
> serving en tres formatos (vLLM multi-LoRA, GGUF/Ollama, HF Hub). Todo dockerizado
> (WSL2 + CUDA), con tracking MLflow y publicación automática por GitHub Actions.

Proyecto 2 del portfolio de IA aplicada de [jairosarabia](https://github.com/jairosarabia)
· Modelos y datasets en [huggingface.co/Jaisato](https://huggingface.co/Jaisato)

---

## ¿Qué es?

AdapterForge implementa el ciclo completo que las ofertas de AI Engineer llaman
"fine-tuning eficiente" y que casi nadie enseña de punta a punta: **construir el
dataset, entrenar el adapter, evaluarlo con honestidad y servirlo de forma
operable**, en hardware de consumo y de forma reproducible por cualquier revisor.

El caso de uso es **JurisTune**: un adapter que especializa un modelo base en
redacción y comprensión jurídico-administrativa española (resúmenes estructurados,
Q&A con cita verificable `[BOE-A-2026-XXXX, art. N]`, extracción de
obligaciones/plazos/sanciones en JSON y lenguaje llano).

```mermaid
flowchart LR
  subgraph Datos
    RAW[(Corpus JSONL estilo BOE<br/>data/raw)] --> EX[extract.py<br/>validación + estructura]
    EX --> SYN[synthesize/<br/>teacher litellm u --offline]
    SYN --> FIL[filter.py<br/>dedup shingles+Jaccard<br/>heurísticas · LLM-judge opcional]
    FIL --> FMT[format.py<br/>chat messages · splits 90/5/5]
  end
  subgraph Entrenamiento GPU 8GB
    FMT --> SFT[sft.py · TRL SFTTrainer<br/>QLoRA NF4 · Unsloth opcional]
    FMT --> DPO[dpo.py · TRL DPOTrainer<br/>pares chosen/rejected]
    SFT --> DPO
    SFT & DPO --> MLF[(MLflow :5000)]
  end
  subgraph Evaluación
    SFT & DPO --> DB[domain_bench.py<br/>EM por campos · ROUGE-L propio]
    DB --> REP[report.py<br/>informe before/after]
  end
  subgraph Serving
    SFT & DPO --> VLLM[vLLM --enable-lora<br/>multi-adapter GPU]
    SFT & DPO --> GGUF[export_gguf.sh<br/>Q4_K_M + Ollama Modelfile]
    SFT & DPO --> HUB[HF Hub Jaisato<br/>adapter + model card]
    HUB --> SPACE[HF Space demo<br/>llama-cpp-python CPU]
  end
```

## Arquitectura y decisiones

- **Núcleo ligero, entrenamiento pesado aislado**: el paquete `adapterforge`
  (datos, configs, métricas) solo depende de pydantic/yaml/httpx/litellm; torch,
  transformers, trl, peft, datasets, mlflow y unsloth viven en grupos opcionales
  y se importan de forma perezosa. Puedes construir y filtrar datasets en
  cualquier máquina sin GPU.
- **Configuración como contrato**: cada experimento es un YAML validado con
  pydantic (`ExperimentConfig`). Las combinaciones inviables en 8 GB
  (`seq_len*batch` excesivo) se rechazan antes de importar torch; el resto de
  riesgos emite avisos suaves con estimación de VRAM.
- **Modo offline de la síntesis**: generadores determinísticos producen el
  dataset de muestra (40 ejemplos SFT + 15 pares DPO incluidos en el repo) sin
  API ni red — el smoke-train y los tests no dependen de ningún proveedor.
  La síntesis es el paquete `src/adapterforge/data/synthesize/`, un módulo
  por responsabilidad: `examples` (modelos `SftExample`/`DpoPair`, mezcla y
  presupuesto), `text` (redacción en castellano), `detectors` (plazos,
  obligaciones, sanciones y tramos validados), `window` (ventana estructural
  dentro del presupuesto de tokens), `prompts` (pools de instrucciones y
  plantillas del teacher), `offline` (generadores por tarea y `rebalance`),
  `dpo` (pares chosen/rejected), `online` (teacher vía litellm) y `cli`.
  El `__init__` reexporta la API pública, así que
  `from adapterforge.data.synthesize import build_sft_offline` y
  `python -m adapterforge.data.synthesize` siguen funcionando igual.
- **Evaluación con dos niveles**: métricas deterministas propias (exact-match
  por campos, ROUGE-L) para el gate de CI, y juez LLM opcional para rúbricas.
- **Serving dual**: vLLM `--enable-lora` (GPU, multi-adapter) y GGUF/Ollama
  (CPU/edge). Ver decisiones completas en `docs/adr/`:
  - [ADR 0001 — QLoRA + Unsloth opcional vs full fine-tuning](docs/adr/0001-qlora-unsloth-vs-full-finetune.md)
  - [ADR 0002 — DPO vs ORPO](docs/adr/0002-dpo-vs-orpo.md)
  - [ADR 0003 — GGUF/Ollama vs vLLM-LoRA para servir](docs/adr/0003-gguf-ollama-vs-vllm-lora.md)

## Requisitos

- **Windows 11 + WSL2 (Ubuntu 24.04)** con `wsl --update` reciente.
- **Driver NVIDIA de Windows** (nunca instalar un driver Linux dentro de WSL:
  el driver de Windows ya expone CUDA a WSL2).
- **Docker Engine dentro de WSL2** + `nvidia-container-toolkit`. Verificación:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

- **Python 3.12** y `make` para el flujo local sin contenedor (datos/eval/tests).
- **GPU NVIDIA de 8 GB de VRAM** (el entrenamiento usa NF4 + gradient
  checkpointing). Recomendado: `.wslconfig` con `memory=48GB`, `swap=16GB`, y el
  repositorio clonado en el filesystem de Linux (`~/proyectos/...`, no `/mnt/c/`).

### Presupuesto de VRAM y matriz de fallback OOM

| Situación | Config recomendada | VRAM aprox. |
|---|---|---|
| Smoke/CI | `SmolLM2-360M`, seq 512, batch 2 | ~2 GB |
| **Cómodo (por defecto)** | `Qwen3-4B`, seq 1024, batch 1, accum 16 | ~3.7-5 GB |
| Alternativa | `Llama-3.2-3B`, seq 1024, batch 1 | ~3-4.5 GB |
| Stretch (con Unsloth) | `Qwen3-8B`/`Llama-3.1-8B`, seq 1024, batch 1, accum 16 | ~6-7.5 GB |

**Si aparece OOM, aplica en este orden** (cada paso es un cambio en el YAML):

1. Reduce `seq_len` (1024 → 768 → 512): las activaciones escalan lineal.
2. Asegura `batch_size: 1` y sube `grad_accum` para mantener el batch efectivo.
3. Instala **Unsloth** en la imagen de entrenamiento (descomenta la línea en
   `docker/Dockerfile.train`): ahorra 1-2 GB y acelera.
4. Baja a un modelo menor (8B → 4B → 3B) o reduce `lora.r` (16 → 8).
5. Último recurso: cierra procesos que ocupen VRAM (Ollama del host, navegador).

La validación de `ExperimentConfig` te avisará antes de tocar la GPU si la
combinación pinta mal (`vram_warnings`), y rechazará en duro lo imposible.

## Puesta en marcha local paso a paso

```bash
# 1) Clonar DENTRO del filesystem de WSL2
git clone https://github.com/jairosarabia/adapterforge.git ~/proyectos/adapterforge
cd ~/proyectos/adapterforge

# 2) Entorno del núcleo (datos/eval/tests; sin GPU)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # rellena tokens solo si vas a usar teacher/HF

# 3) Verificar calidad
make lint && make test

# 4) Generar el dataset de muestra (modo offline, sin API)
make data                      # escribe data/processed/{sample_sft,sample_dpo}.jsonl y splits

# 5) Construir la imagen de entrenamiento y lanzar el smoke (GPU)
docker compose --profile train build trainer
make train-smoke               # 30 pasos; métricas en artifacts/smoke/metrics.json
python scripts/check_smoke.py artifacts/smoke/metrics.json

# 6) Entrenamiento real + tracking
make mlflow                    # UI en http://localhost:5000
make train                     # configs/qlora_qwen3-4b.yaml
make dpo                       # fase DPO sobre los pares de preferencia
```

> Síntesis con teacher real (opcional): sustituye el paso offline por
> `python -m adapterforge.data.synthesize --input data/interim/documents.jsonl \
> --output data/interim/sft_raw.jsonl --model openai/gpt-4o-mini` (o
> `--backend openai --api-base http://host.docker.internal:11434/v1 --model qwen3:8b`
> para un teacher 100 % local vía Ollama).

## Cómo probarlo

```bash
# Tests unitarios (sin GPU ni red) y lint
pytest && ruff check .

# Pipeline de datos end-to-end en local
make data && head -1 data/processed/sample_sft.jsonl | python -m json.tool

# Benchmark de dominio contra el Ollama del host (u otro endpoint OpenAI-compatible)
python -m adapterforge.eval.domain_bench \
  --api-base http://localhost:11434/v1 --model qwen3:4b \
  --output artifacts/eval/base.json

# Servir base + adapter con vLLM y comparar before/after
docker compose --profile serve-vllm up -d vllm
python -m adapterforge.eval.domain_bench --api-base http://localhost:8000/v1 \
  --model Qwen/Qwen3-4B-Instruct-2507 --output artifacts/eval/base.json
python -m adapterforge.eval.domain_bench --api-base http://localhost:8000/v1 \
  --model juristune --output artifacts/eval/adapter.json
make eval-report && cat artifacts/eval/report.md
```

## Configuración

Dos niveles, ambos validados con pydantic:

1. **Proceso** (`.env`, prefijo `AF_`, ver `.env.example` y
   `src/adapterforge/settings.py`): teacher/juez (`AF_TEACHER_MODEL`,
   `AF_JUDGE_MODEL`), endpoint OpenAI-compatible (`AF_API_BASE`, `AF_API_KEY`),
   HF (`HF_TOKEN`, `AF_HF_REPO_ID`), MLflow (`AF_MLFLOW_TRACKING_URI`).
   Desde contenedores, el Ollama del host es
   `http://host.docker.internal:11434` (los servicios declaran
   `extra_hosts: host-gateway`).
2. **Experimento** (`configs/*.yaml`, modelo `ExperimentConfig`): base_model,
   lora (r/alpha/dropout/target_modules), seq_len, lr, epochs,
   batch/grad_accum, seed, output_dir, `max_steps` opcional (smoke) y sección
   `dpo` (beta, max_prompt_len). Los YAML incluidos:

   | Config | Uso |
   |---|---|
   | `qlora_smoke.yaml` | Smoke de CI (SmolLM2-360M, 30 pasos) |
   | `qlora_qwen3-4b.yaml` | Experimento principal JurisTune |
   | `qlora_llama-3.2-3b.yaml` | Alternativa (modelo gated de Meta) |
   | `dpo_qwen3-4b.yaml` | Fase DPO sobre el adapter SFT |

## Evaluación

- **Benchmark propio** (`evals/domain_bench.jsonl`, 25 casos con referencia):
  `extract` (exact-match por campos sobre el JSON de la respuesta), `qa`
  (containment + ROUGE-L) y `resumen` (ROUGE-L propio, sin dependencias).
  Ejecutable contra cualquier API OpenAI-compatible o contra un fichero de
  predicciones (`--predictions`), con `--min-score` como quality gate.
- **Informe before/after** (`eval/report.py`): tabla por tipo de tarea con
  deltas y regresiones caso a caso; es la tabla que acompaña cada release.
- **Capacidades generales** (`scripts/lm_eval_runner.sh`, grupo opcional
  `eval`): lm-evaluation-harness sobre base vs adapter para vigilar el olvido
  catastrófico (criterio: degradación < 2 puntos).
- **Juez LLM opcional** en el filtrado del dataset (`filter.py --judge`) con
  rúbrica 1-5 vía litellm (fidelidad/formato/utilidad).

## Despliegue / Publicación

| Destino | Qué | Cómo |
|---|---|---|
| **HF Hub (Jaisato)** | Adapter safetensors + model card con métricas | `python -m adapterforge.serve.push_to_hub ...` o tag `v*` (release-hub.yml) |
| **HF Space demo** | Gradio + `llama-cpp-python` CPU cargando el GGUF desde el Hub | carpeta `space/`; deploy-hf.yml la sincroniza con `Jaisato/adapterforge-demo` |
| **vLLM multi-LoRA (GPU)** | Base residente + N adapters por petición | `docker compose --profile serve-vllm up vllm`; guía en `serve/vllm_lora/README.md` |
| **Ollama (CPU/edge/VPS)** | GGUF Q4_K_M + Modelfile | `make export` y `ollama create juristune -f artifacts/gguf/Modelfile` |

### Secrets de GitHub Actions

| Secret | Usado por | Descripción |
|---|---|---|
| `HF_TOKEN` | release-hub.yml, deploy-hf.yml | Token de **escritura** del namespace `Jaisato` (publicar adapter y Space) |
| `OPENAI_API_KEY` (opcional) | jobs de síntesis/juez si se automatizan | Clave del proveedor del teacher vía litellm |

## CI/CD

| Workflow | Disparador | Qué hace |
|---|---|---|
| `ci.yml` | push a master / PR / manual | Workflow reutilizable [`jaisato/.github` python-ci](https://github.com/jaisato/.github): `ruff check` + `pytest` (Python 3.12, núcleo sin GPU/red) y `pip-audit` informativo; más `actionlint` sobre los cuatro workflows. Acciones ancladas a SHA (Dependabot las actualiza semanalmente) |
| `smoke-train.yml` | PR que toca `train/**`, `configs/**`, Dockerfile o el sample SFT | Runner `[self-hosted, gpu]`: construye la imagen, ejecuta `make train-smoke` (30 pasos reales) y **verifica que la loss final < inicial** parseando `metrics.json` |
| `release-hub.yml` | tag `v*` | Levanta vLLM con el adapter, corre `domain_bench` con `--min-score` como gate y, si pasa, publica en HF Hub con `HF_TOKEN` |
| `deploy-hf.yml` | tag `v*` / manual | Sube `space/` al Space `Jaisato/adapterforge-demo` |

El runner self-hosted es la propia máquina WSL2 con GPU (los runners gratuitos
de GitHub no tienen GPU); se registra con las etiquetas `self-hosted, gpu`.

## Puntos de extensión

Buscables en el código como `TODO(jairo):`

- **ORPO** como alternativa a SFT+DPO en una sola fase (`train/dpo.py`, ADR 0002).
- **Packing de secuencias y curriculum** por dificultad (`data/format.py`, `train/sft.py`).
- **lm-eval-harness completo integrado** con gate en CI (`scripts/lm_eval_runner.sh`).
- **Cuantización AWQ** de la base fusionada para la ruta vLLM (`serve/export_gguf.sh`).
- **Weights & Biases** como tracking alternativo a MLflow (`train/common.py`).
- **Corpus real de NormaRAG** (P1 del portfolio): conector a su Postgres/MinIO y
  mini-conector BOE standalone (`data/extract.py`).
- **Pares de preferencia online** (teacher fuerte vs débil) para DPO (`data/synthesize/dpo.py`).
- **Carga dinámica de adapters** en vLLM y auth por API key (`serve/vllm_lora/README.md`).

## Estructura del repositorio

```text
adapterforge/
├── compose.yaml                  # profiles: train (GPU), track (mlflow:5000), serve-vllm
├── Makefile                      # data / train / train-smoke / dpo / eval / export / test / lint
├── pyproject.toml                # core mínimo + grupos dev/train/eval/track
├── configs/                      # experimentos YAML validados (ExperimentConfig)
├── prompts/                      # plantillas del teacher: resumen, qa, obligaciones, llano
├── data/
│   ├── raw/sample_corpus.jsonl   # 12 documentos estilo BOE (inventados, realistas)
│   └── processed/                # sample_sft.jsonl (~40) + sample_dpo.jsonl (15) + splits
├── evals/domain_bench.jsonl      # 25 casos con referencia (extract/qa/resumen)
├── src/adapterforge/
│   ├── settings.py               # pydantic-settings (env AF_)
│   ├── config.py                 # ExperimentConfig + validación VRAM/OOM
│   ├── llm.py                    # cliente litellm / OpenAI-compatible (httpx)
│   ├── jsonl.py                  # utilidades JSONL (stdlib)
│   ├── data/                     # extract · synthesize (--offline) · filter · format
│   ├── train/                    # sft.py · dpo.py · runtime (Unsloth try-import) · common
│   ├── eval/                     # metrics (EM/ROUGE-L propio) · domain_bench · report
│   └── serve/                    # merge_adapter · push_to_hub · render_modelfile
├── serve/                        # export_gguf.sh · ollama/Modelfile.j2 · vllm_lora/ · templates/
├── space/                        # demo HF Space (Gradio + llama-cpp-python CPU)
├── docker/Dockerfile.train       # CUDA 12.4.1, versiones fijadas, usuario no root
├── scripts/                      # check_smoke.py · deploy_space.py · lm_eval_runner.sh
├── tests/                        # ~30 tests sin GPU/red (respx para el path HTTP)
├── docs/adr/                     # 3 decisiones de arquitectura
└── .github/workflows/            # ci · smoke-train · release-hub · deploy-hf
```

## Licencia

[MIT](LICENSE) © 2026 Jairo Sarabia. El corpus de muestra es **ficticio**
(documentos inventados con estructura BOE); los textos reales del BOE son
reutilizables citando la fuente, condición que se documenta en la dataset card
al publicar datasets derivados.
