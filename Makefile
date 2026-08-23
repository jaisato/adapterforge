# AdapterForge — objetivos de desarrollo y operación.
# Ejecutar desde WSL2 (el entrenamiento requiere Docker + nvidia-container-toolkit).

PY ?= python
CONFIG ?= configs/qlora_qwen3-4b.yaml
DPO_CONFIG ?= configs/dpo_qwen3-4b.yaml
SMOKE_CONFIG ?= configs/qlora_smoke.yaml
API_BASE ?= http://localhost:11434/v1
EVAL_MODEL ?= qwen3:4b
BASE_MODEL ?= qwen3-base-2507-eval
ADAPTER_MODEL ?= juristune-v5-eval
REPEATS ?= 3
BENCH ?= evals/domain_bench_v2.jsonl

.PHONY: help data data-v5 data-v5-teacher train train-smoke train-v3 train-v4 train-v5 dpo eval eval-pair eval-report export test lint mlflow clean

help: ## Lista los objetivos disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# NOTA: los datasets juristune-v3_*/v4_*/v5_* commiteados en data/processed/
# son los canónicos (cada versión quedó congelada con el código de su época).
# Regenerar con el código actual produce el contenido v5: por eso los targets
# de ingesta escriben SIEMPRE el prefijo de la versión vigente (v5).
data-v5: ## Ingesta v5: desambiguación por tramo y propósito + filtro + splits por doc
	$(PY) -m adapterforge.data.synthesize --input data/interim/documents.jsonl \
		--output data/interim/sft_raw_v5.jsonl --offline --balance default
	$(PY) -m adapterforge.data.filter --input data/interim/sft_raw_v5.jsonl \
		--corpus data/interim/documents.jsonl \
		--output data/interim/sft_filtered_v5.jsonl \
		--report data/interim/filter_report_v5.json
	$(PY) -m adapterforge.data.format --input data/interim/sft_filtered_v5.jsonl \
		--output-dir data/processed --prefix juristune-v5 --split-by doc

data-v5-teacher: data-v5 ## Ejemplos qa del teacher (ventana anclada) + filtro con juez LLM
	$(PY) -m adapterforge.data.synthesize --input data/interim/documents.jsonl \
		--output data/interim/sft_teacher_v5.jsonl --tasks qa --max-docs 250 \
		--backend openai --api-base $(API_BASE) --model qwen2.5:7b-instruct \
		--token-budget 900
	$(PY) -m adapterforge.data.filter --input data/interim/sft_teacher_v5.jsonl \
		--corpus data/interim/documents.jsonl \
		--output data/interim/sft_teacher_v5_filtered.jsonl \
		--report data/interim/filter_report_v5_teacher.json \
		--judge --judge-model qwen2.5:7b-instruct --backend openai --api-base $(API_BASE)
	cat data/interim/sft_filtered_v5.jsonl data/interim/sft_teacher_v5_filtered.jsonl \
		> data/interim/sft_v5_full.jsonl
	$(PY) -m adapterforge.data.format --input data/interim/sft_v5_full.jsonl \
		--output-dir data/processed --prefix juristune-v5 --split-by doc

train-v3: ## SFT v3 (template oficial + val + early stopping) en el contenedor CUDA
	docker compose --profile train run --rm trainer \
		python -m adapterforge.train.sft --config configs/qlora_qwen3-4b_v3.yaml

train-v4: ## SFT v4 (dataset con desambiguación + resumen estilo bench)
	docker compose --profile train run --rm trainer \
		python -m adapterforge.train.sft --config configs/qlora_qwen3-4b_v4.yaml

data: ## Pipeline de datos completo en modo offline (sin API)
	$(PY) -m adapterforge.data.extract --input data/raw/sample_corpus.jsonl --output data/interim/documents.jsonl
	$(PY) -m adapterforge.data.synthesize --input data/interim/documents.jsonl --output data/interim/sft_raw.jsonl --offline --max-per-task 10
	$(PY) -m adapterforge.data.filter --input data/interim/sft_raw.jsonl --corpus data/interim/documents.jsonl --output data/interim/sft_filtered.jsonl --report data/interim/filter_report.json
	$(PY) -m adapterforge.data.format --input data/interim/sft_filtered.jsonl --output-dir data/processed --single-output data/processed/sample_sft.jsonl
	$(PY) -m adapterforge.data.synthesize --input data/interim/documents.jsonl --output data/interim/dpo_raw.jsonl --offline --dpo
	$(PY) -m adapterforge.data.format --input data/interim/dpo_raw.jsonl --dpo --single-output data/processed/sample_dpo.jsonl

train: ## SFT QLoRA dentro del contenedor CUDA (perfil train)
	docker compose --profile train run --rm trainer \
		python -m adapterforge.train.sft --config $(CONFIG) --mlflow

train-smoke: ## Smoke-train de 30 pasos (CI self-hosted GPU y validación local)
	docker compose --profile train run --rm trainer \
		python -m adapterforge.train.sft --config $(SMOKE_CONFIG) --smoke

dpo: ## Fase DPO sobre los pares de preferencia (perfil train)
	docker compose --profile train run --rm trainer \
		python -m adapterforge.train.dpo --config $(DPO_CONFIG)

eval: ## Benchmark de dominio contra un endpoint OpenAI-compatible
	$(PY) -m adapterforge.eval.domain_bench --bench $(BENCH) --api-base $(API_BASE) \
		--model $(EVAL_MODEL) --repeats $(REPEATS) --output artifacts/eval/latest.json

eval-pair: ## Before/after intercalado en una sesión (n=REPEATS por caso) + informe
	$(PY) -m adapterforge.eval.domain_bench --bench $(BENCH) --api-base $(API_BASE) \
		--model $(BASE_MODEL) --model $(ADAPTER_MODEL) --repeats $(REPEATS) \
		--output-dir artifacts/eval/pair
	$(PY) -m adapterforge.eval.report \
		--before artifacts/eval/pair/$(subst /,-,$(subst :,-,$(BASE_MODEL))).json \
		--after artifacts/eval/pair/$(subst /,-,$(subst :,-,$(ADAPTER_MODEL))).json \
		--output artifacts/eval/report_pair.md

eval-report: ## Informe before/after (requiere base.json y adapter.json previos)
	$(PY) -m adapterforge.eval.report --before artifacts/eval/base.json \
		--after artifacts/eval/adapter.json --output artifacts/eval/report.md

export: ## Merge + GGUF Q4_K_M + Modelfile de Ollama (ver serve/export_gguf.sh)
	bash serve/export_gguf.sh

test: ## Tests unitarios (sin GPU ni red)
	pytest

lint: ## Lint con ruff
	ruff check .

mlflow: ## Levanta el servidor MLflow en http://localhost:5000
	docker compose --profile track up -d mlflow

clean: ## Limpia cachés de Python y artefactos temporales de test
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

train-v5: ## SFT v5 (selección sensible a cualificador)
	docker compose --profile train run --rm trainer \
		python -m adapterforge.train.sft --config configs/qlora_qwen3-4b_v5.yaml
