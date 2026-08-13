# AdapterForge — objetivos de desarrollo y operación.
# Ejecutar desde WSL2 (el entrenamiento requiere Docker + nvidia-container-toolkit).

PY ?= python
CONFIG ?= configs/qlora_qwen3-4b.yaml
DPO_CONFIG ?= configs/dpo_qwen3-4b.yaml
SMOKE_CONFIG ?= configs/qlora_smoke.yaml
API_BASE ?= http://localhost:11434/v1
EVAL_MODEL ?= qwen3:4b

.PHONY: help data train train-smoke dpo eval eval-report export test lint mlflow clean

help: ## Lista los objetivos disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

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
	$(PY) -m adapterforge.eval.domain_bench --api-base $(API_BASE) \
		--model $(EVAL_MODEL) --output artifacts/eval/latest.json

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
