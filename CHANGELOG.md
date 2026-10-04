# Changelog

Todos los cambios relevantes de AdapterForge se documentan aquí. El formato
sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y las
versiones, [SemVer](https://semver.org/lang/es/): la versión es la del paquete
`adapterforge` (`pyproject.toml`, `adapterforge.__version__`, tag `vX.Y.Z`).
Las iteraciones del adapter y del dataset JurisTune (`juristune-v1`…`v6`,
`configs/qlora_qwen3-4b_v*.yaml`) son otro eje y no cambian esta versión;
cada release indica qué adapter publica. El proceso completo está en
[docs/RELEASING.md](docs/RELEASING.md).

## [Unreleased]

Pull requests abiertas; al fusionarlas antes de etiquetar, mueve sus entradas
a la sección de la versión que se publica.

### Añadido

- `make data-pull` / `python -m adapterforge.data.pull` para descargar el
  corpus BOE y los datasets `juristune-v*` desde el dataset privado
  `Jaisato/juristune-datasets`; extra `hub` en `pyproject.toml`
  ([#8](https://github.com/jaisato/adapterforge/pull/8)).
- Resultados de JurisTune v6 en el README con la evidencia versionada en
  `evals/results/v6/` (`results.json` del bench v2, informe y métricas de
  entrenamiento) ([#8](https://github.com/jaisato/adapterforge/pull/8)).
- Dependabot para GitHub Actions y `actionlint` sobre los workflows
  ([#9](https://github.com/jaisato/adapterforge/pull/9)).

### Cambiado

- `ci.yml` usa el workflow reutilizable `python-ci` de `jaisato/.github`
  (ruff + pytest + `pip-audit` informativo) y todas las acciones quedan
  ancladas a SHA ([#9](https://github.com/jaisato/adapterforge/pull/9)).
- Enlaces del repositorio corregidos a `github.com/jaisato/adapterforge` en
  README, `pyproject.toml`, model card y Space
  ([#8](https://github.com/jaisato/adapterforge/pull/8)).
- `adapterforge.data.synthesize` pasa de un módulo de 1906 líneas a un paquete
  con un módulo por responsabilidad (examples, text, detectors, window,
  prompts, offline, dpo, online, cli); API pública, CLI y salida idénticas
  ([#10](https://github.com/jaisato/adapterforge/pull/10)).

## [0.1.0] - 2026-10-04

Primera release del paquete. Publica el adapter **JurisTune v6**
(`configs/qlora_qwen3-4b_v6.yaml`, dataset `juristune-v6`) en
`Jaisato/juristune-qwen3-4b-lora` y la demo en el Space
`Jaisato/adapterforge-demo`. Incluye todo lo fusionado en `master` desde el
inicio del repositorio (no existía ningún tag anterior).

### Añadido

- Pipeline de datos sin GPU: `data.extract` (validación y articulado del
  corpus estilo BOE), `data.synthesize` (teacher vía litellm o endpoint
  OpenAI-compatible, y modo `--offline` determinista), `data.filter`
  (dedup por shingles + Jaccard, heurísticas y juez LLM opcional),
  `data.format` (mensajes chat y splits por documento) y `data.budget`
  (presupuesto de tokens medido con el tokenizer real; gate en la síntesis).
- Conector BOE real (`scripts/fetch_boe_corpus.py`, API de datos abiertos)
  y corpus de muestra ficticio versionado (`data/raw/sample_corpus.jsonl`,
  `data/processed/sample_*.jsonl`) para smoke-train y tests.
- Entrenamiento QLoRA en 8 GB: `train.sft` (TRL `SFTTrainer`, NF4, Unsloth
  opcional, chat template oficial de Qwen3 forzado y verificado, validación
  con early stopping, abort ante loss no finita, gate de ingesta,
  reanudación `--resume` desde checkpoint) y `train.dpo` (TRL `DPOTrainer`);
  configs YAML validadas con `ExperimentConfig` (rechazo de combinaciones
  inviables y avisos de VRAM); tracking MLflow opcional.
- Evaluación honesta: benchmark de dominio propio (`evals/domain_bench.jsonl`,
  25 casos; `evals/domain_bench_v2.jsonl`, 111 casos) con métricas
  deterministas (exact-match por campos, containment, ROUGE-L), varios
  modelos intercalados por caso, `--repeats` con predicciones persistidas y
  `--min-score` como quality gate; informe before/after con IC 95 % del
  delta pareado y regresiones robustas (`eval.report`); `scripts/lm_eval_runner.sh`
  para lm-evaluation-harness.
- Serving en tres formatos: `serve.merge_adapter` + `serve/export_gguf.sh`
  (GGUF Q4_K_M y Modelfile de Ollama renderizado), vLLM `--enable-lora`
  multi-adapter en `compose.yaml`, `serve.push_to_hub` con model card
  renderizada y demo HF Space (`space/`, Gradio + llama-cpp-python en CPU).
- Iteraciones del dataset y del adapter JurisTune: v1 (dataset 3986
  ejemplos), v2/v2b, v3 (ventana estructural, etiquetas validadas, split por
  documento), v4 (desambiguación de plazos, resumen estilo bench, bench v2),
  v5 (multa por tramo sancionador, eco del propósito, supresión de pares
  contradictorios) y v6 (resumen denso, eco de la fórmula normativa,
  containment insensible a artículos): v6 es el adapter final.
- CI/CD: `ci.yml` (ruff + pytest sin GPU ni red), `smoke-train.yml` (runner
  self-hosted con GPU: 30 pasos reales y verificación de que la loss
  desciende), `release-hub.yml` (tag `v*`: quality gate con el benchmark de
  dominio y publicación en HF Hub) y `deploy-hf.yml` (Space).
- Proceso de release: este `CHANGELOG.md`, `docs/RELEASING.md`,
  `scripts/check_release.py` (misma versión en `pyproject.toml`,
  `__version__`, CHANGELOG y tag; lo ejecuta `pytest` y el job `verify` de
  `release-hub.yml`).
- Documentación: README con arquitectura, presupuesto de VRAM y matriz de
  fallback OOM, puesta en marcha y ADR 0001-0003 (QLoRA + Unsloth frente a
  full fine-tuning, DPO frente a ORPO, GGUF/Ollama frente a vLLM-LoRA).

### Cambiado

- `release-hub.yml` y el servicio `vllm` de `compose.yaml` apuntan al adapter
  final (`artifacts/qlora_qwen3-4b-v6/adapter`, parametrizable con
  `ADAPTER_DIR`) y el gate usa `evals/domain_bench_v2.jsonl`; el job falla
  pronto si el adapter o sus métricas no están en el runner.
- La demo del Space usa Gradio 6.15.1
  ([#2](https://github.com/jaisato/adapterforge/pull/2)).
- Los datasets generados (`data/raw/corpus_boe.jsonl`,
  `data/processed/juristune*.jsonl`) superan el límite de GitHub y dejan de
  versionarse: viven en el dataset privado `Jaisato/juristune-datasets`.
  Tampoco se versionan la caché compilada de Unsloth ni `teacher.log`
  ([#5](https://github.com/jaisato/adapterforge/pull/5)).
- `ci.yml` se ejecuta también en los push a `master` y
  `test_train_guards` se recolecta sin `transformers`
  ([#4](https://github.com/jaisato/adapterforge/pull/4)).
- `pytest >= 9.0.3` (PYSEC-2026-1845)
  ([#7](https://github.com/jaisato/adapterforge/pull/7)).

### Corregido

- `data.format`: los ratios de split deben ser tres números finitos no
  negativos que sumen 1; cada registro se conserva exactamente una vez
  ([#5](https://github.com/jaisato/adapterforge/pull/5)).
- `adapterforge.llm`: `LlmError` cuando una respuesta HTTP correcta trae un
  JSON inválido ([#5](https://github.com/jaisato/adapterforge/pull/5)).
- `eval.metrics`: el containment compara tokens completos (`2027` ya no
  «contiene» `1 de enero de 2027`)
  ([#6](https://github.com/jaisato/adapterforge/pull/6)).
- `make data-v6` (solo offline) escribe el prefijo de trabajo
  `juristune-v6-offline` y no pisa el dataset canónico, que solo escribe
  `data-v6-teacher`.

### Seguridad

- `serve.push_to_hub` no sube `training_args.bin` (pickle del Trainer con
  `hub_token` si se pasó), ni estados de optimizador/scheduler/RNG ni
  checkpoints, y borra los que una subida anterior hubiera dejado en el repo
  del Hub ([#7](https://github.com/jaisato/adapterforge/pull/7)).
- Todos los workflows declaran `permissions: contents: read`; `smoke-train`
  no ejecuta PRs de forks en el runner propio
  ([#7](https://github.com/jaisato/adapterforge/pull/7)).
- MLflow y vLLM se publican solo en loopback (`127.0.0.1`): ninguno
  autentica peticiones ([#7](https://github.com/jaisato/adapterforge/pull/7)).

[Unreleased]: https://github.com/jaisato/adapterforge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jaisato/adapterforge/releases/tag/v0.1.0
