# Changelog

Todos los cambios relevantes de AdapterForge se documentan aquí. El formato
sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y las
versiones, [SemVer](https://semver.org/lang/es/): la versión es la del paquete
`adapterforge` (`pyproject.toml`, `adapterforge.__version__`, tag `vX.Y.Z`).
Las iteraciones del adapter y del dataset JurisTune (`juristune-v1`…`v6`,
`configs/qlora_qwen3-4b_v*.yaml`) son otro eje y no cambian esta versión;
cada release indica qué adapter publica. La fecha de cada versión es la del
día del tag: mientras se prepara queda `YYYY-MM-DD`. El proceso completo está
en [docs/RELEASING.md](docs/RELEASING.md).

## [Unreleased]

## [0.1.0] - YYYY-MM-DD

Primera release del paquete. Publica el adapter **JurisTune v6**
(`configs/qlora_qwen3-4b_v6.yaml`, dataset `juristune-v6`) en
`Jaisato/juristune-qwen3-4b-lora`. La demo del Space
`Jaisato/adapterforge-demo` no se despliega con el tag: necesita antes el
GGUF en `Jaisato/juristune-qwen3-4b-gguf`, que se publica aparte
([docs/RELEASING.md](docs/RELEASING.md)). Incluye todo lo fusionado en
`master` desde el inicio del repositorio (no existía ningún tag anterior).

### Añadido

- Pipeline de datos sin GPU: `data.extract` (validación y articulado del
  corpus estilo BOE), `data.synthesize` (teacher vía litellm o endpoint
  OpenAI-compatible, y modo `--offline` determinista), `data.filter`
  (dedup por shingles + Jaccard, heurísticas y juez LLM opcional),
  `data.format` (mensajes chat y splits por documento) y `data.budget`
  (presupuesto de tokens medido con el tokenizer real; gate en la síntesis).
- `make data-pull` / `python -m adapterforge.data.pull` para descargar el
  corpus BOE y los datasets `juristune-v*` desde el dataset privado
  `Jaisato/juristune-datasets`: descarga en la raíz del repositorio, aborta
  si git no ignoraría algún fichero en su destino y no sustituye los
  ficheros locales salvo con `--force`; extra `hub` en `pyproject.toml`
  ([#8](https://github.com/jaisato/adapterforge/pull/8)).
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
  delta pareado (media ± t·SE, t de Student con n−1 g.l. y 1.96 con n > 30;
  por tipo de tarea solo con n ≥ 5), resumen JSON (`--json-output`) y gate de
  regresiones caso a caso (`--fail-on-regression`, con `--regression-tolerance`
  para las bajadas sin rangos de repeticiones; `FAIL_ON_REGRESSION=1|true|yes`
  y `REGRESSION_TOLERANCE` en `make eval-pair`/`eval-report`) (`eval.report`,
  [#18](https://github.com/jaisato/adapterforge/pull/18),
  [#19](https://github.com/jaisato/adapterforge/pull/19));
  `scripts/lm_eval_runner.sh` para lm-evaluation-harness.
- Resultados de JurisTune v6 en el README con la evidencia versionada en
  `evals/results/v6/`: `results.json` del bench v2 del run final y del run
  base-vs-v6, informes, métricas de entrenamiento
  ([#8](https://github.com/jaisato/adapterforge/pull/8)) y sha256 de los
  pesos del adapter publicado.
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
- CI/CD: `ci.yml` llama al workflow reutilizable `python-ci` de
  `jaisato/.github` (referenciado por `@main`, no anclado a un SHA): ruff +
  pytest sin GPU ni red y `pip-audit` informativo, más un job propio que pasa
  `actionlint` a los workflows ([#9](https://github.com/jaisato/adapterforge/pull/9));
  `smoke-train.yml` (runner self-hosted con GPU: 30 pasos reales y
  verificación de que la loss desciende), `release-hub.yml` (tag `v*`:
  quality gate con el benchmark de dominio y publicación en HF Hub) y
  `deploy-hf.yml` (Space, a mano).
- Dependabot para las acciones de GitHub Actions
  ([#9](https://github.com/jaisato/adapterforge/pull/9)): una PR semanal
  agrupada y sin saltos de versión mayor, que se hacen a mano por el runner
  self-hosted ([#11](https://github.com/jaisato/adapterforge/pull/11)).
- Proceso de release: este `CHANGELOG.md`, `docs/RELEASING.md` y
  `scripts/check_release.py` (misma versión en `pyproject.toml`,
  `__version__`, CHANGELOG y tag; con el tag, además, fecha real y
  `[Unreleased]` vacío). Lo ejecutan `pytest` y el job `verify` de
  `release-hub.yml`, que también exige que el commit etiquetado esté en
  `master` ([#11](https://github.com/jaisato/adapterforge/pull/11)).
- Documentación: README con arquitectura, presupuesto de VRAM y matriz de
  fallback OOM, puesta en marcha y ADR 0001-0003 (QLoRA + Unsloth frente a
  full fine-tuning, DPO frente a ORPO, GGUF/Ollama frente a vLLM-LoRA).

### Cambiado

- `release-hub.yml` publica el adapter final (JurisTune v6). El run vive
  fuera del workspace del runner, indicado por la variable de repositorio
  `ADAPTER_DIR`, porque `actions/checkout` limpia el workspace al empezar
  cada job. El job comprueba que es el run versionado en `evals/results/v6/`
  (mismo `metrics.json` y sha256 de los pesos), el gate usa
  `evals/domain_bench_v2.jsonl` con umbral 0.85 (por encima de la base,
  0.840) y la instalación usa el extra `hub`. El servicio `vllm` de
  `compose.yaml` monta `ADAPTER_DIR` con la sintaxis larga, que admite rutas
  absolutas y relativas ([#11](https://github.com/jaisato/adapterforge/pull/11)).
- `deploy-hf.yml` ya no se dispara con los tags `v*`, solo a mano, y antes
  de subir nada comprueba que existe el GGUF que carga el Space: sin él,
  `space/app.py` falla al importarse y el Space queda en «Runtime error»
  ([#11](https://github.com/jaisato/adapterforge/pull/11)).
- `adapterforge.data.synthesize` pasa de un módulo de 1906 líneas a un paquete
  con un módulo por responsabilidad (examples, text, detectors, window,
  prompts, offline, dpo, online, cli). La API pública, las opciones de la
  CLI y la salida no cambian; el texto de uso de `--help` muestra ahora
  `python -m adapterforge.data.synthesize`
  ([#10](https://github.com/jaisato/adapterforge/pull/10)).
- Enlaces del repositorio corregidos a `github.com/jaisato/adapterforge` en
  README, `pyproject.toml`, model card y Space
  ([#8](https://github.com/jaisato/adapterforge/pull/8)). La model card
  indica la fase real del entrenamiento (la v6 es solo SFT) en vez de
  «QLoRA -> DPO» ([#11](https://github.com/jaisato/adapterforge/pull/11)).
- La demo del Space usa Gradio 6.15.1
  ([#2](https://github.com/jaisato/adapterforge/pull/2)).
- Los datasets generados (`data/raw/corpus_boe.jsonl`,
  `data/processed/juristune*.jsonl`) superan el límite de GitHub y dejan de
  versionarse: viven en el dataset privado `Jaisato/juristune-datasets`
  ([5ce3e60](https://github.com/jaisato/adapterforge/commit/5ce3e60), commit
  directo en `master`).
- Tampoco se versionan la caché compilada de Unsloth
  ([#4](https://github.com/jaisato/adapterforge/pull/4)) ni `teacher.log`
  ([#5](https://github.com/jaisato/adapterforge/pull/5)).
- `ci.yml` se ejecuta también en los push a `master` y
  `test_train_guards` se recolecta sin `transformers`
  ([#4](https://github.com/jaisato/adapterforge/pull/4)).
- `pytest >= 9.0.3` (PYSEC-2026-1845)
  ([#7](https://github.com/jaisato/adapterforge/pull/7)).

### Corregido

- La demo del Space no llegaba a construirse: `space/README.md` pedía Gradio
  6.15.1 (`sdk_version`) y `space/requirements.txt` 6.16.0, y Hugging Face
  instala las dos a la vez. `sdk_version` pasa a 6.16.0 y un test mantiene
  iguales las dos versiones
  ([#23](https://github.com/jaisato/adapterforge/pull/23)).
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
- `serve.push_to_hub` (y `data.pull`) leen el token también como `HF_TOKEN`
  del `.env`; antes solo llegaba `AF_HF_TOKEN`
  ([#8](https://github.com/jaisato/adapterforge/pull/8)).

### Seguridad

- `serve.push_to_hub` no sube `training_args.bin` (pickle del Trainer con
  `hub_token` si se pasó), ni estados de optimizador/scheduler/RNG ni
  checkpoints, y borra los que una subida anterior hubiera dejado en el repo
  del Hub ([#7](https://github.com/jaisato/adapterforge/pull/7)).
- Todos los workflows declaran `permissions: contents: read`; `smoke-train`
  no ejecuta PRs de forks en el runner propio
  ([#7](https://github.com/jaisato/adapterforge/pull/7)).
- Las acciones que usan los workflows del repositorio quedan ancladas a SHA
  y el checkout no persiste credenciales
  ([#9](https://github.com/jaisato/adapterforge/pull/9)); el workflow
  reutilizable de `jaisato/.github` se referencia por `@main`.
- MLflow y vLLM se publican solo en loopback (`127.0.0.1`): ninguno
  autentica peticiones ([#7](https://github.com/jaisato/adapterforge/pull/7)).
- MLflow 2.22.4 (antes 2.22.0) en la imagen del servidor y en la de
  entrenamiento: corrige PYSEC-2025-52 y PYSEC-2026-1640 sin salir de la
  serie 2.x; el resto de avisos de MLflow exige la serie 3.

[Unreleased]: https://github.com/jaisato/adapterforge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jaisato/adapterforge/releases/tag/v0.1.0
