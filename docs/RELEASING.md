# Proceso de release

Cómo se numera, se etiqueta y se publica una versión de AdapterForge, y qué
tiene que existir antes para que `release-hub.yml` llegue al Hub.

## Dos ejes de versión que no hay que mezclar

| Eje | Dónde vive | Ejemplo |
|---|---|---|
| **Paquete `adapterforge`** (SemVer) | `pyproject.toml` (`version`), `src/adapterforge/__init__.py` (`__version__`), sección `## [X.Y.Z] - fecha` de `CHANGELOG.md`, tag git `vX.Y.Z`, GitHub Release | `0.1.0` / `v0.1.0` |
| **Iteración del adapter y del dataset JurisTune** | `configs/qlora_qwen3-4b_v*.yaml`, `artifacts/qlora_qwen3-4b-v*/`, ficheros `juristune-v*_*.jsonl` del dataset `Jaisato/juristune-datasets`, nombres `juristune-v*-eval` en Ollama | `v6` (el adapter final) |

Una release del paquete publica **un** adapter concreto: el run cuyo
`adapter/` indica la variable de repositorio `ADAPTER_DIR` (hoy, una copia
del run `artifacts/qlora_qwen3-4b-v6/` fuera del workspace del runner).
`release-hub.yml` comprueba que ese run es el documentado en `RUN_EVIDENCE`
(`evals/results/v6/`): su `metrics.json` tiene que ser idéntico a
`train_metrics.json` y sus pesos casar con `adapter_model.sha256`. La
sección del CHANGELOG de cada versión dice qué adapter publica. Cambiar de
adapter publicado es, como mínimo, un cambio MINOR, y exige versionar antes
su evidencia y apuntar `RUN_EVIDENCE` a ella.

Los cuatro ficheros del primer eje tienen que contar la misma versión.
`scripts/check_release.py` lo comprueba; con `--tag` también que el tag
coincide, que la sección de la versión tiene fecha real (ni el marcador
`YYYY-MM-DD` ni posterior a mañana) y que `[Unreleased]` está vacío. Lo
ejecutan `pytest` en cada CI (`tests/test_release_coherence.py`, sin tag) y
el job `verify` de `release-hub.yml` (con el tag) antes de tocar la GPU.

## Cuándo subir la versión (SemVer en 0.x)

- **PATCH** (`0.1.0` → `0.1.1`): correcciones y documentación sin cambiar
  CLI, configs ni formato de datos.
- **MINOR** (`0.1.0` → `0.2.0`): funcionalidad nueva, cambios en la API
  pública (`from adapterforge... import`), flags de CLI, esquema de
  `ExperimentConfig`, formato de los JSONL, o un adapter publicado distinto.
  En `0.x` también cubre los cambios incompatibles.
- **MAJOR**: cuando la API se declare estable (`1.0.0`).

## Checklist de release

1. **Fusionar** lo que vaya a entrar. Mover las entradas de `## [Unreleased]`
   del `CHANGELOG.md` a una sección nueva `## [X.Y.Z] - YYYY-MM-DD` y dejar
   `[Unreleased]` vacío. Actualizar los enlaces del final
   (`[Unreleased]: .../compare/vX.Y.Z...HEAD`, `[X.Y.Z]: .../releases/tag/vX.Y.Z`).
2. **Subir la versión** (si procede) en `pyproject.toml` y en
   `src/adapterforge/__init__.py`, las dos.
3. **Fijar la fecha** el día en que se va a etiquetar: `YYYY-MM-DD` pasa a
   ser la fecha real (la del tag) en un commit `chore(release): vX.Y.Z`, por
   PR como el resto. Antes de eso `check_release.py --tag` falla a propósito.
4. **Comprobar** en local, con el `master` que se va a etiquetar:

   ```bash
   python scripts/check_release.py --tag vX.Y.Z
   make lint && make test
   ```

5. **Requisitos previos de `release-hub.yml`** (sin ellos el tag no publica
   nada; ver [Qué dispara el tag](#qué-dispara-el-tag)):
   - Runner self-hosted con GPU registrado y en línea con las etiquetas
     `self-hosted, gpu` (`gh api repos/jaisato/adapterforge/actions/runners`).
     Es la máquina WSL2 con Docker + nvidia-container-toolkit.
   - El run del adapter **fuera del workspace del runner**. `artifacts/` no
     se versiona y `actions/checkout` limpia el workspace (`git clean -ffdx`)
     al empezar cada job, así que nada de lo que se copie dentro de
     `<runner>/_work/` llega a la publicación. Se copia `adapter/` (con
     `adapter_config.json` y `adapter_model.safetensors`) y su
     `metrics.json` a un directorio de la máquina del runner y se apunta
     `ADAPTER_DIR` a su `adapter/` con ruta absoluta:

     ```bash
     dest="$HOME/adapterforge-release/qlora_qwen3-4b-v6"
     mkdir -p "$dest"
     cp -r artifacts/qlora_qwen3-4b-v6/adapter artifacts/qlora_qwen3-4b-v6/metrics.json "$dest"/
     gh variable set ADAPTER_DIR -R jaisato/adapterforge --body "$dest/adapter"
     ```

     El job falla pronto, con mensaje, si la variable falta, no es absoluta o
     apunta dentro del workspace, si falta algún fichero o si el run no es
     el de `evals/results/v6/`. `push_to_hub` escribe la model card
     (`README.md`) en ese `adapter/` y no sube `training_args.bin`, estados
     del optimizador ni checkpoints.
   - Secret `HF_TOKEN` del repositorio con permiso de **escritura** en el
     namespace `Jaisato` (`gh secret set HF_TOKEN`), que usan
     `release-hub.yml` y `deploy-hf.yml`.
   - Espacio en disco y red para que vLLM descargue `Qwen/Qwen3-4B-Instruct-2507`
     la primera vez (volumen `hf-cache`).
6. **Etiquetar y empujar** (el push del tag es el disparador):

   ```bash
   git checkout master && git pull
   git tag -a vX.Y.Z -m "AdapterForge X.Y.Z — JurisTune v6" && git push origin vX.Y.Z
   ```

7. **Vigilar** el workflow del tag:

   ```bash
   gh run list -R jaisato/adapterforge --workflow release-hub.yml
   ```

   El `results.json` del gate queda como artefacto `release-eval` del run.
8. **GitHub Release** con las notas del CHANGELOG (no la crea ningún workflow):

   ```bash
   gh release create vX.Y.Z -R jaisato/adapterforge --title "vX.Y.Z" --notes-file notas.md
   ```

9. Comprobar en el Hub la model card de
   [`Jaisato/juristune-qwen3-4b-lora`](https://huggingface.co/Jaisato/juristune-qwen3-4b-lora).
   La demo del Space se despliega aparte, cuando el GGUF esté publicado
   ([Lo que el tag no publica](#lo-que-el-tag-no-publica)).

## Qué dispara el tag

Un push de tag `v*` lanza **solo** `release-hub.yml`:

| Job | Runner | Qué hace |
|---|---|---|
| `verify` | `ubuntu-latest` | `python scripts/check_release.py --tag $GITHUB_REF_NAME`: el tag es `v` + la versión de `pyproject.toml`, que coincide con `__version__` y con la última sección del CHANGELOG, con fecha real y `[Unreleased]` vacío. Después, que el commit etiquetado está en `master`. Si algo falla, el job de GPU no arranca. |
| `evaluate-and-publish` (`needs: verify`) | `[self-hosted, gpu]` | Instala el paquete con el extra `hub`; comprueba `ADAPTER_DIR` (absoluta, fuera del workspace, con `adapter_config.json`, `adapter_model.safetensors` y `../metrics.json`) y que es el run de `RUN_EVIDENCE`; levanta vLLM (`compose.yaml`, perfil `serve-vllm`, que monta `ADAPTER_DIR` como `juristune`); corre `adapterforge.eval.domain_bench --bench $BENCH --min-score $BENCH_MIN_SCORE` contra el adapter; **solo si el gate pasa**, `adapterforge.serve.push_to_hub` sube `ADAPTER_DIR` con la model card renderizada (`serve/templates/model_card.md` + `metrics.json` + resultados del bench) a `Jaisato/juristune-qwen3-4b-lora` (crea el repo si no existe). Siempre apaga vLLM y sube `artifacts/eval/adapter.json` como artefacto. |

Variables del workflow:

- `ADAPTER_DIR`: variable de repositorio (`gh variable set`), ruta absoluta
  al `adapter/` del run en la máquina del runner, fuera de `_work/`.
- `RUN_EVIDENCE` (`evals/results/v6`): evidencia versionada del run que se
  publica (`train_metrics.json`, `adapter_model.sha256`).
- `BENCH` (`evals/domain_bench_v2.jsonl`, 111 casos) y `BENCH_MIN_SCORE`
  (`0.85`): por encima de la base (0.840) y por debajo de v6 (0.898), ambos
  medidos sirviendo el modelo como GGUF Q4_K_M en Ollama (los números de
  referencia están en `evals/results/v6/`). En el gate el adapter se sirve
  con vLLM sobre la base sin cuantizar, así que la puntuación puede variar
  algo; si el gate no pasa por poco, revisa `release-eval` antes de tocar el
  umbral.

Lo que **no** necesita: `OPENAI_API_KEY` ni ningún teacher (no sintetiza
datos), MLflow, ni el dataset de HF.

## Lo que el tag no publica

- El **GGUF** que carga el Space (`Jaisato/juristune-qwen3-4b-gguf`, variable
  `AF_GGUF_REPO` del Space) se genera con `serve/export_gguf.sh` y se sube a
  mano (`TODO(jairo)` en `src/adapterforge/serve/push_to_hub.py`). Sin él el
  Space no arranca: `space/app.py` descarga el GGUF al importarse y el Space
  queda en «Runtime error».
- La **demo del Space** (`Jaisato/adapterforge-demo`): `deploy-hf.yml` sube
  `space/` con `HF_TOKEN` (`scripts/deploy_space.py`), pero ya no se dispara
  con los tags: se lanza a mano cuando el GGUF está publicado, y antes de
  subir nada comprueba que existe (`AF_GGUF_REPO`/`AF_GGUF_FILE`, con los
  mismos valores por defecto que `space/app.py`; si el Space usa otros,
  defínelos también como variables del repositorio):

  ```bash
  gh workflow run deploy-hf.yml -R jaisato/adapterforge
  ```

- El **dataset** (`Jaisato/juristune-datasets`) se mantiene aparte
  (`make data-pull` lo descarga).
- La **GitHub Release** (paso 8).

## Si algo sale mal

- `verify` falla: la versión no cuadra, falta la fecha, quedan entradas en
  `[Unreleased]` o el tag no apunta a un commit de `master`. Corrige en
  `master` por PR y vuelve a etiquetar. Mientras no se haya publicado nada
  en el Hub, se puede borrar el tag
  (`git push origin :refs/tags/vX.Y.Z && git tag -d vX.Y.Z`) y recrearlo;
  un tag ya publicado no se mueve: se saca un PATCH.
- `evaluate-and-publish` se queda en cola: no hay runner en línea con las
  etiquetas (`gh api repos/jaisato/adapterforge/actions/runners`). GitHub lo
  cancela a las 24 h; registra el runner y relanza el run (`gh run rerun <id>`).
- Falla la comprobación del run: revisa `ADAPTER_DIR` (absoluta, fuera de
  `_work/`, con `metrics.json` en el directorio padre) y que la copia es la
  del run v6 (`sha256sum adapter_model.safetensors`).
- El gate no pasa: no se publica nada. Revisa `release-eval` (predicciones
  crudas por caso), corrige datos o entrenamiento y relanza.
- `push_to_hub` falla por permisos: `HF_TOKEN` debe ser un token de
  escritura del namespace `Jaisato`.

## Estado de partida (2026-10-04)

- `0.1.0` está declarada en `pyproject.toml` y `__version__` pero nunca se
  etiquetó: no hay tags ni GitHub Releases. La primera release es `v0.1.0`
  y publica JurisTune v6; el CHANGELOG recoge todo lo fusionado desde el
  primer commit, con la fecha pendiente (`YYYY-MM-DD`).
- En el Hub solo existe el dataset privado `Jaisato/juristune-datasets`; el
  repo del adapter lo crea la primera release. El repo GGUF y el Space
  siguen pendientes ([Lo que el tag no publica](#lo-que-el-tag-no-publica)).
- No hay runner self-hosted registrado, ni secret `HF_TOKEN`, ni variable
  `ADAPTER_DIR`: son el paso 5 del checklist.

Para empezar, con `master` ya fusionado y los requisitos del paso 5 listos:
una PR `chore(release): v0.1.0` que cambie `YYYY-MM-DD` por la fecha del
día y, tras fusionarla,

```bash
python scripts/check_release.py --tag v0.1.0
git tag -a v0.1.0 -m "AdapterForge 0.1.0 — JurisTune v6" && git push origin v0.1.0
```
