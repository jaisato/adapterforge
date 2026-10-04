# Proceso de release

Cómo se numera, se etiqueta y se publica una versión de AdapterForge, y qué
tiene que existir antes para que `release-hub.yml` llegue al Hub.

## Dos ejes de versión que no hay que mezclar

| Eje | Dónde vive | Ejemplo |
|---|---|---|
| **Paquete `adapterforge`** (SemVer) | `pyproject.toml` (`version`), `src/adapterforge/__init__.py` (`__version__`), sección `## [X.Y.Z] - fecha` de `CHANGELOG.md`, tag git `vX.Y.Z`, GitHub Release | `0.1.0` / `v0.1.0` |
| **Iteración del adapter y del dataset JurisTune** | `configs/qlora_qwen3-4b_v*.yaml`, `artifacts/qlora_qwen3-4b-v*/`, ficheros `juristune-v*_*.jsonl` del dataset `Jaisato/juristune-datasets`, nombres `juristune-v*-eval` en Ollama | `v6` (el adapter final) |

Una release del paquete publica **un** adapter concreto: el que indica
`ADAPTER_DIR` en `.github/workflows/release-hub.yml` (hoy
`artifacts/qlora_qwen3-4b-v6/adapter`, con `METRICS` y el benchmark `BENCH`
del mismo run). La sección del CHANGELOG de cada versión dice cuál es. Cambiar
de adapter publicado es, como mínimo, un cambio MINOR.

Los cuatro ficheros del primer eje tienen que contar la misma versión.
`scripts/check_release.py` lo comprueba (y con `--tag` también que el tag
coincide); lo ejecutan `pytest` en cada CI (`tests/test_release_coherence.py`)
y el job `verify` de `release-hub.yml` antes de tocar la GPU.

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
   del `CHANGELOG.md` a una sección nueva `## [X.Y.Z] - AAAA-MM-DD` (fecha del
   día del tag) y dejar `[Unreleased]` vacío. Actualizar los enlaces del
   final (`[Unreleased]: .../compare/vX.Y.Z...HEAD`, `[X.Y.Z]: .../releases/tag/vX.Y.Z`).
2. **Subir la versión** (si procede) en `pyproject.toml` y en
   `src/adapterforge/__init__.py`, las dos.
3. **Comprobar** en local, con el `master` que se va a etiquetar:

   ```bash
   python scripts/check_release.py --tag vX.Y.Z
   make lint && make test
   ```

   Commit (por PR, como el resto) con el título `chore(release): vX.Y.Z`.
4. **Requisitos previos de `release-hub.yml`** (sin ellos el tag no publica
   nada; ver [Qué dispara el tag](#qué-dispara-el-tag)):
   - Runner self-hosted con GPU registrado y en línea con las etiquetas
     `self-hosted, gpu` (`gh api repos/jaisato/adapterforge/actions/runners`).
     Es la máquina WSL2 con Docker + nvidia-container-toolkit.
   - El adapter y sus métricas **en el workspace del runner**:
     `artifacts/` no se versiona, así que hay que copiar el run completo
     (`adapter/` con `adapter_config.json` y `metrics.json`) a
     `<runner>/_work/adapterforge/adapterforge/artifacts/qlora_qwen3-4b-v6/`
     antes de etiquetar. El job falla pronto, con mensaje, si falta.
   - Secret `HF_TOKEN` del repositorio con permiso de **escritura** en el
     namespace `Jaisato` (`gh secret set HF_TOKEN`), que usan
     `release-hub.yml` y `deploy-hf.yml`.
   - Espacio en disco y red para que vLLM descargue `Qwen/Qwen3-4B-Instruct-2507`
     la primera vez (volumen `hf-cache`).
5. **Etiquetar y empujar** (el push del tag es el disparador):

   ```bash
   git checkout master && git pull
   git tag vX.Y.Z && git push origin vX.Y.Z
   # o anotado, recomendable: git tag -a vX.Y.Z -m "AdapterForge X.Y.Z — JurisTune v6"
   ```

6. **Vigilar** los dos workflows del tag:

   ```bash
   gh run list -R jaisato/adapterforge --workflow release-hub.yml
   gh run list -R jaisato/adapterforge --workflow deploy-hf.yml
   ```

   El `results.json` del gate queda como artefacto `release-eval` del run.
7. **GitHub Release** con las notas del CHANGELOG (no la crea ningún workflow):

   ```bash
   gh release create vX.Y.Z -R jaisato/adapterforge --title "vX.Y.Z" --notes-file notas.md
   ```

8. Comprobar en el Hub la model card de
   [`Jaisato/juristune-qwen3-4b-lora`](https://huggingface.co/Jaisato/juristune-qwen3-4b-lora)
   y que el Space [`Jaisato/adapterforge-demo`](https://huggingface.co/spaces/Jaisato/adapterforge-demo)
   arranca. Si el Space no encuentra el GGUF, ver
   [Lo que el tag no publica](#lo-que-el-tag-no-publica).

## Qué dispara el tag

Un push de tag `v*` lanza dos workflows independientes:

### `release-hub.yml`

| Job | Runner | Qué hace |
|---|---|---|
| `verify` | `ubuntu-latest` | `python scripts/check_release.py --tag $GITHUB_REF_NAME`: el tag es `v` + la versión de `pyproject.toml`, que coincide con `__version__` y con la última sección del CHANGELOG. Si falla, el job de GPU no arranca. |
| `evaluate-and-publish` (`needs: verify`) | `[self-hosted, gpu]` | Instala el núcleo + `huggingface_hub`; comprueba que existen `$ADAPTER_DIR/adapter_config.json` y `$METRICS`; levanta vLLM (`compose.yaml`, perfil `serve-vllm`, que monta `./$ADAPTER_DIR` como `juristune`); corre `adapterforge.eval.domain_bench --bench $BENCH --min-score $BENCH_MIN_SCORE` contra el adapter; **solo si el gate pasa**, `adapterforge.serve.push_to_hub` sube `$ADAPTER_DIR` con la model card renderizada (`serve/templates/model_card.md` + `$METRICS` + resultados del bench) a `Jaisato/juristune-qwen3-4b-lora` (crea el repo si no existe). Siempre apaga vLLM y sube `artifacts/eval/adapter.json` como artefacto. |

Variables (bloque `env` del workflow): `ADAPTER_DIR`
(`artifacts/qlora_qwen3-4b-v6/adapter`), `METRICS`
(`artifacts/qlora_qwen3-4b-v6/metrics.json`), `BENCH`
(`evals/domain_bench_v2.jsonl`, 111 casos) y `BENCH_MIN_SCORE` (`0.55`, un
suelo de cordura: v6 puntúa 0.898 en ese benchmark servido como GGUF; los
números de referencia están en `evals/results/v6/`).

Lo que **no** necesita: `OPENAI_API_KEY` ni ningún teacher (no sintetiza
datos), MLflow, ni el dataset de HF.

### `deploy-hf.yml`

En `ubuntu-latest`: sube la carpeta `space/` al Space
`Jaisato/adapterforge-demo` con `HF_TOKEN` (`scripts/deploy_space.py`). También
se puede lanzar a mano (`workflow_dispatch`).

## Lo que el tag no publica

- El **GGUF** que carga el Space (`Jaisato/juristune-qwen3-4b-gguf`, variable
  `AF_GGUF_REPO`) se genera con `serve/export_gguf.sh` y se sube a mano
  (`TODO(jairo)` en `serve/push_to_hub.py`). Sin él el Space arranca pero no
  puede cargar el modelo.
- El **dataset** (`Jaisato/juristune-datasets`) se mantiene aparte
  (`make data-pull` lo descarga).
- La **GitHub Release** (paso 7).

## Si algo sale mal

- `verify` falla: la versión no cuadra. Corrige los ficheros en `master`
  por PR y vuelve a etiquetar. Mientras no se haya publicado nada en el Hub,
  se puede borrar el tag (`git push origin :refs/tags/vX.Y.Z && git tag -d vX.Y.Z`)
  y recrearlo; un tag ya publicado no se mueve: se saca un PATCH.
- `evaluate-and-publish` se queda en cola: no hay runner en línea con las
  etiquetas (`gh api repos/jaisato/adapterforge/actions/runners`). GitHub lo
  cancela a las 24 h; registra el runner y relanza el run (`gh run rerun <id>`).
- El gate no pasa: no se publica nada. Revisa `release-eval` (predicciones
  crudas por caso), corrige datos o entrenamiento y relanza.
- `push_to_hub` falla por permisos: `HF_TOKEN` debe ser un token de
  escritura del namespace `Jaisato`.

## Estado de partida (2026-10-04)

- `0.1.0` está declarada en `pyproject.toml` y `__version__` pero nunca se
  etiquetó: no hay tags ni GitHub Releases. La primera release es `v0.1.0`
  y publica JurisTune v6; el CHANGELOG recoge todo lo fusionado desde el
  primer commit.
- En el Hub solo existe el dataset privado `Jaisato/juristune-datasets`; el
  repo del adapter y el Space los crea la primera release.
- No hay runner self-hosted registrado ni secret `HF_TOKEN`: son el punto 4
  del checklist.

Para empezar, con `master` ya fusionado y comprobado:

```bash
git tag v0.1.0 && git push origin v0.1.0
```
