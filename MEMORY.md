# 🧠 Memoria del Proyecto (MEMORY.md)

Verdad científica y contexto inmutable del proyecto. **Consulta este archivo y `AGENTS.md` antes de decidir o implementar lógica de negocio.**

## 1. Estado
*   **Rama:** `refactor/production-architecture`.
*   **Fase 0 completada + saneamiento posterior:** paquete `mldd` instalable (src-layout, hatchling), CLI Typer (`mldd version`; `train/tune/predict` llegan en Fases 4-7), Makefile cross-platform, CI (ruff → mypy strict → pytest, Py3.12), pre-commit. Gates locales en verde.
*   **Deuda con ticket:** `src/mldd/dataset.py` aún replica la lógica académica incorrecta (zona gris 5-6, asunción nM, sin filtro `standard_relation`, sin split LRRK2). Congelada por `tests/test_scientific_contract.py` (xfail strict); reescribir en Fase 2 (T2.2-T2.3) ANTES de cualquier entrenamiento. `train.py` es un baseline RF estático, a reemplazar en Fases 4-5.
*   **Fase 1 completada (config centralizada):** `configs/` (YAML de Hydra) es la única fuente de verdad; `mldd.config.load_config(overrides)` compone con Hydra y valida con Pydantic v2 (`extra="forbid"`), lanzando `ConfigError`. Targets/seed/umbrales/params ya NO viven en `constants.py` (ahora solo `Task` + contratos de columnas). Perfiles: 6 targets × 2 tasks, política de `stratify` fijada (clasificación sí, regresión no; ver `TrainingConfig`).
*   **Datos:** `data/` y `models/` gitignored; los CSV crudos de ChEMBL están sin trackear (regenerables vía ingesta, gestión por manifests).

## 2. Verdad Científica (NO ALTERAR)

### Targets y Datasets (ChEMBL IDs)
*   AChE `CHEMBL220` · MAO-B `CHEMBL2039` *(¡nunca CHEMBL203!)* · GSK3B `CHEMBL262` · LRRK2 `CHEMBL1075104` (3 datasets: Global, WT, G2019S vía regex estricto sobre `assay_description`).

### Curación y Etiquetas
*   Etiqueta: `pchembl_value`. Rescate solo con `standard_relation == '='`: nM → `9 - log10(v)`; µM → `6 - log10(v)`.
*   Deduplicación por InChIKey (media); descartar moléculas con `ΔpChEMBL > 1.0`.
*   Clasificación: activo `≥ 6`, inactivo `< 6`. **Sin zona gris: los valores 5-6 NO se excluyen.**

### Features
*   Estandarización RDKit (LargestFragmentChooser + Uncharger). Lipinski: MW, LogP, HBD, HBA.
*   PubChem FP 881 bits vía padelpy (`removesalt=True, standardizenitro=True`).
*   Filtros de varianza: 0 pre-proceso, 0.16 post-merge. Outliers: Z-score `|z|>3`, 5 descriptores, 1 pasada.

## 3. DoD
1. Cero hardcode: parámetros/umbrales/targets/paths en `configs/` (los tests fallan si aparecen literales científicos en `src/`).
2. Pydantic v2 en fronteras I/O; Pandera para DataFrames.
3. I/O solo en `cli.py`, `io.py`, `trainer.py`, `predictor.py`; el resto, funciones puras.
4. `ruff`, `mypy --strict`, `pytest` en verde; logging central (`logging_utils.py`), nunca `print()` en `src/mldd/`.

## 4. Parity Check (±0.05, KFold 5, semilla 42)
| Target | R² | MCC |
|---|---|---|
| AChE | 0.711 | 0.65 |
| GSK3B | 0.676 | 0.69 |
| MAO-B | 0.649 | 0.62 |
| LRRK2 | 0.673 | 0.46 |
| LRRK2_WT | 0.571 | 0.47 |
| LRRK2_G2019S | 0.638 | 0.52 |

## 5. Errores ya cometidos (no repetir)
*   IDs ChEMBL inventados (CHEMBL1908/204/203): usar solo los de §2, guardados por `tests/test_constants.py`.
*   sklearn ≥1.6: `mean_squared_error(..., squared=False)` ya no existe → `root_mean_squared_error`.
*   rdkit ≥2025.9 trae stubs rotos que abortan mypy → `scripts/ensure_rdkit_stubs_ok.py` (lo ejecutan `make typecheck` y el CI).
*   typer ≥0.27: con un único `@app.command()` el comando absorbe la raíz; usar `@app.callback()`.
