# 🧠 Memoria del Proyecto (MEMORY.md)

Verdad científica y contexto inmutable del proyecto. **Consulta este archivo y `AGENTS.md` antes de decidir o implementar lógica de negocio.**

## 1. Estado
*   **Rama:** `refactor/production-architecture`.
*   **Fases 0-2 completadas:** paquete `mldd` instalable (src-layout, CLI Typer, CI ruff→mypy strict→pytest Py3.12, pre-commit); `configs/` (Hydra) es la única fuente de verdad, validada con Pydantic v2 (`extra="forbid"` → `ConfigError`); targets/umbrales/params ya NO en `constants.py` (solo `Task` + contratos de columnas); política `stratify` fijada (clasificación sí, regresión no). I/O de red en `io.py` (fetch ChEMBL con retry tenacity) y curación pura en `dataset.py` (rescate pChEMBL unit-aware, dedup InChIKey `Δ ≤ 1.0`, etiquetado sin zona gris, split LRRK2 WT/G2019S por regex); escáner de literales sin exenciones. **Pendiente:** ingesta masiva real (6 targets) con el CLI en Fase 5.
*   **Fase 3 completada (features):** ciencia pura en `features.py` (estandarización RDKit → `smiles_clean`, Lipinski sobre estructuras estandarizadas, Z-score `|z|>3` a 1 pasada sobre 5 descriptores, `VarianceThreshold` 0.0/0.16, merge X+y posicional) + frontera PaDEL en `io.py` (`.smi` temporal → jar embebido de padelpy). El `-fingerprints` del jar embebido reproduce **bit a bit** el `PubchemFingerprinter.xml` del notebook → sin URLs ni artefactos externos. Alineación X/y por columna `Name` (no por orden de salida: PaDEL multi-hilo reordenó filas adyacentes casi idénticas entre runs). `morgan_fp` declarado en config pero sin implementar. Parity MAO-B: ver §4.
*   **Deuda con ticket:** `train.py` es un baseline RF estático con I/O acoplado (audit B15), a reemplazar en Fases 4-5.
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
*   PubChem FP 881 bits vía padelpy (`removesalt=True, standardizenitro=True`); PaDEL recibe `canonical_smiles` crudos (NB04), no `smiles_clean`.
*   Filtros de varianza: 0 pre-proceso, 0.16 post-merge. Outliers: Z-score `|z|>3`, 5 descriptores, 1 pasada (media/std únicos, sin recálculo).

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

*   **Feature parity (Fase 3, MAO-B vs artefactos de tesis):** pipeline reproduce 4418→4387 filas (31 outliers) y la matriz final (4387, 599) **bit a bit: 0 celdas distintas de 2.62M** tras el alineamiento por `Name` (las 5-7 celdas previas eran filas intercambiadas por PaDEL multi-hilo, no deriva de versión). Deriva residual solo en la capa RDKit local vs Colab: 596 filas HBA (definición ≥2022.09), 10 `smiles_clean` (canónica estéreo C=N), 1 pChEMBL (ULP float); no afecta a la matriz de modelos (usa `canonical_smiles` crudos) ni al set de outliers.

## 5. Errores ya cometidos (no repetir)
*   IDs ChEMBL inventados (CHEMBL1908/204/203): usar solo los de §2, guardados por `tests/test_constants.py`.
*   sklearn ≥1.6: `mean_squared_error(..., squared=False)` ya no existe → `root_mean_squared_error`.
*   rdkit ≥2025.9 trae stubs rotos que abortan mypy → `scripts/ensure_rdkit_stubs_ok.py` (lo ejecutan `make typecheck` y el CI).
*   typer ≥0.27: con un único `@app.command()` el comando absorbe la raíz; usar `@app.callback()`.
*   PaDEL multi-hilo puede emitir filas fuera de orden y descartar moléculas sin avisar: alinear SIEMPRE por la columna `Name` y validar ids únicos + filas antes de entrenar (`io.compute_pubchem_fingerprints`).
