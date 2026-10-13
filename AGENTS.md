# AGENTS.md — ML-Drug-Discovery-Neuro
Plataforma de MLOps para el descubrimiento de fármacos neurológicos. Migra la experimentación académica (AChE, MAO-B, GSK3B, LRRK2) a una arquitectura de producción escalable, tipada y completamente reproducible.

## Stack y estructura
**Tecnologías clave:** Python 3.12/3.13, scikit-learn ≥1.6, XGBoost, LightGBM, RDKit ≥2026.03, PaDEL (Java 17), Pydantic v2, Hydra 1.3, Pandera, MLflow ≥2.14, pytest, Ruff, Mypy. *(Los floors viven en `pyproject.toml`; el venv local puede resolver versiones superiores.)*

**Estructura clave (lo no obvio):**
*   `configs/`: **Única fuente de verdad**. Contiene los YAML de Hydra. Ningún parámetro, umbral o target debe existir fuera de esta carpeta. *(Puente temporal autorizado hasta Fase 1/T1.1: `src/mldd/constants.py` y `src/mldd/config.py` alojan hoy targets/seed/params — ver MEMORY.md §1.)*
*   `src/mldd/`: Paquete principal bajo `src-layout`. Diseñado con interfaces desacopladas (data, features, models, training).
*   `notebooks/`: Carpeta en modo *read-only*. Contiene la experimentación original (basura académica/scripts) que sirve **solo como referencia**. No ejecutar ni importar lógica de aquí.
*   `data/` y `models/`: Directorios `.gitignored`. La gestión de I/O se realiza vía manifests generados dinámicamente por la aplicación.

## Comandos
*   **Instalación:** `pip install -e ".[dev]"`
*   **Linting y Formateo:** `make lint` (ejecuta `ruff check src tests scripts` y `ruff format --check ...`); `make format` autocorrige.
*   **Typechecking:** `make typecheck` (neutraliza los stubs rotos de rdkit vía `scripts/ensure_rdkit_stubs_ok.py` y ejecuta `mypy` en modo strict sobre `src` y `tests`, configurado en `pyproject.toml`).
*   **Testing:** `make test` (ejecuta `pytest -q`); `make cov` añade cobertura (`--cov=mldd`).
*   **Ejecución CLI:** hoy solo `mldd version`; `mldd train target=maob task=regression` (Typer + Hydra) se habilita en Fase 5.

## Convenciones
*   **Código y Docstrings:** Todo en **Inglés**. Nombres descriptivos (`snake_case` para variables/funciones, `PascalCase` para clases).
*   **Tipado Estricto:** Obligatorio en todas las firmas. Usar siempre `from __future__ import annotations`.
*   **Commits:** Usar especificación *Conventional Commits* (ej. `feat(data): add ChEMBL retry logic`).
*   **Patrones:** 
    *   *Funciones puras* para lógica de transformación de datos (sin I/O acoplado).
    *   Uso de Pydantic para validación en fronteras de I/O y Pandera para schemas de DataFrames.
    *   *Archivo de referencia:* `src/mldd/config.py` (hoy: rutas vía pathlib + constants; en Fase 1 será el estándar de dataclasses Hydra+Pydantic).

## Reglas de dominio / trampas conocidas
*   **IDs ChEMBL Trampa:** El código viejo usaba `CHEMBL203` (erróneo) en lugar del real `CHEMBL2039` para MAO-B. Revisa siempre los IDs oficiales.
*   **Umbrales Científicos:** Un molécula es activa si `pChEMBL ≥ 6`, inactiva si `< 6`. **No hay zona gris**, los valores entre 5 y 6 no se excluyen.
*   **Split LRRK2:** Diferenciar WT de G2019S no se hace con un simple flag, requiere aplicar *Regex* estricto sobre `assay_description`.
*   **Bug de Scikit-Learn:** La función `mean_squared_error` en versiones ≥1.6 ya no soporta `squared=False`. Usar `root_mean_squared_error`.
*   **Conversiones:** Durante el curado, `standard_value` está en nM o µM; el cálculo a pChEMBL difiere por 3 órdenes de magnitud. Ignorar las unidades destruye los datos.

## Forma de trabajar
*   **Planificación:** Antes de escribir una línea de código, verifica en `MEMORY.md` y `AGENTS.md` las restricciones de dominio. Piensa en TDD (escribe los tests de los casos borde primero).
*   **Tamaño:** Realiza cambios atómicos y limitados a la tarea (ej. si arreglas la ingesta, no refactorices modelos en el mismo PR).
*   **Cierre:** Al finalizar, explica qué lograste, qué divergencias académicas corregiste respecto a los notebooks y actualiza los artefactos correspondientes.

## Límites
*   ✅ **Siempre:** Añade logs estructurados (nunca `print()` dentro de `src/mldd/`; los scripts de arranque en `scripts/` pueden escribir a stdout), documenta las excepciones lanzadas y usa type hints exhaustivos.
*   ⚠ **Pregunta antes:** Añadir dependencias nuevas al `pyproject.toml`, crear módulos base en `src/mldd/`, o modificar cualquier umbral/fórmula científica.
*   🚫 **Nunca:** Hardcodear rutas, URLs, credenciales, IDs de targets o hiperparámetros dentro de `src/`. Nunca elimines ni alteres el código original de la carpeta `notebooks/`.

## Verificación
Para considerar una tarea terminada, el código debe superar:
1.  `make lint typecheck test` debe pasar en verde (0 advertencias).
2.  **Parity Check Científico:** El modelo de producción entrenado debe obtener resultados a **±0.05** en MCC (Clasificación) y R² (Regresión) respecto a los benchmarks de los notebooks (ver tabla en `MEMORY.md`).

## Memoria
*   Al empezar, lee `MEMORY.md` para conocer el estado del proyecto, el contexto científico inmutable y las decisiones tomadas.
*   Al terminar una tarea, actualízalo: refleja el estado actual, decisiones arquitectónicas importantes (con su porqué) y errores a evitar en el futuro.
*   Mantenlo breve (máximo ~50 líneas): resume, consolida o elimina lo que ya no aporte valor operativo.
*   Si un hallazgo o convención en `MEMORY.md` se convierte en una regla técnica permanente o un patrón de código, propón moverlo a `AGENTS.md` en lugar de dejarlo en la memoria.
*   No guardes nunca datos sensibles (claves, tokens de API, rutas locales de usuarios).