# Machine Learning for drug discovery in neurodegenerative diseases

Predicting molecular bioactivity against four key neurodegenerative proteins using a dual classification and regression machine learning pipeline. Built entirely from raw ChEMBL data and structured under KDD process, leverages RDKit and PaDEL-Descriptor for molecular featurization, with explainable AI (XAI) to extract interpretable pharmacophore rules.

![ChEMBL](https://img.shields.io/badge/Data-ChEMBL_v36-E63946?style=flat)
![RDKit](https://img.shields.io/badge/RDKit-2025.09.6-emerald)
![padelpy](https://img.shields.io/badge/padelpy-0.1.16-blue)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Bayesian_Optimization-4089c1)
![SHAP](https://img.shields.io/badge/XAI-SHAP-8A2BE2?style=flat)

---

## Context

Neurodegenerative diseases — Alzheimer's, Parkinson's, and related dementias — currently affect over 55 million people worldwide with no disease-modifying treatments available. A key bottleneck in drug discovery is the early identification of potent and selective bioactive compounds from vast chemical spaces, a task that has historically required expensive and time-consuming wet-lab assays.

This project applies a full ML pipeline to predict molecular bioactivity against **four validated neurodegeneration targets**, enabling computational pre-screening that can reduce experimental burden and accelerate hit identification.

**Targets:**

| Target | Disease relevance | Biological role |
|---|---|---|
| **AChE** (Acetylcholinesterase) | Alzheimer's disease | Degrades acetylcholine — inhibition restores cholinergic transmission |
| **MAO-B** (Monoamine Oxidase B) | Parkinson's disease, oxidative stress | Metabolizes dopamine — inhibition reduces neuronal damage |
| **GSK-3β** (Glycogen Synthase Kinase-3β) | Alzheimer's / tauopathies | Hyperphosphorylates tau protein — key driver of neurofibrillary tangle formation |
| **LRRK2** (Leucine-Rich Repeat Kinase 2) | Familial & sporadic Parkinson's | Modeled in three variants: unified, wild-type (WT), and G2019S mutation |

---

## Results summary

### Regression — Potency prediction (pChEMBL value, derived from IC₅₀)

*All models reported are the **Optuna-optimized** final winners after two-stage selection (LazyPredict screening → Bayesian hyperparameter tuning).*

| Target | Final model (post Optuna) | R² | RMSE |
|---|---|---|---|
| AChE | XGBoost | **0.711** | **0.719** |
| GSK-3β | XGBoost | **0.676** | **0.719** |
| MAO-B | XGBoost | **0.649** | **0.737** |
| LRRK2 unified | XGBoost | **0.673** | **0.608** |
| LRRK2 WT | XGBoost | **0.571** | **0.601** |
| LRRK2 G2019S | RF / XGBoost | **0.638** | **0.649** |

> R² values above 0.60 on held-out test sets (80/20 split) in medicinal chemistry are considered **competitive** given the inherent noise of bioassay data.

### Classification — Active / Inactive screening filter

| Target | Best model | MCC | AUC-ROC |
|---|---|---|---|
| AChE | XGBoost | **0.65** | **0.90** |
| GSK-3β | LGBM | **0.69** | **0.91** |
| MAO-B | RF | **0.62** | **0.89** |
| LRRK2 unified | LGBM | **0.46** | **0.89** |
| LRRK2 WT | LGBM | **0.47** | **0.85** |
| LRRK2 G2019S | LDA / Passive Agressive | **0.52** | **0.85** |


> MCC (Matthews Correlation Coefficient) was selected as the primary metric due to severe class imbalance in medicinal chemistry datasets (particularly LRRK2). MCC is the only metric robust to imbalanced binary classification — accuracy and F1 are misleading in this context.

<!--
**Key empirical finding:** Extensive benchmarking with LazyPredict across dozens of models demonstrated that **tree-based ensemble methods (XGBoost, LightGBM, HistGradientBoosting) consistently outperformed deep learning architectures** on these high-dimensional tabular fingerprint datasets. This is consistent with the literature on structured molecular descriptor data.

---
-->

## Dataset

| Field | Detail |
|---|---|
| Source | ChEMBL v36 — via official Python API (`chembl_webresource_client`) |
| Access | Public — programmatically extracted |
| Raw extraction | Bioassay-filtered IC₅₀ measurements per target |
| Final curated compounds | **15,456 unique molecules** |
| Molecular representation | PubChem Fingerprints — 881-bit binary topological vectors (PaDEL-Descriptor) |

**Dataset breakdown by target (post-curation):**

| Target | Compounds |
|---|---|
| AChE | 5,593 |
| MAO-B | 4,387 |
| GSK-3β | 3,046 |
| LRRK2 (unified) | 2,430 |
| **Total** | **15,456** |

---
<!--
## Pipeline — KDD process

The project follows the **Knowledge Discovery in Databases (KDD)** framework end-to-end, from raw API extraction to interpretable pharmacophoric insights.

```
ChEMBL v36 API
      │
      ▼
01_Data_Acquisition          ← Programmatic extraction + bioassay filtering
      │
      ▼
02_Data_Curation_and_EDA     ← SMILES → InChIKey standardization · Z-score outlier removal
      │                         Lipinski Rule-of-5 validation · class imbalance analysis
      ▼
03_Feature_Engineering       ← PaDEL: 881-bit PubChem Fingerprints · constant feature removal
      │
      ▼
04_Model_Benchmarking        ← LazyPredict mass screening → Optuna hyperparameter optimization
      │                         Dual approach: Binary classification + pChEMBL regression
      ▼
05_Explainable_AI_SHAP       ← TreeExplainer SHAP values · Feature Importance
                                Pharmacophoric rule extraction from black-box models
```

---

## Methodology

### 1. Data acquisition & curation
Data were extracted programmatically from **ChEMBL v36** using the official Python client — no pre-processed dataset was used. The curation pipeline included:
- Bioassay filtering (IC₅₀ and Ki only, standard relation = `'='`)
- SMILES → InChIKey standardization via RDKit to remove duplicates
- Z-score outlier removal on pChEMBL values (|z| > 3)
- Active/Inactive labeling: threshold at pChEMBL ≥ 6 (IC₅₀ ≤ 1 µM)

### 2. Feature engineering — molecular fingerprints
Molecular structures (SMILES) were converted to **881-dimensional binary PubChem Fingerprints** using PaDEL-Descriptor. These topological vectors encode the presence/absence of specific molecular substructures, enabling ML models to learn structure-activity relationships without requiring 3D conformations.

Constant and near-constant features were removed prior to modeling to reduce noise.

### 3. Model benchmarking & optimization
A mass screening with **LazyPredict** evaluated dozens of classifiers and regressors. Top candidates were then optimized via **Optuna** Bayesian hyperparameter search with stratified 5-fold cross-validation.

Final models evaluated: XGBoost · LightGBM · HistGradientBoosting · Random Forest · SVM/SVR  
Baselines: Linear Discriminant Analysis (LDA) · PassiveAggressive Classifier

Validation: 80/20 stratified train/test split + cross-validation on training set.

### 4. Explainable AI — pharmacophoric interpretation
**SHAP TreeExplainer** was applied to regression models to identify which molecular fingerprint bits most strongly influence potency predictions. This XAI layer converts black-box models into interpretable pharmacophoric hypotheses: which molecular substructures increase or decrease predicted binding affinity, and by how much.

This is the scientifically most valuable output of the pipeline — bridging computational predictions with medicinal chemistry insight.

---

## Repository structure

```
ML-Drug-Discovery-Neuro/
├── data/
│   ├── raw/                              ← ChEMBL raw extractions per target
│   └── processed/                        ← Curated datasets + fingerprint matrices
├── notebooks/
│   ├── 01_Data_Acquisition_ChEMBL.ipynb
│   ├── 02_Data_Curation_and_EDA.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Benchmarking_and_Training.ipynb
│   └── 05_Explainable_AI_SHAP.ipynb
├── results/
│   ├── Resultados_AChE.csv               ← Predicted vs. experimental pChEMBL values
│   ├── LRRK2.csv
│   ├── figures/                          ← Lipinski chemical space · SHAP summary plots ·
│   │                                        regression scatter plots · correlation matrices
│   └── xai/                              ← Feature importance · SHAP beeswarm plots
├── LICENSE
└── README.md
```

---

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/carladdm/ML-Drug-Discovery-Neuro.git
cd ML-Drug-Discovery-Neuro

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-extract data from ChEMBL — or use processed data in data/processed/
jupyter notebook notebooks/01_Data_Acquisition_ChEMBL.ipynb

# 4. Run the pipeline in order
jupyter notebook notebooks/02_Data_Curation_and_EDA.ipynb
```

> **Note on PaDEL-Descriptor:** Fingerprint generation requires Java ≥ 8 and the PaDEL-Descriptor JAR. See [PaDEL official site](http://www.yapcwsoft.com/dd/padeldescriptor/) for installation. Alternatively, pre-computed fingerprint matrices are available in `data/processed/`.

---

## Key visualizations

<!-- Agregar capturas reales de results/figures/ -->
<!-- Suggested: -->
<!-- ![Lipinski chemical space](results/figures/lipinski_chemical_space.png) -->
<!-- ![SHAP summary AChE](results/xai/shap_summary_AChE.png) -->
<!-- ![Regression scatter GSK3B](results/figures/scatter_GSK3B_regression.png) -->
<!-- ![Model comparison](results/figures/model_benchmarking_comparison.png) -->

-->
<!--
---

## Scientific conclusions

**On model performance:** Tree-based ensembles — particularly XGBoost and LightGBM — achieve R² > 0.65 in regression and MCC > 0.65 in classification across multiple neurodegeneration targets using 881-dimensional fingerprint representations. This is consistent with the broader QSAR literature on tabular molecular descriptor data, where deep learning does not consistently outperform gradient boosting in the low-to-medium data regime (< 10K compounds per target).

**On LRRK2:** Modeling the G2019S mutation separately revealed distinct SAR (Structure-Activity Relationship) patterns relative to wild-type, reflecting the conformational changes induced by the mutation at the kinase activation loop. This has direct implications for selectivity design in Parkinson's drug discovery.

**On XAI:** SHAP analysis of regression models extracted molecular substructures (fingerprint bits) with the highest positive and negative impact on predicted pChEMBL values. These pharmacophoric features are consistent with known SAR trends in the medicinal chemistry literature for AChE and GSK-3β inhibitors.

**Limitations:**
- Models trained on in vitro biochemical assay data — cell permeability, metabolic stability, and in vivo efficacy are not modeled
- PubChem Fingerprints encode 2D topology only — 3D pharmacophoric effects (stereochemistry, conformational flexibility) are not captured
- LRRK2 dataset is smaller (2,430 compounds) — predictions carry higher uncertainty than AChE models

**Next steps:**
- [ ] Incorporate ECFP4 / Morgan fingerprints as alternative representations and compare
- [ ] Extend to ADMET property prediction (toxicity, solubility) using the same pipeline
- [ ] Apply Graph Neural Networks (GNNs) on molecular graphs as a deep learning comparison

---

## Why this project matters to me

This pipeline sits at the intersection of my two scientific backgrounds: **Materials Science computational modeling** (where I built simulation programs and worked with high-dimensional structure-property data at CONICET/INTEMA) and my current MSc thesis work in Data Science applied to biomedicine.

The decision to start from raw ChEMBL data — rather than a pre-cleaned Kaggle dataset — was deliberate. In real drug discovery workflows, data curation is where most projects fail. Building and validating that curation pipeline was the hardest and most scientifically meaningful part of this work.

---

## References & tools

- **ChEMBL v36:** Mendez D. et al., *Nucleic Acids Research*, 2019. [chembl.ac.uk](https://www.ebi.ac.uk/chembl/)
- **RDKit:** Open-source cheminformatics. [rdkit.org](https://www.rdkit.org/)
- **PaDEL-Descriptor:** Yap C.W., *J. Comput. Chem.*, 2011.
- **SHAP:** Lundberg S.M. & Lee S.I., *NeurIPS*, 2017.
- **Optuna:** Akiba T. et al., *KDD*, 2019.
- **LazyPredict:** [github.com/shankarpandala/lazypredict](https://github.com/shankarpandala/lazypredict)

---

## About the author

Carla Di Monno — Data Scientist | Chemical Engineer | Materials Science PhD candidate.
MSc in Big Data & Data Science (Distinction in Machine Learning, GPA 8.96/10).

[LinkedIn](https://www.linkedin.com/in/ing-carladimonno) · [GitHub](https://github.com/carladdm) · [ORCID](https://orcid.org/0000-0003-1463-9086)

-->
