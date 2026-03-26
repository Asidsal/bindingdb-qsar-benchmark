# BindingDB QSAR Benchmark

**Benchmarking Machine Learning Models for Bioactivity Prediction Across Human Protein Targets Using BindingDB: A Large-Scale Scaffold-Split Evaluation**

Abubakar Siddiq Salihu, Muhammad Sulaiman Rahma, Wan Mohd Nuzul Hakim Wan Salleh, Nura Suleiman Gwaram, Sulaiman Sani Yusuf

Department of Pure and Industrial Chemistry, Umaru Musa Yar'adua University (UMYU), Katsina, Nigeria

*Submitted to Journal of Chemical Information and Modeling, 2026*

---

## Overview

This repository contains the complete analysis pipeline for a large-scale QSAR benchmark using the BindingDB database. The study evaluates four machine learning model families across 198 human protein targets using scaffold-based train-test splitting.

**Key findings:**
- Random Forest with Morgan ECFP4 fingerprints achieves the highest median AUROC (0.968) across all target classes
- Performance hierarchy: RF >> XGBoost approximately equal to SVM > DNN
- Model performance shows only weak dependence on training set size (r = 0.235) and is independent of chemical diversity (partial r = -0.029)
- Random splitting inflated AUROC by more than 0.10 in 19% of targets, confirming the need for scaffold-based evaluation

---

## Repository Structure

```
bindingdb-qsar-benchmark/
    article01_bindingdb_qsar_benchmark.py   main analysis script (13 sections)
    requirements.txt                         Python dependencies
    README.md                                this file
    article01_outputs/
        data/
            bindingdb_curated.parquet        curated compound-target pairs
            morgan_fps.npy                   Morgan fingerprint matrix
            desc_matrix.npy                  physicochemical descriptor matrix
            combined_fps.npy                 combined feature matrix
            smiles_to_idx.pkl                SMILES to matrix row index
            model_results.csv                main benchmark results
            Table1_overall_performance.csv
            Table2_by_targetclass.csv
            Table2_with_IQR.csv
            pairwise_wilcoxon.csv
            performance_driver_correlations.csv
            TableS1_feature_ablation.csv
            TableS2_endpoint_stratification.csv
            scaffold_vs_random_comparison.csv
        figures/
            Figure1_dataset_overview.png/.svg
            Figure2_model_performance.png/.svg
            Figure3_class_heatmap.png/.svg
            Figure4_size_vs_performance.png/.svg
            Figure5_target_ranking.png/.svg
            Figure6_scaffold_vs_random.png/.svg
```

---

## Requirements

- Python 3.10+
- BindingDB complete TSV file (download from https://www.bindingdb.org)

```bash
conda create -n bindingdb_qsar python=3.10
conda activate bindingdb_qsar
pip install -r requirements.txt
```

---

## Usage

1. Download BindingDB_All.tsv from https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp

2. Update the path in Section 1 of the script:
```python
BINDINGDB_PATH = Path(r"C:/path/to/BindingDB_All.tsv")
```

3. Run sections in order (1 through 13). Long-running sections save intermediate outputs so reruns can skip them:

| Section | Description | Runtime |
|---------|-------------|---------|
| 1 | Imports and configuration | < 1 min |
| 2 | Data extraction (DuckDB) | 2-5 min |
| 3 | Curation, standardisation, filtering | ~4 hours (once only) |
| 4 | Target class annotation | < 1 min |
| 4B | Chemical diversity metrics | ~15 min |
| 5 | Molecular featurisation | ~5 hours (once only) |
| 6 | Scaffold split and metric functions | < 1 min |
| 7 | Model training (scaffold split) | 4-8 hours overnight |
| 8 | Publication figures | ~5 min |
| 9 | Statistical tests (Friedman + Wilcoxon + BH) | < 1 min |
| 9B | Performance driver correlations | < 1 min |
| 9C | Endpoint stratification (Ki/Kd subset) | ~20 min |
| 10 | Summary tables | < 1 min |
| 11 | Feature ablation | ~45 min |
| 12 | Scaffold vs random split comparison | ~20 min |
| 13 | Manuscript key statistics | < 1 min |

4. After Section 3 completes, reload the parquet on future runs:
```python
df = pd.read_parquet(DATA_DIR / 'bindingdb_curated.parquet')
```

5. After Section 5 completes, reload feature matrices on future runs:
```python
morgan_matrix   = np.load(DATA_DIR / 'morgan_fps.npy')
desc_matrix     = np.load(DATA_DIR / 'desc_matrix.npy')
combined_matrix = np.load(DATA_DIR / 'combined_fps.npy')
with open(DATA_DIR / 'smiles_to_idx.pkl', 'rb') as f:
    smiles_to_idx = pickle.load(f)
```

---

## Data

BindingDB data are available at https://www.bindingdb.org under a Creative Commons Attribution 3.0 licence. The curated dataset used in this study was downloaded on 28 February 2026.

Pre-computed feature matrices and model results are available on Zenodo: [DOI to be added upon acceptance]

---

## Citation

If you use this code or data, please cite:

> Salihu AS, Rahma MS, Wan Salleh WMNH, Gwaram NS, Yusuf SS. Benchmarking Machine Learning Models for Bioactivity Prediction Across Human Protein Targets Using BindingDB: A Large-Scale Scaffold-Split Evaluation. *J. Chem. Inf. Model.* 2026 (submitted).

---

## Contact

Abubakar Siddiq Salihu
Department of Pure and Industrial Chemistry
Umaru Musa Yar'adua University, Katsina, Nigeria
abusiddiq.salihu@umyu.edu.ng
ORCID: 0000-0002-4425-7524
