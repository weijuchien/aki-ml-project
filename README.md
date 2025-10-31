## AKI ML Project - Quick Start

### Overview
This repository trains and analyzes AKI prediction models (Logistic Regression and Random Forest).
Typical workflow:
- Prepare the cohort in Google Colab using `src/fetch_data.ipynb` (exports `final_cohort.csv`).
- Run the local training/evaluation pipeline with `src/pipeline.py`.
- Optionally run SHAP analysis or generate plots from saved summaries.

### 1) Prepare data in Google Colab
1. Open `src/fetch_data.ipynb` in Google Colab.
2. Run all cells. The notebook authenticates to BigQuery, builds labels, and merges vitals/labs.
3. When finished, it downloads `final_cohort.csv` to your machine.
4. Move it into the project data folder:
```bash
mv ~/Downloads/final_cohort.csv data/final_cohort.csv
```

Note: Ensure the filename is exactly `final_cohort.csv`.

### 2) Local environment setup
Install dependencies (minimal example):
```bash
pip install -r requirements.txt
```

### 3) Train and evaluate
Run nested CV for both Logistic regression and Random forest models using the prepared cohort:
```bash
python src/pipeline.py -d data/final_cohort.csv
```
Outputs:
- `results/nested_cv_results.csv`
- `results/lr_best_params.json` and `results/rf_best_params.json`

### 4) Optional: SHAP analysis
Use saved best parameters to generate SHAP plots:
```bash
python src/shap_analysis.py -i data/final_cohort.csv \
  --lr-params results/lr_best_params.json \
  --rf-params results/rf_best_params.json \
  -o results/shap_results
```
Plots and tables will be saved under `results/shap_results/{lr,rf}/`.

### 5) Optional: Plot feature importance from summary (no retraining)
If `results/shap_results/feature_importance_summary.csv` exists, render per-model plots directly:
```bash
# Random Forest
python src/plot_feature_importance.py --model rf
# Logistic Regression
python src/plot_feature_importance.py --model lr
```
Saved to:
- `results/shap_results/rf/feature_importance.png`
- `results/shap_results/lr/feature_importance.png`
