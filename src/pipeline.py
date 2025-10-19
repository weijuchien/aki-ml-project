#!/usr/bin/env python3

"""
AKI Prediction Nested CV Runner
Author: Vita
Description: Run nested cross-validation with parallel feature engineering and model tuning
"""

import os
import gc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, brier_score_loss
)
import argparse

# ============================================================
# Constants
# ============================================================
EVALUATION_METRICS = ['precision', 'recall', 'f1', 'roc_auc', 'auprc', 'brier_score']


# ============================================================
# 1️⃣ Feature Engineering
# ============================================================
def feature_engineering(df, observation_hours=6, label_col="aki_label"):
    """
    Feature engineering with STRICT temporal boundaries to prevent leakage.

    Key fixes:
    1. ONLY uses data from [intime, intime + observation_hours)
    2. Does NOT use any information from prediction window
    3. Each stay contributes exactly ONE sample
    """
    df = df.copy()
    df = df.sort_values(["stay_id", "charttime"]).reset_index(drop=True)

    candidate_vitals = [
        "heart_rate", "sbp", "dbp", "meanbp", "resp_rate",
        "temperature", "spo2", "glucose"
    ]
    vitals = [c for c in candidate_vitals if c in df.columns]

    rows, y_rows, groups = [], [], []
    stays_skipped = 0

    for stay_id, g in df.groupby("stay_id", sort=False):
        # Check if stay has label
        if label_col not in g.columns or g[label_col].isna().all():
            stays_skipped += 1
            continue

        intime0 = g["intime"].iloc[0]

        # STRICT OBSERVATION WINDOW: [intime, intime + observation_hours)
        obs_end = intime0 + pd.Timedelta(hours=observation_hours)

        # CRITICAL: Only use data BEFORE obs_end
        gw_obs = g[g["charttime"] < obs_end].copy()

        if gw_obs.empty:
            stays_skipped += 1
            continue

        feat = {
            "stay_id": stay_id,
            "n_obs_rows": len(gw_obs),
        }

        # ----- Creatinine features (ONLY from observation window) -----
        crea_obs = gw_obs[gw_obs["lab_label"] == "Creatinine"].copy()

        if crea_obs.empty:
            stays_skipped += 1
            continue

        # Baseline should already be in the dataframe from create_label_no_leakage
        baseline_val = crea_obs["baseline_creatinine"].iloc[0]

        if baseline_val and not np.isnan(baseline_val) and baseline_val != 0:
            # Use LAST value in observation window (not future!)
            last_crea = crea_obs.sort_values("charttime")["lab_value"].iloc[-1]
            feat["crea_to_base_ratio"] = last_crea / baseline_val
        else:
            feat["crea_to_base_ratio"] = np.nan

        v = crea_obs["lab_value"].values
        t = (crea_obs["charttime"] - crea_obs["charttime"].min()).dt.total_seconds().values / 3600.0

        feat.update({
            "crea_mean": np.nanmean(v),
            "crea_std": np.nanstd(v),
            "crea_min": np.nanmin(v),
            "crea_median": np.nanmedian(v),
            "crea_max": np.nanmax(v),
            "crea_last": crea_obs.sort_values("charttime")["lab_value"].iloc[-1],
            "crea_count": len(crea_obs),
            "crea_slope_per_h": np.polyfit(t, v, 1)[0] if (len(v) >= 2 and np.var(t) > 0) else 0.0,
            "baseline_creatinine": baseline_val
        })

        # ----- Vital signs (ONLY from observation window) -----
        for col in vitals:
            col_obs = gw_obs[col].dropna()
            if len(col_obs) == 0:
                feat.update({
                    f"{col}_mean": np.nan,
                    f"{col}_std": np.nan,
                    f"{col}_min": np.nan,
                    f"{col}_median": np.nan,
                    f"{col}_max": np.nan,
                    f"{col}_last": np.nan
                })
            else:
                vals = col_obs.values
                feat.update({
                    f"{col}_mean": np.nanmean(vals),
                    f"{col}_std": np.nanstd(vals),
                    f"{col}_min": np.nanmin(vals),
                    f"{col}_median": np.nanmedian(vals),
                    f"{col}_max": np.nanmax(vals),
                    f"{col}_last": col_obs.iloc[-1]
                })

        rows.append(feat)
        y_rows.append(int(g[label_col].iloc[0]))
        groups.append(stay_id)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=object)

    X = pd.DataFrame(rows)
    y = pd.Series(y_rows, index=X.index, name=label_col).astype(int)
    groups = pd.Series(groups, index=X.index, name="stay_id")

    # Drop identifier
    X = X.drop(columns=["stay_id"])

    if stays_skipped > 0:
        print(f"[Info] Skipped {stays_skipped} stays with insufficient observation data")

    return X, y, groups


# ============================================================
# 2️⃣ Parallelized feature extraction
# ============================================================
def process_single_stay(stay_id, df, observation_hours, label_col):
    g = df[df["stay_id"] == stay_id]
    X_one, y_one, group_one = feature_engineering(
        g, observation_hours=observation_hours, label_col=label_col
    )

    if X_one.empty:
        return None
    X_one["stay_id"] = stay_id

    y_one = pd.Series(y_one, name=label_col)

    return X_one, y_one, group_one


def parallel_feature_engineering(df, observation_hours=6, label_col="aki_label", n_jobs=4):
    stay_groups = [(stay_id, g.copy()) for stay_id, g in df.groupby("stay_id")]

    # Use joblib for better multiprocessing support
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_single_stay)(stay_id, g, observation_hours, label_col)
        for stay_id, g in stay_groups
    )

    results = [r for r in results if r is not None]
    X_all = pd.concat([r[0] for r in results], ignore_index=True)
    y_all = pd.concat([r[1] for r in results], ignore_index=True)
    # groups_all = pd.concat([r[2] for r in results], ignore_index=True)

    return X_all, y_all  # , groups_all


# ============================================================
# 3️⃣ Cross-validation pipeline
# ============================================================
def process_fold(df_raw, train_stays, test_stays, fold_idx, model_type,
                 inner_splits=3, random_state=42, observation_hours=6, label_col='aki_label'):
    """
    Process a single fold with proper temporal boundaries and no data leakage.

    Key changes:
    1. Filter dataframe by stays (not by rows)
    2. Feature engineering happens AFTER split, using only observation window
    3. Imputation happens WITHIN each fold
    """
    print(f"\n{'='*60}")
    print(f"Processing Fold {fold_idx + 1}")
    print(f"{'='*60}")

    # CRITICAL: Filter dataframe by stays (not by rows)
    train_df = df_raw[df_raw["stay_id"].isin(train_stays)].copy()
    test_df = df_raw[df_raw["stay_id"].isin(test_stays)].copy()

    # Feature engineering (happens AFTER split, using only observation window)
    print("[3/4] Extracting features from observation window...")
    X_train, y_train, groups_train = feature_engineering(
        train_df, observation_hours, label_col
    )
    X_test, y_test, groups_test = feature_engineering(
        test_df, observation_hours, label_col
    )

    if X_train.empty or X_test.empty:
        print(f"[Warning] Fold {fold_idx + 1} has insufficient data, skipping...")
        return None

    # Model setup
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=random_state)
        param_grid = {'model__max_depth': [5, 10, 20]}
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=random_state)
        param_grid = {'model__C': [0.01, 0.1, 1, 10]}
    else:
        raise ValueError(f"Model {model_type} not supported")

    # Pipeline: imputation -> scaling -> SMOTE -> model
    # CRITICAL: Imputation happens HERE, not globally
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=random_state)),
        ('model', model)
    ])

    # Inner CV for hyperparameter tuning
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    print("[4/4] Running inner CV for hyperparameter tuning...")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='f1',
        n_jobs=4,
        verbose=0
    )

    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV F1: {grid.best_score_:.3f}")

    # Evaluate on test set
    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:, 1]

    results = {
        "fold": fold_idx + 1,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "auprc": average_precision_score(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "test_prevalence": y_test.mean()
    }

    print(f"\nFold {fold_idx + 1} Results:")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall: {results['recall']:.3f}")
    print(f"  F1: {results['f1']:.3f}")
    print(f"  ROC-AUC: {results['roc_auc']:.3f}")
    print(f"  AUPRC: {results['auprc']:.3f}")
    print(f"  Brier Score: {results['brier_score']:.3f}")

    # Clean up memory
    del X_train, y_train, X_test, y_test
    gc.collect()

    return results


# ============================================================
# 4️⃣ Nested CV Runner
# ============================================================
def run_nested_cv_with_gridsearch(df_raw, model_type, outer_splits=10, inner_splits=3,
                                  random_state=42, observation_hours=6, label_col="aki_label"):
    """
    Nested CV with proper temporal boundaries and no data leakage.
    Uses the updated process_fold function for cleaner modularity.
    """
    print("[1/4] Preparing stratified splits at stay level...")
    stay_labels = df_raw.groupby('stay_id')[label_col].max()
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(stay_labels.index, stay_labels.values)):
        train_stays = stay_labels.index[train_idx]
        test_stays = stay_labels.index[test_idx]
        folds.append((fold_idx, train_stays, test_stays))

    print(f"[2/4] Starting Nested CV with {outer_splits} folds...")

    # Use threading backend to avoid pickle issues
    with tqdm_joblib(tqdm(desc="Outer Folds Progress", total=len(folds))):
        all_results = Parallel(n_jobs=4, backend="threading")(
            delayed(process_fold)(
                df_raw, train_stays, test_stays, fold_idx,
                model_type, inner_splits, random_state, observation_hours, label_col
            )
            for fold_idx, train_stays, test_stays in folds
        )

    # Filter out None results (failed folds)
    all_results = [r for r in all_results if r is not None]

    print("\n" + "="*60)
    print("FINAL RESULTS (All Folds)")
    print("="*60)

    results_df = pd.DataFrame(all_results)
    for metric in EVALUATION_METRICS:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"{metric.upper()}: {mean_val:.3f} ± {std_val:.3f}")

    return results_df


# ============================================================
# 5️⃣ Main entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nested CV AKI Prediction Pipeline")
    parser.add_argument(
        "-d", "--data", type=str, default="../data/final_cohort.csv",
        help="Path to input CSV file (merged ICU data)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["intime", "outtime", "charttime"])

    # Run both models
    all_results = []

    for model_type in ["lr", "rf"]:
        print(f"\n{'='*80}")
        print(f"Running {model_type.upper()} Model")
        print(f"{'='*80}")

        results_df = run_nested_cv_with_gridsearch(df, model_type)
        results_df["model"] = model_type  # Add model type column
        all_results.append(results_df)

    # Combine results from both models
    combined_results = pd.concat(all_results, ignore_index=True)

    # Create results directory if it doesn't exist
    os.makedirs("../results", exist_ok=True)
    result_file = "nested_cv_results.csv"
    combined_results.to_csv(f"../results/{result_file}", index=False)

    print(f"\n============== Saved results to {result_file} ==============")
    print(f"Total folds: {len(combined_results)} ({len(combined_results[combined_results['model']=='lr'])} LR + {len(combined_results[combined_results['model']=='rf'])} RF)")  # noqa: E501

    # Print summary by model
    print("\n" + "-"*60)
    print("SUMMARY BY MODEL")
    print("-"*60)
    for model in ["lr", "rf"]:
        model_results = combined_results[combined_results["model"] == model]
        print(f"\n{model.upper()} Model:")
        for metric in EVALUATION_METRICS:
            mean_val = model_results[metric].mean()
            std_val = model_results[metric].std()
            print(f"  {metric.upper()}: {mean_val:.3f} ± {std_val:.3f}")
