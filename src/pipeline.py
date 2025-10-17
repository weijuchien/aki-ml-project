#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import multiprocessing

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from scipy.sparse import issparse
import argparse


# ============================================================
# 1️⃣ Feature Engineering
# ============================================================
def feature_engineering(df, observation_hours=6, label_col="aki_label"):
    df = df.copy()
    df = df.sort_values(["stay_id", "charttime"]).reset_index(drop=True)

    candidate_vitals = [
        "heart_rate", "sbp", "dbp", "meanbp", "resp_rate", "temperature", "spo2", "glucose"
    ]
    vitals = [c for c in candidate_vitals if c in df.columns]

    rows, y_rows, groups = [], [], []
    stay_id_skip = 0

    for stay_id, g in df.groupby("stay_id", sort=False):
        if label_col not in g.columns or g[label_col].isna().all():
            stay_id_skip += 1
            continue

        intime0 = g["intime"].iloc[0]
        obs_start = intime0
        obs_end = intime0 + pd.Timedelta(hours=observation_hours)
        gw_obs = g[(g["charttime"] >= obs_start) & (g["charttime"] < obs_end)]

        if gw_obs.empty:
            stay_id_skip += 1
            continue

        feat = {"stay_id": stay_id}

        # --- Creatinine ---
        crea_obs = gw_obs[gw_obs["lab_label"] == "Creatinine"].copy()
        baseline_val = crea_obs["baseline_creatinine"].iloc[0] if not crea_obs.empty else np.nan

        if baseline_val and not np.isnan(baseline_val) and baseline_val != 0:
            feat["crea_to_base_last_ratio"] = crea_obs["lab_value"].iloc[-1] / baseline_val
        else:
            feat["crea_to_base_last_ratio"] = np.nan

        if not crea_obs.empty:
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
                "crea_slope_per_h": np.polyfit(t, v, 1)[0] if (len(v) >= 2 and np.nanvar(t) > 0) else 0.0,
            })
        else:
            stay_id_skip += 1
            continue

        # --- Vitals ---
        for col in vitals:
            col_obs = gw_obs[col].dropna()
            if len(col_obs) == 0:
                feat.update({f"{col}_mean": np.nan, f"{col}_std": np.nan, f"{col}_min": np.nan,
                             f"{col}_median": np.nan, f"{col}_max": np.nan, f"{col}_last": np.nan})
            else:
                vals = col_obs.values
                feat[f"{col}_mean"] = np.nanmean(vals)
                feat[f"{col}_std"] = np.nanstd(vals)
                feat[f"{col}_min"] = np.nanmin(vals)
                feat[f"{col}_median"] = np.nanmedian(vals)
                feat[f"{col}_max"] = np.nanmax(vals)
                feat[f"{col}_last"] = col_obs.iloc[-1]

        rows.append(feat)
        y_rows.append(int(g[label_col].iloc[0]))
        groups.append(stay_id)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=object)

    X = pd.DataFrame(rows)
    y = pd.Series(y_rows, index=X.index, name=label_col).astype(int)
    groups = pd.Series(groups, index=X.index, name="stay_id")
    X = X.drop(columns=["stay_id"])

    for c in X.columns:
        if X[c].dtype == "O":
            X[c] = pd.to_numeric(X[c], errors="ignore")

    if stay_id_skip > 0:
        print(f"[Info] Skip {stay_id_skip} stays with missing or invalid data")

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


def parallel_feature_engineering(df, observation_hours=6, label_col="aki_label", n_jobs=-1):
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
    print(f"\n============= Processing Fold {fold_idx + 1} =============")
    outer_train = df_raw[df_raw["stay_id"].isin(train_stays) & df_raw["aki_label"].notna()].copy()
    outer_test = df_raw[df_raw["stay_id"].isin(test_stays) & df_raw["aki_label"].notna()].copy()

    X_train, y_train = parallel_feature_engineering(outer_train, observation_hours, label_col, n_jobs=4)
    X_test, y_test = parallel_feature_engineering(outer_test, observation_hours, label_col, n_jobs=4)

    if model_type == 'rf':
        model = RandomForestClassifier(random_state=random_state)
        search_param_grid = {'model__max_depth': [5, 10, 20]}
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=random_state)
        search_param_grid = {'model__C': [0.01, 0.1, 1, 10]}
    else:
        raise ValueError(f"{model_type} is not supported.")

    scaler = StandardScaler(with_mean=False) if issparse(X_train.values) else StandardScaler()

    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler),
        ('smote', SMOTE(random_state=random_state)),
        ('model', model)
    ])

    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=search_param_grid,
        cv=inner_cv,
        scoring='f1',
        n_jobs=4
    )
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best F1 score:", grid.best_score_)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    results = {
        "fold": fold_idx + 1,
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "auprc": average_precision_score(y_test, y_prob)
    }

    del X_train, y_train, X_test, y_test
    gc.collect()

    return results


# ============================================================
# 4️⃣ Nested CV Runner
# ============================================================
def run_nested_cv_with_gridsearch(df_raw, model_type, outer_splits=10, inner_splits=3,
                                  random_state=42, observation_hours=6, label_col="aki_label"):
    stay_labels = df_raw.groupby('stay_id')[label_col].max()
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(stay_labels.index, stay_labels.values)):
        train_stays = stay_labels.index[train_idx]
        test_stays = stay_labels.index[test_idx]
        folds.append((fold_idx, train_stays, test_stays))

    print(f"\n[Info] Starting Nested CV with {outer_splits} folds...")

    ctx = multiprocessing.get_context("spawn")
    with tqdm_joblib(tqdm(desc="Outer Folds Progress", total=len(folds))) as progress_bar:
        with Parallel(n_jobs=4, backend="loky", context=ctx) as parallel:
            all_results = parallel(
                delayed(process_fold)(
                    df_raw, train_stays, test_stays, fold_idx,
                    model_type, inner_splits, random_state, observation_hours, label_col
                )
                for fold_idx, train_stays, test_stays in folds
            )

    print("\n############## Cross-validation complete! ##############")
    for r in all_results:
        print(f"Fold {r['fold']}: F1={r['f1']:.3f}, ROC={r['roc_auc']:.3f}, AUPRC={r['auprc']:.3f}")

    return pd.DataFrame(all_results)


# ============================================================
# 5️⃣ Main entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nested CV AKI Prediction Pipeline")
    parser.add_argument("-d", "--data", type=str, default="../data/final_cohort.csv", help="Path to input CSV file (merged ICU data)")
    parser.add_argument("-m", "--model", type=str, choices=["lr", "rf"], default="lr", help="Model type")
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["intime", "outtime", "charttime"]).head(10000)
    results_df = run_nested_cv_with_gridsearch(df, args.model)

    # Create results directory if it doesn't exist
    os.makedirs("../results", exist_ok=True)
    results_df.to_csv("../results/nested_cv_results.csv", index=False)

    print("\n✅ Saved results to nested_cv_results.csv")
