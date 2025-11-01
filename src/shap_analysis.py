#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pipeline import parallel_feature_engineering


# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ============================================================
# Constants
# ============================================================
MODEL_NAMES = {'lr': 'Logistic Regression', 'rf': 'Random Forest'}


# ============================================================
# 1. Data Loading and Model Loading
# ============================================================
def load_best_parameters(params_path):
    """Load best parameters from pipeline results."""
    print(f"Loading best parameters from {params_path}...")

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    with open(params_path, 'r') as f:
        params_info = json.load(f)

    print("Parameters loaded successfully!")
    print(f"Model type: {params_info['model_type']}")
    print(f"Best parameters: {params_info['best_parameters']}")

    return params_info


def train_model_with_params(params_info, X_train, y_train, random_state=42):
    """Train model using best parameters from nested CV."""
    print("Training model with best parameters...")

    model_type = params_info['model_type']
    best_params = params_info['best_parameters']

    # Model setup using best parameters
    if model_type == 'rf':
        model = RandomForestClassifier(
            max_depth=best_params.get('model__max_depth'),
            random_state=random_state
        )
    elif model_type == 'lr':
        model = LogisticRegression(
            C=best_params.get('model__C'),
            max_iter=1000,
            solver='liblinear',
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Pipeline: imputation -> scaling -> SMOTE -> model
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=random_state)),
        ('model', model)
    ])

    # Train model
    pipeline.fit(X_train, y_train)
    print(f"Model trained successfully with parameters: {best_params}")

    return pipeline


def prepare_data_for_shap(df, test_size=0.2, random_state=42):
    """
    Prepare data for SHAP analysis with proper train/test split.

    This creates a clean train/test split for model training and SHAP analysis.
    """
    print("Preparing data for SHAP analysis...")
    print("Using proper train/test split for model training and SHAP analysis")

    # Split data
    X, y = parallel_feature_engineering(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Data prepared: {len(X_train)} train samples, {len(X_test)} test samples")
    print("No data leakage - clean train/test split")

    return X_train, X_test, y_train, y_test


# ============================================================
# 2. SHAP Analysis Functions
# ============================================================
def calculate_shap_values(pipeline, X_train, X_test, model_type):
    """Calculate SHAP values for the model."""
    print("Calculating SHAP values...")

    # Get the actual model from pipeline
    model = pipeline.named_steps['model']
    print(f"Extracted model type: {type(model)}")

    # For SHAP analysis, we need to transform the data through the pipeline
    # but stop before the model step
    X_train_transformed = pipeline.named_steps['scaler'].transform(
        pipeline.named_steps['imputer'].transform(X_train)
    )
    X_test_transformed = pipeline.named_steps['scaler'].transform(
        pipeline.named_steps['imputer'].transform(X_test)
    )

    print(f"Transformed data shapes - Train: {X_train_transformed.shape}, Test: {X_test_transformed.shape}")

    # Create SHAP explainer based on model type
    if model_type == 'rf':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_transformed)
        # For binary classification, get positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    elif model_type == 'lr':
        explainer = shap.LinearExplainer(model, X_train_transformed)
        shap_values = explainer.shap_values(X_test_transformed)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Sample training data for background
    X_train_sample = X_train_transformed[:100] if len(X_train_transformed) > 100 else X_train_transformed

    print(f"SHAP values calculated: {shap_values.shape}")
    print(f"SHAP values range: {np.min(shap_values):.4f} to {np.max(shap_values):.4f}")
    print(f"Expected value: {explainer.expected_value}")

    return explainer, shap_values, X_train_sample, X_test_transformed


def create_feature_importance_plot(shap_values, feature_names, model_name, save_path=None):
    """Create feature importance plot based on SHAP values."""
    print("Creating feature importance plot...")

    # Handle binary classification SHAP values (3D array)
    if len(shap_values.shape) == 3:
        # For binary classification, use positive class (index 1)
        shap_values = shap_values[:, :, 1]
        print(f"Using positive class SHAP values, shape: {shap_values.shape}")

    # Calculate mean absolute SHAP values
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    print(f"Mean SHAP values shape: {mean_shap_values.shape}")
    print(f"Feature names length: {len(feature_names)}")

    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap_values
    }).sort_values('importance', ascending=True)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Mean |SHAP value|')
    # plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to: {save_path}")

    plt.show()

    return importance_df


def create_shap_summary_plot(shap_values, X_test_sample, feature_names, model_name, save_path=None):
    """Create SHAP summary plot."""
    print("Creating SHAP summary plot...")

    # Handle binary classification SHAP values (3D array)
    if len(shap_values.shape) == 3:
        # For binary classification, use positive class (index 1)
        shap_values = shap_values[:, :, 1]
        print(f"Using positive class SHAP values for summary plot, shape: {shap_values.shape}")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot - {model_name}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved SHAP summary plot to: {save_path}")

    plt.show()


def create_shap_waterfall_plot(explainer, X_test_sample, feature_names, model_name, save_path=None):
    """Create SHAP waterfall plot for a single prediction."""
    print("Creating SHAP waterfall plot...")

    try:
        # Convert to pandas DataFrame for easier handling
        X_test_df = pd.DataFrame(X_test_sample, columns=feature_names)
        sample_data = X_test_df.iloc[0]

        # Get SHAP values for first sample
        shap_values_single = explainer.shap_values(sample_data)

        # Debug: print SHAP values info
        print(f"SHAP values type: {type(shap_values_single)}")
        print(f"SHAP values shape: {np.array(shap_values_single).shape}")

        # Handle different SHAP value formats
        if isinstance(shap_values_single, list) and len(shap_values_single) == 2:
            # Binary classification: use positive class (index 1)
            shap_values_single = shap_values_single[1]
            print("Using positive class SHAP values for binary classification")
        elif isinstance(shap_values_single, np.ndarray) and len(shap_values_single.shape) == 2:
            # 2D array for binary classification: use positive class (index 1)
            shap_values_single = shap_values_single[:, 1]
            print("Using positive class SHAP values from 2D array")

        # Ensure we have a numpy array
        shap_values_single = np.array(shap_values_single)

        print(f"Final SHAP values shape: {shap_values_single.shape}")
        print(f"SHAP values range: {np.min(shap_values_single):.4f} to {np.max(shap_values_single):.4f}")

        # Handle expected_value for binary classification
        if isinstance(explainer.expected_value, np.ndarray):
            expected_value = explainer.expected_value[1]  # Use positive class
        else:
            expected_value = explainer.expected_value
        print(f"Expected value: {expected_value}")

        # Check if SHAP values are all zeros or very small
        if np.allclose(shap_values_single, 0, atol=1e-10):
            print("Warning: All SHAP values are close to zero!")
            return

        # Create Explanation object for waterfall plot
        explanation = shap.Explanation(
            values=shap_values_single,
            base_values=expected_value,
            data=sample_data.values,
            feature_names=feature_names
        )

        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - {model_name} (Sample 1)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved SHAP waterfall plot to: {save_path}")

        plt.show()

    except Exception as e:
        print(f"Error creating waterfall plot: {e}")
        print("Trying alternative waterfall plot method...")

        # Alternative method: use shap.plots.waterfall
        try:
            # Handle numpy array input for alternative method
            if isinstance(X_test_sample, np.ndarray):
                sample_data = X_test_sample[0]  # First row
            else:
                sample_data = X_test_sample.iloc[0]

            # Handle expected_value for alternative method
            if isinstance(explainer.expected_value, np.ndarray):
                expected_value = explainer.expected_value[1]  # Use positive class
            else:
                expected_value = explainer.expected_value

            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(expected_value, shap_values_single, sample_data)
            plt.title(f'SHAP Waterfall Plot - {model_name} (Sample 1)', fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved SHAP waterfall plot to: {save_path}")

            plt.show()
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            print("Skipping waterfall plot creation")


# ============================================================
# 3. Main SHAP Analysis Function
# ============================================================
def run_shap_analysis(csv_path, params_paths, output_dir=None):
    """
    Run complete SHAP analysis using best parameters from nested CV.

    This approach avoids data leakage by:
    1. Loading best parameters from nested CV
    2. Training model on clean train/test split
    3. Using test set for SHAP analysis
    """
    print("Starting SHAP Analysis for AKI Prediction Models")
    print("="*80)
    print("Using best parameters from nested CV")
    print("="*80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load and preprocess data
    df = pd.read_csv(csv_path, parse_dates=["intime", "outtime", "charttime"])
    all_results = []

    for model_type, params_path in params_paths.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {MODEL_NAMES[model_type]} ({model_type.upper()})")
        print(f"{'='*60}")

        # Load best parameters
        params_info = load_best_parameters(params_path)

        # Prepare data for SHAP analysis
        X_train, X_test, y_train, y_test = prepare_data_for_shap(df)

        # Train model with best parameters
        pipeline = train_model_with_params(params_info, X_train, y_train)

        # Calculate SHAP values
        explainer, shap_values, X_train_sample, X_test_transformed = calculate_shap_values(
            pipeline, X_train, X_test, model_type
        )

        # Create visualizations
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(exist_ok=True)

        # Feature importance plot
        importance_df = create_feature_importance_plot(
            shap_values, X_train.columns, MODEL_NAMES[model_type],
            model_output_dir / "feature_importance.png"
        )

        # SHAP summary plot
        # create_shap_summary_plot(
        #     shap_values, X_test_transformed, X_test.columns, MODEL_NAMES[model_type],
        #     model_output_dir / "shap_summary.png"
        # )

        # SHAP waterfall plot
        # create_shap_waterfall_plot(
        #     explainer, X_test_transformed, X_test.columns, MODEL_NAMES[model_type],
        #     model_output_dir / "shap_waterfall.png"
        # )
        exit(0)
        # Save feature importance results
        importance_df['model'] = model_type
        all_results.append(importance_df)

        print(f"SHAP analysis completed for {MODEL_NAMES[model_type]}")

    # Combine and save all results
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(output_dir / "feature_importance_summary.csv", index=False)

    print(f"\nSHAP analysis complete! Results saved to: {output_dir}")

    return combined_results


# ============================================================
# 4. Main Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP Analysis for AKI Prediction Models")
    parser.add_argument(
        "-i", "--input", type=str,
        default="../data/final_cohort.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "-o", "--output", type=str,
        default="../results/shap_results",
        help="Output directory for SHAP results (default: shap_results/)"
    )
    parser.add_argument(
        "--lr-params", type=str,
        default="../results/lr_best_params.json",
        help="Path to Logistic Regression best parameters"
    )
    parser.add_argument(
        "--rf-params", type=str,
        default="../results/rf_best_params.json",
        help="Path to Random Forest best parameters"
    )

    args = parser.parse_args()

    # Prepare parameter paths
    params_paths = {}
    if os.path.exists(args.lr_params):
        params_paths['lr'] = args.lr_params

    if os.path.exists(args.rf_params):
        params_paths['rf'] = args.rf_params

    if not params_paths:
        print("Error: No parameter files found!")
        print("Please run the pipeline first to generate parameter files, or specify correct parameter paths.")
        exit(1)

    try:
        results = run_shap_analysis(args.input, params_paths, args.output)
        print("\nSHAP analysis completed successfully!")

        # Print top features for each model
        print("\n" + "="*80)
        print("TOP FEATURES BY MODEL")
        print("="*80)

        for model_type in params_paths.keys():
            model_results = results[results['model'] == model_type].head(10)
            print(f"\n{MODEL_NAMES[model_type]} - Top 10 Features:")
            for idx, row in model_results.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
        raise
