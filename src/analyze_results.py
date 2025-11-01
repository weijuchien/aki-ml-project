#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import argparse
from pathlib import Path

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ============================================================
# Constants
# ============================================================
EVALUATION_METRICS = ['precision', 'recall', 'f1', 'roc_auc', 'auprc', 'brier_score']
MODEL_TYPES = ['lr', 'rf']
MODEL_NAMES = {'lr': 'Logistic Regression', 'rf': 'Random Forest'}


# ============================================================
# 1. Statistical Analysis Functions
# ============================================================
def load_results(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    return df


def calculate_summary_statistics(df):
    """Calculate comprehensive summary statistics for each model."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    summary_stats = []

    for model in MODEL_TYPES:
        model_data = df[df['model'] == model]
        if model_data.empty:
            continue

        print(f"\n{MODEL_NAMES[model]} ({model.upper()}):")
        print("-" * 50)

        model_stats = {'model': model}

        for metric in EVALUATION_METRICS:
            if metric in model_data.columns:
                values = model_data[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    median_val = values.median()
                    q25 = values.quantile(0.25)
                    q75 = values.quantile(0.75)
                    min_val = values.min()
                    max_val = values.max()

                    model_stats.update({
                        f'{metric}_mean': mean_val,
                        f'{metric}_std': std_val,
                        f'{metric}_median': median_val,
                        f'{metric}_q25': q25,
                        f'{metric}_q75': q75,
                        f'{metric}_min': min_val,
                        f'{metric}_max': max_val,
                        f'{metric}_count': len(values)
                    })

                    print(f"  {metric.upper():12}: {mean_val:.3f} ± {std_val:.3f} "
                          f"(median: {median_val:.3f}, range: [{min_val:.3f}, {max_val:.3f}])")

        summary_stats.append(model_stats)

    return pd.DataFrame(summary_stats)


def perform_statistical_tests(df):
    """Perform statistical tests to compare models."""
    print("\n" + "="*80)
    print("STATISTICAL TESTS")
    print("="*80)

    # Check if we have both models
    lr_data = df[df['model'] == 'lr']
    rf_data = df[df['model'] == 'rf']

    if lr_data.empty or rf_data.empty:
        print("Warning: Cannot perform statistical tests - missing model data")
        return None

    # Ensure same number of folds for paired tests
    min_folds = min(len(lr_data), len(rf_data))
    lr_subset = lr_data.head(min_folds)
    rf_subset = rf_data.head(min_folds)

    test_results = []

    for metric in EVALUATION_METRICS:
        if metric not in df.columns:
            continue

        lr_values = lr_subset[metric].dropna()
        rf_values = rf_subset[metric].dropna()

        if len(lr_values) < 2 or len(rf_values) < 2:
            continue

        # Paired t-test
        try:
            t_stat, t_pvalue = ttest_rel(lr_values, rf_values)
        except Exception:
            t_stat, t_pvalue = np.nan, np.nan

        # Effect size (Cohen's d)
        try:
            pooled_std = np.sqrt(((len(lr_values) - 1) * lr_values.std()**2 +
                                  (len(rf_values) - 1) * rf_values.std()**2) /
                                 (len(lr_values) + len(rf_values) - 2))
            cohens_d = (lr_values.mean() - rf_values.mean()) / pooled_std
        except Exception:
            cohens_d = np.nan

        test_results.append({
            'metric': metric,
            'lr_mean': lr_values.mean(),
            'rf_mean': rf_values.mean(),
            'mean_difference': lr_values.mean() - rf_values.mean(),
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'cohens_d': cohens_d
        })

        print(f"\n{metric.upper()}:")
        print(f"  LR mean: {lr_values.mean():.3f}, RF mean: {rf_values.mean():.3f}")
        print(f"  Difference: {lr_values.mean() - rf_values.mean():.3f}")
        print(f"  Paired t-test: t={t_stat:.3f}, p={t_pvalue:.3f}")
        print(f"  Effect size (Cohen's d): {cohens_d:.3f}")

        # Interpret significance
        if t_pvalue < 0.05:
            significance = ("Significant" if abs(cohens_d) > 0.2
                            else "Significant but small effect")
        else:
            significance = "Not significant"
        print(f"  Interpretation: {significance}")

    return pd.DataFrame(test_results)


# ============================================================
# 2. Visualization Functions
# ============================================================
def create_metric_comparison_plot(df, save_path=None):
    """Create box plots comparing metrics between models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # fig.suptitle('Model Performance Comparison Across Metrics',
    #              fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for i, metric in enumerate(EVALUATION_METRICS):
        if metric not in df.columns:
            continue

        ax = axes[i]

        plot_data = []
        labels = []

        for model in MODEL_TYPES:
            model_data = df[df['model'] == model][metric].dropna()
            if len(model_data) > 0:
                plot_data.append(model_data)
                labels.append(MODEL_NAMES[model])

        if plot_data:
            bp = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True)

            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)

            ax.set_title(f'{metric.upper()}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)

            for j, data in enumerate(plot_data):
                mean_val = data.mean()
                ax.plot(j+1, mean_val, 'D', color='red', markersize=8,
                        markeredgecolor='black')

    fig.text(0.02, 0.02,
             'Note: Red diamonds (♦) = Mean values\n'
             'Black circles (●) = Outliers (beyond 1.5×IQR)',
             fontsize=10, ha='left', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, wspace=0.4)  # wspace controls horizontal spacing between columns

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metric comparison plot to: {save_path}")

    plt.show()


def create_performance_heatmap(df, save_path=None):
    """Create a heatmap showing performance metrics."""
    # Prepare data for heatmap
    heatmap_data = []

    for model in MODEL_TYPES:
        model_data = df[df['model'] == model]
        if not model_data.empty:
            row = []
            for metric in EVALUATION_METRICS:
                if metric in model_data.columns:
                    mean_val = model_data[metric].dropna().mean()
                    row.append(mean_val)
                else:
                    row.append(np.nan)
            heatmap_data.append(row)

    if heatmap_data:
        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=[MODEL_NAMES[m] for m in MODEL_TYPES
                   if not df[df['model'] == m].empty],
            columns=[m.upper() for m in EVALUATION_METRICS]
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlBu_r',
                    cbar_kws={'label': 'Score (Note: Lower Brier Score = Better)'})
        # plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Models')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved performance heatmap to: {save_path}")

        plt.show()


def create_fold_progression_plot(df, save_path=None):
    """Create line plots showing performance across folds."""
    # Define color mapping to match metric_comparison.png
    model_colors = {
        'lr': 'lightblue',      # Light blue for Logistic Regression
        'rf': 'lightcoral'       # Light coral for Random Forest
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # fig.suptitle('Performance Across Folds', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for i, metric in enumerate(EVALUATION_METRICS):
        if metric not in df.columns:
            continue

        ax = axes[i]

        for model in MODEL_TYPES:
            model_data = df[df['model'] == model].sort_values('fold')
            if not model_data.empty:
                color = model_colors.get(model, 'gray')
                ax.plot(model_data['fold'], model_data[metric],
                        marker='o', linewidth=2, color=color,
                        label=MODEL_NAMES[model], markersize=6)

        ax.set_title(f'{metric.upper()}', fontweight='bold')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)  # Increase horizontal spacing between columns

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved fold progression plot to: {save_path}")

    plt.show()


def create_correlation_matrix(df, save_path=None):
    """Create correlation matrix between metrics."""

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_data = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_data, dtype=bool))
    sns.heatmap(correlation_data, mask=mask, annot=True, fmt='.3f',
                cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix of Metrics and Features',
              fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation matrix to: {save_path}")

    plt.show()


# ============================================================
# 3. Main Analysis Function
# ============================================================
def run_complete_analysis(csv_path, output_dir=None):
    """Run complete analysis including statistics and visualizations."""
    print("Starting AKI Prediction Results Analysis")
    print("="*80)

    # Create output directory
    if output_dir is None:
        output_dir = Path(csv_path).parent / "analysis_results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    df = load_results(csv_path)

    # Statistical analysis
    summary_stats = calculate_summary_statistics(df)
    test_results = perform_statistical_tests(df)

    # Save statistical results
    summary_stats.to_csv(output_dir / "summary_statistics.csv", index=False)
    if test_results is not None:
        test_results.to_csv(output_dir / "statistical_tests.csv", index=False)

    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    create_metric_comparison_plot(df, output_dir / "metric_comparison.png")
    create_performance_heatmap(df, output_dir / "performance_heatmap.png")
    create_fold_progression_plot(df, output_dir / "fold_progression.png")
    # create_correlation_matrix(df, output_dir / "correlation_matrix.png")

    print(f"\nAnalysis complete! Results saved to: {output_dir}")

    return summary_stats, test_results


# ============================================================
# 4. Main Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze AKI Prediction Results")
    parser.add_argument(
        "-i", "--input", type=str,
        default="../results/nested_cv_results.csv",
        help="Path to input CSV file with results"
    )
    parser.add_argument(
        "-o", "--output", type=str,
        default=None,
        help="Output directory for analysis results (default: analysis_results/)"
    )

    args = parser.parse_args()

    try:
        summary_stats, test_results = run_complete_analysis(args.input, args.output)
        print("\nAnalysis completed successfully!")

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
