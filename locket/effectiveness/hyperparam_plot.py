import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors

from locket.config import PROJECT_DIR
from locket.utils.logger import logger

# Define color scheme for different metrics
METRIC_COLORS = {
    "mmlu_accuracy": "#1f77b4",  # Blue
    "sql_accuracy": "#ff7f0e",  # Orange
    "samsum_accuracy": "#2ca02c",  # Green
    "math_accuracy": "#d62728",  # Red
}

# Define color scheme for refusal rates
REFUSAL_COLORS = {
    "mmlu_refusal_rate": "#9467bd",  # Purple
    "sql_refusal_rate": "#8c564b",  # Brown
    "samsum_refusal_rate": "#e377c2",  # Pink
    "math_refusal_rate": "#7f7f7f",  # Gray
}

# Define label mapping for cleaner plot legends
METRIC_LABELS = {
    "mmlu_accuracy": "MMLU",
    "sql_accuracy": "SQL",
    "samsum_accuracy": "SAMSUM",
    "math_accuracy": "MATH",
    "mmlu_refusal_rate": "MMLU Refusal",
    "sql_refusal_rate": "SQL Refusal",
    "samsum_refusal_rate": "SAMSUM Refusal",
    "math_refusal_rate": "MATH Refusal",
}


def load_hyperparameter_results(file_path: Optional[str] = None) -> List[Dict]:
    """Load hyperparameter sweep results from JSON file."""
    if file_path is None:
        file_path = Path(PROJECT_DIR) / "logs" / "hyperparameter_sweep_results.json"
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} hyperparameter configurations from {file_path}")
    return data


def plot_hyperparameter_sweep(
    results: List[Dict],
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
) -> None:
    """Create one figure per metric per model with accuracy and refusal lines."""
    if output_dir is None:
        output_dir = Path(PROJECT_DIR) / "logs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Get unique models
    models = df["model"].unique()

    for model in models:
        model_df = df[df["model"] == model]

        # Determine if this model uses single_scale or merging_tau
        if model_df["single_scale"].notna().any():
            param_name = "single_scale"
            param_values = model_df["single_scale"].dropna().unique()
            param_label = "Single Scale"
        elif model_df["merging_tau"].notna().any():
            param_name = "merging_tau"
            param_values = model_df["merging_tau"].dropna().unique()
            param_label = "Merging Tau (τ)"
        else:
            logger.warning(f"No hyperparameter found for model {model}")
            continue

        # Sort parameter values
        param_values = sorted(param_values)

        # For each metric, plot accuracy and refusal (if present)
        for accuracy_key, acc_color in METRIC_COLORS.items():
            if accuracy_key not in model_df.columns:
                continue

            refusal_key = accuracy_key.replace("_accuracy", "_refusal_rate")
            has_refusal = (
                refusal_key in model_df.columns and model_df[refusal_key].notna().any()
            )

            # Collect series
            accuracy_values = []
            refusal_values = [] if has_refusal else None
            for param_val in param_values:
                row = model_df[model_df[param_name] == param_val]
                if not row.empty and accuracy_key in row.columns:
                    accuracy_values.append(row[accuracy_key].iloc[0])
                else:
                    accuracy_values.append(None)

                if has_refusal:
                    if not row.empty and refusal_key in row.columns:
                        refusal_values.append(row[refusal_key].iloc[0])
                    else:
                        refusal_values.append(None)

            # Filter valid points for accuracy
            acc_idx = [i for i, v in enumerate(accuracy_values) if v is not None]
            if not acc_idx:
                continue
            x_acc = [param_values[i] for i in acc_idx]
            y_acc = [accuracy_values[i] for i in acc_idx]

            # Prepare refusal filtered series
            if has_refusal:
                ref_idx = [i for i, v in enumerate(refusal_values) if v is not None]
                x_ref = [param_values[i] for i in ref_idx]
                y_ref = [refusal_values[i] for i in ref_idx]

            # Plot single figure for this metric
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(
                x_acc,
                y_acc,
                color=acc_color,
                marker="o",
                markersize=6,
                linewidth=2,
                label=f"{METRIC_LABELS[accuracy_key]} Accuracy",
                alpha=0.85,
            )

            if has_refusal and y_ref:
                ax.plot(
                    x_ref,
                    y_ref,
                    color=REFUSAL_COLORS.get(refusal_key, "#7f7f7f"),
                    marker="s",
                    markersize=6,
                    linewidth=2,
                    label=f"{METRIC_LABELS[refusal_key]}",
                    alpha=0.85,
                    linestyle="--",
                )

            # Labels and formatting
            ax.set_xlabel(param_label, fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title(
                f"Model: {model}\nMetric: {METRIC_LABELS[accuracy_key]}",
                fontsize=14,
            )
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(loc="best", fontsize=10)

            # Percentage formatting if data is in [0,1]
            if all(v <= 1.0 for v in y_acc if v is not None):
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

            plt.tight_layout()

            # Save plot
            if save_plots:
                safe_model_name = model.replace("/", "_").replace(":", "")
                metric_short = METRIC_LABELS[accuracy_key].lower()
                output_path = (
                    output_dir / f"hyperparam_{safe_model_name}_{metric_short}.png"
                )
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved plot to {output_path}")

            plt.show()


def _lighten_color(color: str | tuple[float, float, float], amount: float = 0.5):
    """Lighten the given color by blending it towards white.

    amount in [0, 1]: 0 returns original color; 1 returns white.
    """
    r, g, b = mcolors.to_rgb(color)
    return (
        r + (1.0 - r) * amount,
        g + (1.0 - g) * amount,
        b + (1.0 - b) * amount,
    )


def plot_features_all_models(
    results: List[Dict],
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
) -> None:
    """For each feature (mmlu/sql/samsum/math), plot all model lines on one figure.

    Each model contributes two lines (if available): accuracy (solid) and refusal (dashed),
    sharing the same base color, with refusal being a lighter shade.
    """
    if output_dir is None:
        output_dir = Path(PROJECT_DIR) / "logs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("No results to plot.")
        return

    models = list(df["model"].unique())
    cmap = plt.get_cmap("tab20", max(1, len(models)))
    model_to_color = {m: cmap(i % cmap.N) for i, m in enumerate(models)}

    features = ["mmlu", "sql", "samsum", "math"]

    for feature in features:
        accuracy_key = f"{feature}_accuracy"
        refusal_key = f"{feature}_refusal_rate"

        if accuracy_key not in df.columns:
            logger.info(f"Skipping feature '{feature}' (no accuracy column present)")
            continue

        fig, ax = plt.subplots(figsize=(12, 7))

        any_values = False

        for model in models:
            model_df = df[df["model"] == model]

            # Determine parameter axis for this model
            if model_df["single_scale"].notna().any():
                param_name = "single_scale"
            elif model_df["merging_tau"].notna().any():
                param_name = "merging_tau"
            else:
                continue

            # Sorted x values
            x_vals = sorted(model_df[param_name].dropna().unique())
            if not x_vals:
                continue

            # Gather y series for accuracy/refusal
            y_acc = []
            y_ref = [] if refusal_key in model_df.columns else None
            for x in x_vals:
                row = model_df[model_df[param_name] == x]
                if not row.empty and accuracy_key in row.columns:
                    y_acc.append(row[accuracy_key].iloc[0])
                else:
                    y_acc.append(None)

                if y_ref is not None:
                    if refusal_key in row.columns:
                        y_ref.append(row[refusal_key].iloc[0])
                    else:
                        y_ref.append(None)

            # Filter valid points
            idx_acc = [i for i, v in enumerate(y_acc) if v is not None]
            if not idx_acc:
                continue

            any_values = True
            x_plot = [x_vals[i] for i in idx_acc]
            y_plot_acc = [y_acc[i] for i in idx_acc]

            base_color = model_to_color[model]
            short_model = model.split("/")[-1]

            ax.plot(
                x_plot,
                y_plot_acc,
                color=base_color,
                marker="o",
                markersize=6,
                linewidth=2,
                label=f"{short_model} Accuracy",
                alpha=0.9,
            )

            # Refusal line
            if y_ref is not None:
                idx_ref = [i for i, v in enumerate(y_ref) if v is not None]
                if idx_ref:
                    x_ref = [x_vals[i] for i in idx_ref]
                    y_plot_ref = [y_ref[i] for i in idx_ref]
                    light_color = _lighten_color(base_color, amount=0.5)
                    ax.plot(
                        x_ref,
                        y_plot_ref,
                        color=light_color,
                        marker="s",
                        markersize=6,
                        linewidth=2,
                        label=f"{short_model} Refusal",
                        alpha=0.9,
                        linestyle="--",
                    )

        # Figure formatting
        ax.set_xlabel("Scale/Tau", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        metric_title = METRIC_LABELS.get(f"{feature}_accuracy", feature.upper())
        ax.set_title(f"Feature: {metric_title}", fontsize=14)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9, ncol=2)

        # Percent formatter if all y in [0,1]
        if any_values:
            # Collect all y values from lines
            y_all = []
            for line in ax.get_lines():
                y_all.extend(line.get_ydata())
            if y_all and all((v is not None) and (v <= 1.0) for v in y_all):
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

        plt.tight_layout()

        if save_plots:
            safe_metric = metric_title.lower().replace(" ", "_")
            output_path = output_dir / f"hyperparam_feature_{safe_metric}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved feature plot to {output_path}")

        plt.show()


def create_comparison_plot(
    results: List[Dict],
    output_dir: Optional[Path] = None,
    save_plot: bool = True,
) -> None:
    """Deprecated in favor of per-metric figures. Kept as no-op for compatibility."""
    logger.info(
        "create_comparison_plot is deprecated; plots are generated per metric per model."
    )


def create_refusal_plot(
    results: List[Dict],
    output_dir: Optional[Path] = None,
    save_plot: bool = True,
) -> None:
    """Create a plot showing refusal rates across models."""
    if output_dir is None:
        output_dir = Path(PROJECT_DIR) / "logs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    refusal_metrics = [m for m in REFUSAL_COLORS.keys() if m in df.columns]

    # Check if we have any refusal data
    has_refusals = False
    for metric in refusal_metrics:
        if df[metric].notna().any():
            has_refusals = True
            break

    if not has_refusals:
        logger.info("No refusal rate data found in results")
        return

    # Create subplots for each refusal metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    plot_idx = 0
    for metric_key in refusal_metrics:
        if df[metric_key].notna().any() and plot_idx < len(axes):
            ax = axes[plot_idx]
            plot_idx += 1

            # Group by model
            for model in df["model"].unique():
                model_df = df[df["model"] == model]

                # Skip if this model doesn't have refusal data
                if model_df[metric_key].isna().all():
                    continue

                # Determine parameter type
                if model_df["single_scale"].notna().any():
                    param_name = "single_scale"
                    param_label = "Single Scale"
                elif model_df["merging_tau"].notna().any():
                    param_name = "merging_tau"
                    param_label = "Merging Tau"
                else:
                    continue

                # Get sorted parameter values
                param_values = sorted(model_df[param_name].dropna().unique())

                # Get metric values
                metric_values = []
                for param_val in param_values:
                    row = model_df[model_df[param_name] == param_val]
                    if not row.empty and metric_key in row.columns:
                        value = row[metric_key].iloc[0]
                        metric_values.append(value)
                    else:
                        metric_values.append(None)

                # Filter valid values
                valid_indices = [
                    i for i, v in enumerate(metric_values) if v is not None
                ]
                valid_param_values = [param_values[i] for i in valid_indices]
                valid_metric_values = [metric_values[i] for i in valid_indices]

                if valid_metric_values:
                    # Shorten model name for legend
                    short_model_name = model.split("/")[-1]
                    ax.plot(
                        valid_param_values,
                        valid_metric_values,
                        marker="s",
                        markersize=4,
                        linewidth=1.5,
                        label=short_model_name,
                        alpha=0.7,
                        linestyle="--",
                    )

            ax.set_xlabel(param_label, fontsize=10)
            ax.set_ylabel("Refusal Rate", fontsize=10)
            ax.set_title(f"{METRIC_LABELS[metric_key]}", fontsize=12)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(loc="best", fontsize=8, ncol=1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Refusal Rates Across Models", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_plot:
        output_path = output_dir / "hyperparam_refusal_rates.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved refusal rate plot to {output_path}")

    plt.show()


def main():
    """Main function to generate all plots."""
    try:
        # Load results
        results = load_hyperparameter_results()

        # Create figures per metric across all models (accuracy + refusal per model)
        logger.info("Creating feature figures across all models...")
        plot_features_all_models(results)

        logger.info("All plots generated successfully!")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info("Please run the main evaluation script first to generate results.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
