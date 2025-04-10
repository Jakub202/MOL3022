#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, matthews_corrcoef

import tensorflow as tf
from data_preprocessing import parse_fasta_file, prepare_data_for_training
from model import SignalPeptideModel

# SP Class mapping for readability
SP_CLASS_MAP = {
    0: "S",  # Sec/SPI signal peptide
    1: "T",  # Tat/SPI or Tat/SPII signal peptide
    2: "L",  # Sec/SPII signal peptide
    3: "P",  # Sec/SPIII signal peptide
    4: "I",  # cytoplasm
    5: "M",  # transmembrane
    6: "O",  # extracellular
    7: "unknown",
}

# Define the SP type to index mapping
SP_TYPE_TO_INDEX = {"SEC/SPI": 0, "TAT/SPI": 1, "SEC/SPII": 2}

# Kingdom mapping
KINGDOM_MAP = {0: "ARCHAEA", 1: "EUKARYA", 2: "NEGATIVE", 3: "POSITIVE", 4: "unknown"}

# Kingdom display names for plotting
KINGDOM_DISPLAY = {
    0: "Archaea",
    1: "Eukaryotes",
    2: "Gram-Negative Bacteria",
    3: "Gram-Positive Bacteria",
}


def load_model_and_predict(model_path, X_seq, X_kingdom, max_length=100):
    """
    Load model and make predictions

    Args:
        model_path: Path to the saved model
        X_seq: Encoded sequences
        X_kingdom: Encoded kingdom information
        max_length: Maximum sequence length used for denormalizing cleavage positions

    Returns:
        Dictionary with predictions
    """
    # Load model
    model = SignalPeptideModel()
    model.load_model(model_path)

    # Make predictions
    return model.predict(X_seq, X_kingdom, max_length)


def calculate_cleavage_site_metrics_by_type(
    true_classes,
    pred_classes,
    true_sp_classes,
    pred_sp_classes,
    true_cleavage,
    pred_cleavage,
    kingdoms,
    max_distance=4,
):
    """
    Calculate precision and recall for cleavage site prediction by SP type and kingdom

    Args:
        true_classes: True SP presence (binary)
        pred_classes: Predicted SP presence (binary)
        true_sp_classes: True SP class indices
        pred_sp_classes: Predicted SP class indices
        true_cleavage: True cleavage site positions
        pred_cleavage: Predicted cleavage site positions
        kingdoms: Kingdom indices for each sequence
        max_distance: Maximum distance to consider

    Returns:
        Dictionary with metrics
    """
    # Initialize metrics dictionaries by SP type and kingdom
    # First dimension: kingdom (0-3)
    # Second dimension: SP type (SEC/SPI=0, TAT/SPI=1, SEC/SPII=2)
    # Third dimension: distance (0-3)
    precision = np.zeros((4, 3, max_distance + 1))
    recall = np.zeros((4, 3, max_distance + 1))

    # Initialize array to store MAE by kingdom
    mae_by_kingdom = np.zeros(4)
    mae_counts = np.zeros(4)

    # Create mapping from SP class index to SP type
    # Sec/SPI = class 0
    # Tat/SPI = class 1
    # Sec/SPII = class 2
    sp_class_to_type = {
        0: 0,  # Sec/SPI -> SEC/SPI
        1: 1,  # Tat/SPI -> TAT/SPI
        2: 2,  # Sec/SPII -> SEC/SPII
    }

    # Calculate metrics for each kingdom and SP type
    for kingdom_idx in range(4):  # 0-3 for the four kingdoms
        # Filter sequences for this kingdom
        kingdom_indices = [i for i, k in enumerate(kingdoms) if k == kingdom_idx]

        if not kingdom_indices:
            continue

        # Calculate MAE for this kingdom (only for correct SP class predictions)
        kingdom_mae_sum = 0
        kingdom_mae_count = 0

        # For each SP type
        for sp_type_idx in range(3):  # 0-2 for the three SP types
            # Get corresponding SP class index
            sp_class_idx = list(sp_class_to_type.keys())[sp_type_idx]

            # Filter sequences with this SP type (true positives)
            sp_type_indices = [
                i
                for i in kingdom_indices
                if true_classes[i] == 1 and true_sp_classes[i] == sp_class_idx
            ]

            # Count true positives (sequences with this SP type)
            true_positive_count = len(sp_type_indices)

            # If no sequences have this SP type in this kingdom, skip
            if true_positive_count == 0:
                continue

            # Count predicted positives (sequences predicted to have this SP type)
            pred_positive_indices = [
                i
                for i in kingdom_indices
                if pred_classes[i] == 1 and pred_sp_classes[i] == sp_class_idx
            ]
            pred_positives = len(pred_positive_indices)

            # If no predictions for this SP type, precision is 0 for all distances
            if pred_positives == 0:
                precision[kingdom_idx, sp_type_idx, :] = 0
            else:
                # For each distance threshold
                for distance in range(max_distance + 1):
                    # Count correct predictions at this distance threshold
                    # A prediction is correct if:
                    # 1. True SP presence is correctly predicted
                    # 2. SP type is correctly predicted
                    # 3. Cleavage site is within the distance threshold

                    # For precision: among predicted positives, how many are correct
                    correct_predictions = 0
                    for i in pred_positive_indices:
                        if (
                            true_classes[i] == 1
                            and true_sp_classes[i] == sp_class_idx
                            and abs(pred_cleavage[i] - true_cleavage[i]) <= distance
                        ):
                            correct_predictions += 1

                    # Calculate precision
                    precision[kingdom_idx, sp_type_idx, distance] = (
                        correct_predictions / pred_positives
                    )

            # For recall: among true positives, how many are correctly predicted
            # If no true positives, recall is 0 for all distances
            if true_positive_count == 0:
                recall[kingdom_idx, sp_type_idx, :] = 0
            else:
                for distance in range(max_distance + 1):
                    correct_predictions = 0
                    for i in sp_type_indices:
                        if (
                            pred_classes[i] == 1
                            and pred_sp_classes[i] == sp_class_idx
                            and abs(pred_cleavage[i] - true_cleavage[i]) <= distance
                        ):
                            correct_predictions += 1

                    # Calculate recall
                    recall[kingdom_idx, sp_type_idx, distance] = (
                        correct_predictions / true_positive_count
                    )

            # Calculate MAE for sequences with correct SP type predictions
            for i in sp_type_indices:
                if pred_classes[i] == 1 and pred_sp_classes[i] == sp_class_idx:
                    # Add to kingdom MAE sum
                    kingdom_mae_sum += abs(pred_cleavage[i] - true_cleavage[i])
                    kingdom_mae_count += 1

        # Calculate MAE for this kingdom
        if kingdom_mae_count > 0:
            mae_by_kingdom[kingdom_idx] = kingdom_mae_sum / kingdom_mae_count
            mae_counts[kingdom_idx] = kingdom_mae_count

    return {
        "precision": precision,
        "recall": recall,
        "mae_by_kingdom": mae_by_kingdom,
        "mae_counts": mae_counts,
    }


def plot_cleavage_site_metrics_by_type(metrics, output_dir="results"):
    """
    Plot cleavage site precision and recall metrics by SP type and kingdom

    Args:
        metrics: Dictionary with metrics from calculate_cleavage_site_metrics_by_type
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    precision = metrics["precision"]
    recall = metrics["recall"]

    # Get max distance
    max_distance = precision.shape[2] - 1

    # Create SP type labels and colors
    sp_types = ["SEC/SPI", "TAT/SPI", "SEC/SPII"]
    colors = {
        "SEC/SPI": {"precision": "lightblue", "recall": "blue"},
        "TAT/SPI": {"precision": "lightgreen", "recall": "green"},
        "SEC/SPII": {"precision": "pink", "recall": "red"},
    }

    # Create distance labels for x-axis
    x_labels = ["0"]
    for d in range(1, max_distance + 1):
        x_labels.append(f"+/- {d}")

    # Create the figure with 2x2 subplots (one for each kingdom)
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot metrics for each kingdom
    for kingdom_idx in range(4):
        ax = axes[kingdom_idx]

        # Set subplot title to kingdom name
        ax.set_title(KINGDOM_DISPLAY[kingdom_idx], fontsize=16)

        # Set x and y limits - adjusted to match the desired range
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-0.1, max_distance + 0.1)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)

        # Plot precision and recall for each SP type
        for sp_type_idx, sp_type in enumerate(sp_types):
            # Get precision and recall values for this SP type and kingdom
            p_values = precision[kingdom_idx, sp_type_idx]
            r_values = recall[kingdom_idx, sp_type_idx]

            # Skip if all values are 0 (no data for this SP type in this kingdom)
            if np.all(p_values == 0) and np.all(r_values == 0):
                continue

            # Plot precision
            ax.plot(
                range(max_distance + 1),
                p_values,
                marker="o",
                color=colors[sp_type]["precision"],
                linewidth=2,
                label=f"Precision {sp_type}",
            )

            # Plot recall
            ax.plot(
                range(max_distance + 1),
                r_values,
                marker="o",
                color=colors[sp_type]["recall"],
                linewidth=2,
                label=f"Recall {sp_type}",
            )

        # Add x-tick labels
        ax.set_xticks(range(max_distance + 1))
        ax.set_xticklabels(x_labels)

    # Set common x and y labels
    fig.text(0.5, 0.04, "Distance from True Cleavage Site", ha="center", fontsize=14)
    fig.text(0.04, 0.5, "Performance", va="center", rotation="vertical", fontsize=14)

    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=3,
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(
        os.path.join(output_dir, "cleavage_site_metrics_by_type.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_confusion_matrix(cm, output_dir="results"):
    """
    Plot and save confusion matrix

    Args:
        cm: Confusion matrix
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix for Signal Peptide Prediction", fontsize=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks([0.5, 1.5], ["No SP", "SP"])
    plt.yticks([0.5, 1.5], ["No SP", "SP"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def benchmark_model(benchmark_file, model_path, output_dir="results", max_length=100):
    """
    Benchmark the signal peptide prediction model

    Args:
        benchmark_file: Path to FASTA file with benchmark data
        model_path: Path to the saved model
        output_dir: Directory to save benchmark results
        max_length: Maximum sequence length

    Returns:
        Dictionary with benchmark metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load benchmark data
    print(f"Loading benchmark data from {benchmark_file}...")
    data_df = parse_fasta_file(benchmark_file)

    print(f"Loaded {len(data_df)} sequences")
    print(f"Sequences with signal peptides: {data_df['has_sp'].sum()}")

    # Prepare data
    print("\nPreparing data...")
    data = prepare_data_for_training(data_df, max_length=max_length)

    # Extract true values
    true_has_sp = data["y_has_sp"]
    true_sp_class_onehot = data["y_sp_class"]
    true_sp_class_idx = np.argmax(true_sp_class_onehot, axis=1)
    true_cleavage = data["y_cleavage"]
    kingdom_indices = np.argmax(data["X_kingdom"], axis=1)

    # Ensure cleavage sites are valid
    sp_mask = true_has_sp == 1
    if np.any(true_cleavage[sp_mask] == 0):
        print("ERROR: Some sequences with signal peptides have cleavage site = 0")
        print("Benchmark aborted - please fix data integrity issues")
        sys.exit(1)

    # Make predictions
    print("\nMaking predictions...")
    predictions = load_model_and_predict(
        model_path, data["X_seq"], data["X_kingdom"], max_length
    )

    pred_has_sp = predictions["sp_binary"]
    pred_sp_class_idx = predictions["sp_class_indices"]
    pred_cleavage = predictions["cleavage_positions"]

    # Calculate classification metrics
    print("\nCalculating metrics...")
    cm = confusion_matrix(true_has_sp, pred_has_sp)
    mcc = matthews_corrcoef(true_has_sp, pred_has_sp)

    # Calculate SP class accuracy (only for sequences with SP)
    sp_mask = true_has_sp == 1
    if np.any(sp_mask):
        sp_class_accuracy = np.mean(
            true_sp_class_idx[sp_mask] == pred_sp_class_idx[sp_mask]
        )
    else:
        sp_class_accuracy = 0

    # Calculate cleavage site metrics by SP type
    cs_metrics_by_type = calculate_cleavage_site_metrics_by_type(
        true_has_sp,
        pred_has_sp,
        true_sp_class_idx,
        pred_sp_class_idx,
        true_cleavage,
        pred_cleavage,
        kingdom_indices,
    )

    # Print results
    print("\nBenchmark Results:")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"SP Class Accuracy (for sequences with SP): {sp_class_accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Print MAE by kingdom
    print("\nCleavage Site Mean Absolute Error by Kingdom:")
    mae_by_kingdom = cs_metrics_by_type["mae_by_kingdom"]
    mae_counts = cs_metrics_by_type["mae_counts"]

    for kingdom_idx in range(4):
        if mae_counts[kingdom_idx] > 0:
            print(
                f"  {KINGDOM_DISPLAY[kingdom_idx]}: {mae_by_kingdom[kingdom_idx]:.2f} (n={int(mae_counts[kingdom_idx])})"
            )

    # Calculate overall MAE
    total_count = np.sum(mae_counts)
    if total_count > 0:
        overall_mae = np.sum(mae_by_kingdom * mae_counts) / total_count
        print(f"  Overall: {overall_mae:.2f} (n={int(total_count)})")

    # Plot confusion matrix
    plot_confusion_matrix(cm, output_dir)

    # Plot cleavage site metrics by SP type
    plot_cleavage_site_metrics_by_type(cs_metrics_by_type, output_dir)

    # Save detailed metrics to file
    results_df = pd.DataFrame(
        {
            "ID": data_df["protein_id"],
            "Kingdom": data_df["kingdom"],
            "True_SP": true_has_sp,
            "Pred_SP": pred_has_sp,
            "True_SP_Class": [SP_CLASS_MAP[i] for i in true_sp_class_idx],
            "Pred_SP_Class": [SP_CLASS_MAP[i] for i in pred_sp_class_idx],
            "True_Cleavage": true_cleavage,
            "Pred_Cleavage": pred_cleavage,
        }
    )

    results_df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)

    # Return metrics
    return {
        "mcc": mcc,
        "confusion_matrix": cm,
        "sp_class_accuracy": sp_class_accuracy,
        "cleavage_metrics_by_type": cs_metrics_by_type,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark signal peptide prediction model"
    )
    parser.add_argument(
        "--benchmark-fasta",
        default="data/benchmark.fasta",
        help="Path to benchmark FASTA file",
    )
    parser.add_argument(
        "--benchmark-data",
        default=None,
        help="String containing benchmark data in CSV format",
    )
    parser.add_argument(
        "--model", default="models/sp_model.keras", help="Path to trained model"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--max-length", type=int, default=100, help="Maximum sequence length"
    )

    args = parser.parse_args()

    if args.benchmark_data:
        # Use provided benchmark data string
        benchmark_from_data_string(args.benchmark_data, args.output_dir)
    else:
        # Use traditional file-based benchmarking
        benchmark_model(
            benchmark_file=args.benchmark_fasta,
            model_path=args.model,
            output_dir=args.output_dir,
            max_length=args.max_length,
        )
