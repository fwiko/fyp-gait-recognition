import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def load_results(results_file):
    """Load results from JSON or CSV file."""
    if results_file.endswith(".json"):
        with open(results_file, "r") as f:
            return json.load(f)
    elif results_file.endswith(".csv"):
        return pd.read_csv(results_file).to_dict("records")
    else:
        raise ValueError("Unsupported file format. Use .json or .csv")


def create_visualizations(results_file, output_prefix="gei_analysis_"):
    """Create visualizations and metrics from test results."""
    # Load results
    results = load_results(results_file)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    # Filter out entries with errors
    valid_df = df.dropna(subset=["true_label", "predicted_label"])

    # Get unique labels
    all_labels = sorted(
        set(valid_df["true_label"].unique()) | set(valid_df["predicted_label"].unique())
    )

    # Prepare data for metrics
    y_true = valid_df["true_label"]
    y_pred = valid_df["predicted_label"]
    y_pred_threshold = valid_df.apply(
        lambda x: x["predicted_label"] if x["above_threshold"] else "unknown", axis=1
    )

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Calculate metrics with threshold
    accuracy_threshold = accuracy_score(y_true, y_pred_threshold)

    # 1. Create Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=all_labels,
        yticklabels=all_labels,
    )
    # plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}confusion_matrix.png", dpi=300)

    # 2. Create Confidence Distribution
    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 105, 5)

    # Separate correct and incorrect predictions
    correct_conf = valid_df[valid_df["correct"]]["confidence"]
    incorrect_conf = valid_df[~valid_df["correct"]]["confidence"]

    plt.hist(correct_conf, bins=bins, alpha=0.7, label="Correct Predictions")
    plt.hist(incorrect_conf, bins=bins, alpha=0.7, label="Incorrect Predictions")

    plt.axvline(x=75, color="r", linestyle="--", label="75% Threshold")
    # plt.title("Confidence Score Distribution", fontsize=16)
    plt.xlabel("Confidence (%)", fontsize=12)
    plt.ylabel("Number of Predictions", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}confidence_distribution.png", dpi=300)

    # 3. Create Classification Report Visualization
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T

    plt.figure(figsize=(12, 6 + len(all_labels) * 0.3))
    sns.heatmap(
        report_df.iloc[:-3, :3], annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5
    )
    # plt.title("Classification Report", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}classification_report.png", dpi=300)

    # 4. Overall Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    metrics = {
        "Accuracy": accuracy,
        "Precision (macro)": precision_macro,
        "Recall (macro)": recall_macro,
        "F1 Score (macro)": f1_macro,
        "Accuracy (75% threshold)": accuracy_threshold,
    }

    bars = plt.bar(
        metrics.keys(),
        metrics.values(),
        color=["#3274A1", "#E1812C", "#3A923A", "#9372B2", "#C03D3E"],
    )
    # plt.title("Overall Performance Metrics", fontsize=16)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(f"{output_prefix}overall_metrics.png", dpi=300)

    # 5. If response time data is available, plot it
    if "response_time" in valid_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(valid_df["response_time"], bins=20, kde=True)
        plt.axvline(
            x=valid_df["response_time"].mean(),
            color="r",
            linestyle="--",
            label=f'Mean: {valid_df["response_time"].mean():.3f}s',
        )
        # plt.title("API Response Time Distribution", fontsize=16)
        plt.xlabel("Response Time (seconds)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}response_time.png", dpi=300)

    # 6. Save metrics to text file
    with open(f"{output_prefix}metrics_summary.txt", "w") as f:
        f.write("GEI Classification Performance Metrics\n")
        f.write("=====================================\n\n")

        f.write(f"Total samples: {len(valid_df)}\n")
        f.write(f"Number of subjects: {len(all_labels)}\n\n")

        # Add overall analysis
        total_correct = valid_df["correct"].sum()
        total_incorrect = len(valid_df) - total_correct
        most_confused = (
            valid_df[~valid_df["correct"]]
            .groupby(["true_label", "predicted_label"])
            .size()
            .reset_index(name="count")
        )
        if not most_confused.empty:
            most_confused = most_confused.sort_values("count", ascending=False).iloc[0]

        f.write("Overall Analysis:\n")
        f.write(
            f"  Total correct identifications: {total_correct} ({total_correct/len(valid_df)*100:.2f}%)\n"
        )
        f.write(
            f"  Total misclassifications: {total_incorrect} ({total_incorrect/len(valid_df)*100:.2f}%)\n"
        )

        # Most common misclassifications
        if not most_confused.empty:
            f.write(
                f"  Most common misclassification: Subject {most_confused['true_label']} as {most_confused['predicted_label']} (occurred {most_confused['count']} times)\n"
            )

        # Response time analysis if available
        if "response_time" in valid_df.columns:
            f.write(
                f"  Average response time: {valid_df['response_time'].mean():.4f} seconds\n"
            )
            f.write(
                f"  Min/Max response time: {valid_df['response_time'].min():.4f}/{valid_df['response_time'].max():.4f} seconds\n"
            )

        # Confidence analysis
        f.write(
            f"  Average confidence (all predictions): {valid_df['confidence'].mean():.2f}%\n"
        )
        f.write(
            f"  Average confidence (correct predictions): {valid_df[valid_df['correct']]['confidence'].mean():.2f}%\n"
        )
        f.write(
            f"  Average confidence (incorrect predictions): {valid_df[~valid_df['correct']]['confidence'].mean():.2f}%\n\n"
        )

        # Add standard metrics
        f.write("Standard Metrics:\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"  Precision (macro): {precision_macro:.4f}\n")
        f.write(f"  Recall (macro): {recall_macro:.4f}\n")
        f.write(f"  F1 Score (macro): {f1_macro:.4f}\n\n")

        f.write("Metrics with 75% Confidence Threshold:\n")
        f.write(f"  Accuracy: {accuracy_threshold:.4f}\n\n")

        f.write("Per-class Classification Report:\n")
        f.write(classification_report(y_true, y_pred, zero_division=0))

        # Add threshold analysis
        above_threshold = valid_df["above_threshold"].sum()
        correct_above_threshold = valid_df[
            valid_df["above_threshold"] & valid_df["correct"]
        ].shape[0]
        incorrect_above_threshold = valid_df[
            valid_df["above_threshold"] & ~valid_df["correct"]
        ].shape[0]
        below_threshold = len(valid_df) - above_threshold

        f.write(f"\nThreshold Analysis (75% confidence):\n")
        f.write(
            f"  Predictions above threshold: {above_threshold} ({above_threshold/len(valid_df)*100:.2f}%)\n"
        )
        f.write(
            f"  Predictions below threshold: {below_threshold} ({below_threshold/len(valid_df)*100:.2f}%)\n"
        )
        f.write(
            f"  Correct predictions above threshold: {correct_above_threshold} ({correct_above_threshold/above_threshold*100:.2f}% of high confidence)\n"
        )
        f.write(
            f"  Incorrect predictions above threshold: {incorrect_above_threshold} ({incorrect_above_threshold/above_threshold*100:.2f}% of high confidence)\n"
        )

        if above_threshold > 0:
            f.write(
                f"  Precision at threshold: {correct_above_threshold/above_threshold:.4f}\n"
            )

    print(f"Analysis complete. Visualizations saved with prefix '{output_prefix}'")


if __name__ == "__main__":
    RESULTS_FILE = "classification_results.json"  # Change to your results file
    OUTPUT_PREFIX = "gei_analysis_"  # Change to desired prefix

    create_visualizations(RESULTS_FILE, OUTPUT_PREFIX)
