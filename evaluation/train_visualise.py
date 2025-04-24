import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from datetime import datetime


def analyze_registration_times(
    time_log_file="registration_times.csv", output_prefix="registration_analysis_"
):
    """Analyze and visualize response times from GEI registration process."""
    if not os.path.exists(time_log_file):
        print(f"Error: Time log file {time_log_file} not found")
        return

    # Load the data
    try:
        df = pd.read_csv(time_log_file)
        print(f"Loaded {len(df)} records from {time_log_file}")
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return

    # Filter out failed requests
    successful_df = df[df["status"] == "success"].copy()
    print(f"Found {len(successful_df)} successful registrations")

    if "timestamp" in successful_df.columns:
        successful_df["timestamp"] = pd.to_datetime(successful_df["timestamp"])
        successful_df["time_elapsed"] = (
            successful_df["timestamp"] - successful_df["timestamp"].min()
        ).dt.total_seconds()

    # Basic statistics
    stats = {
        "count": len(successful_df),
        "mean": successful_df["response_time"].mean(),
        "median": successful_df["response_time"].median(),
        "min": successful_df["response_time"].min(),
        "max": successful_df["response_time"].max(),
        "std": successful_df["response_time"].std(),
    }

    # 1. Chronological response time plot
    plt.figure(figsize=(12, 6))

    # Add index for sequential ordering if needed
    successful_df["request_number"] = range(1, len(successful_df) + 1)

    # Plot response time vs request number
    plt.plot(
        successful_df["request_number"],
        successful_df["response_time"],
        marker="o",
        linestyle="-",
        alpha=0.6,
        color="#3274A1",
    )

    # Add moving average line
    window_size = min(20, len(successful_df) // 5) if len(successful_df) > 20 else 1
    if window_size > 0:
        successful_df["moving_avg"] = (
            successful_df["response_time"].rolling(window=window_size).mean()
        )
        plt.plot(
            successful_df["request_number"],
            successful_df["moving_avg"],
            color="red",
            linewidth=2,
            label=f"Moving Average (window={window_size})",
        )

    # plt.title("GEI Registration Response Times", fontsize=16)
    plt.xlabel("Request Number", fontsize=12)
    plt.ylabel("Response Time (seconds)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}response_time_chronological.png", dpi=300)

    # 2. Response time distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(successful_df["response_time"], bins=20, kde=True, color="#3274A1")
    plt.axvline(
        x=stats["mean"],
        color="red",
        linestyle="--",
        label=f"Mean: {stats['mean']:.3f}s",
    )
    plt.axvline(
        x=stats["median"],
        color="green",
        linestyle="-",
        label=f"Median: {stats['median']:.3f}s",
    )

    # plt.title("GEI Registration Response Time Distribution", fontsize=16)
    plt.xlabel("Response Time (seconds)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}response_time_distribution.png", dpi=300)

    # 3. Trend analysis - response time as dataset grows
    if len(successful_df) > 10:
        plt.figure(figsize=(12, 6))

        # Calculate cumulative average
        successful_df["cumulative_avg"] = (
            successful_df["response_time"].expanding().mean()
        )

        # Plot both the individual response times and cumulative average
        plt.plot(
            successful_df["request_number"],
            successful_df["response_time"],
            "o",
            alpha=0.4,
            label="Individual Response Times",
        )
        plt.plot(
            successful_df["request_number"],
            successful_df["cumulative_avg"],
            "r-",
            linewidth=2,
            label="Cumulative Average",
        )

        # Add trend line
        z = np.polyfit(
            successful_df["request_number"], successful_df["response_time"], 1
        )
        p = np.poly1d(z)
        plt.plot(
            successful_df["request_number"],
            p(successful_df["request_number"]),
            "g--",
            linewidth=1.5,
            label=f"Trend: y={z[0]:.6f}x+{z[1]:.6f}",
        )

        # plt.title("Response Time Trend as Training Dataset Grows", fontsize=16)
        plt.xlabel("Number of GEI Samples Registered", fontsize=12)
        plt.ylabel("Response Time (seconds)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}response_time_trend.png", dpi=300)

    # 4. Save summary to text file
    with open(f"{output_prefix}response_time_summary.txt", "w") as f:
        f.write("GEI Registration Response Time Analysis\n")
        f.write("=======================================\n\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data source: {time_log_file}\n\n")

        f.write("Basic Statistics:\n")
        f.write(f"  Total requests analyzed: {stats['count']}\n")
        f.write(f"  Mean response time: {stats['mean']:.4f} seconds\n")
        f.write(f"  Median response time: {stats['median']:.4f} seconds\n")
        f.write(
            f"  Min/Max response time: {stats['min']:.4f}/{stats['max']:.4f} seconds\n"
        )
        f.write(f"  Standard deviation: {stats['std']:.4f} seconds\n\n")

        # Growth trend analysis
        if len(successful_df) > 10:
            first_quarter_avg = successful_df.iloc[: len(successful_df) // 4][
                "response_time"
            ].mean()
            last_quarter_avg = successful_df.iloc[-len(successful_df) // 4 :][
                "response_time"
            ].mean()
            percent_change = (
                (last_quarter_avg - first_quarter_avg) / first_quarter_avg
            ) * 100

            f.write("Growth Trend Analysis:\n")
            f.write(f"  First 25% of requests avg: {first_quarter_avg:.4f} seconds\n")
            f.write(f"  Last 25% of requests avg: {last_quarter_avg:.4f} seconds\n")
            f.write(f"  Change: {percent_change:.2f}%\n")
            f.write(f"  Linear trend slope: {z[0]:.8f} seconds/request\n\n")

            if percent_change > 10:
                f.write(
                    "  FINDING: Response times show SIGNIFICANT INCREASE as more samples are registered\n"
                )
            elif percent_change < -10:
                f.write(
                    "  FINDING: Response times show SIGNIFICANT DECREASE as more samples are registered\n"
                )
            else:
                f.write(
                    "  FINDING: Response times remain RELATIVELY STABLE as more samples are registered\n"
                )

    print(f"Analysis complete. Results saved with prefix '{output_prefix}'")


if __name__ == "__main__":
    # Default file name
    log_file = "registration_times.csv"
    output_prefix = "registration_analysis_"

    # Check if file name is provided as argument
    if len(sys.argv) > 1:
        log_file = sys.argv[1]

    # Check if output prefix is provided as second argument
    if len(sys.argv) > 2:
        output_prefix = sys.argv[2]

    analyze_registration_times(log_file, output_prefix)
