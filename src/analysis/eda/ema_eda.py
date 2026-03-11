"""
EMA Exploratory Data Analysis (EDA)

Comprehensive analysis of EMA (Ecological Momentary Assessment) data
from the StudentLife dataset. Generates publication-ready visualizations.

Outputs saved to: reports/figures/ema/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ──────────────────────── Configuration ────────────────────────
EMA_PATH = Path("data/processed/ema")
FIG_PATH = Path("reports/figures/ema")
FIG_PATH.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

STRESS_LABELS = {1: "A little\nstressed", 2: "Definitely\nstressed",
                 3: "Stressed\nout", 4: "Feeling\ngood", 5: "Feeling\ngreat"}
STRESS_COLORS = {1: "#FFA726", 2: "#EF5350", 3: "#D32F2F", 4: "#66BB6A", 5: "#43A047"}

# Study dates
STUDY_START = datetime(2013, 3, 27)


def load_data():
    """Load all processed EMA data."""
    data = {}
    for fpath in sorted(EMA_PATH.glob("*.csv")):
        name = fpath.stem.replace("ema_", "").replace("stress_with_features", "stress_features")
        data[name] = pd.read_csv(fpath, parse_dates=["timestamp"] if "timestamp" in
                                  pd.read_csv(fpath, nrows=0).columns else [])
    return data


def plot_stress_distribution(stress_df):
    """1. Overall stress level distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    counts = stress_df["stress_level"].value_counts().sort_index()
    colors = [STRESS_COLORS[i] for i in counts.index]
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_xlabel("Stress Level")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Self-Reported Stress Distribution")
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xticklabels([STRESS_LABELS[i] for i in range(1, 6)])

    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     str(count), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values, labels=[STRESS_LABELS[i].replace("\n", " ") for i in counts.index],
                colors=colors, autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 10})
    axes[1].set_title("Stress Level Proportions")

    plt.tight_layout()
    plt.savefig(FIG_PATH / "01_stress_distribution.png")
    plt.close()
    print("  ✓ 01_stress_distribution.png")


def plot_stress_over_time(stress_df):
    """2. Stress trends across the 10-week term."""
    if "week_of_term" not in stress_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Weekly mean stress score
    weekly = stress_df.groupby("week_of_term")["stress_score"].agg(["mean", "std", "count"])
    axes[0].plot(weekly.index, weekly["mean"], "o-", color="#D32F2F", linewidth=2, markersize=8)
    axes[0].fill_between(weekly.index,
                          weekly["mean"] - weekly["std"],
                          weekly["mean"] + weekly["std"],
                          alpha=0.2, color="#D32F2F")
    axes[0].set_xlabel("Week of Term")
    axes[0].set_ylabel("Stress Score (1=great, 5=stressed out)")
    axes[0].set_title("Average Stress Level Across Term")
    axes[0].set_xticks(range(1, 11))
    axes[0].axhline(y=3, color="gray", linestyle="--", alpha=0.5, label="Neutral")
    axes[0].legend()

    # Weekly response count
    axes[1].bar(weekly.index, weekly["count"], color="#42A5F5", edgecolor="white")
    axes[1].set_xlabel("Week of Term")
    axes[1].set_ylabel("Number of Responses")
    axes[1].set_title("EMA Response Rate Over Term")
    axes[1].set_xticks(range(1, 11))

    plt.tight_layout()
    plt.savefig(FIG_PATH / "02_stress_over_time.png")
    plt.close()
    print("  ✓ 02_stress_over_time.png")


def plot_stress_by_time_of_day(stress_df):
    """3. Stress patterns by hour of day and day of week."""
    if "hour" not in stress_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By hour
    hourly = stress_df.groupby("hour")["stress_score"].mean()
    axes[0].plot(hourly.index, hourly.values, "o-", color="#7B1FA2", linewidth=2, markersize=6)
    axes[0].fill_between(hourly.index, hourly.values, alpha=0.15, color="#7B1FA2")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Average Stress Score")
    axes[0].set_title("Stress by Time of Day")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].axhline(y=stress_df["stress_score"].mean(), color="gray",
                     linestyle="--", alpha=0.5, label="Overall mean")
    axes[0].legend()

    # By day of week
    stress_df["dow"] = pd.to_datetime(stress_df["timestamp"]).dt.dayofweek
    daily = stress_df.groupby("dow")["stress_score"].mean()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors = ["#EF5350" if d < 5 else "#66BB6A" for d in daily.index]
    axes[1].bar(daily.index, daily.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Average Stress Score")
    axes[1].set_title("Stress by Day of Week")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_names)

    plt.tight_layout()
    plt.savefig(FIG_PATH / "03_stress_time_patterns.png")
    plt.close()
    print("  ✓ 03_stress_time_patterns.png")


def plot_participant_variability(stress_df):
    """4. Stress variation across participants."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Per-participant mean stress
    participant_stats = stress_df.groupby("participant_id")["stress_score"].agg(["mean", "std", "count"])
    participant_stats = participant_stats.sort_values("mean")

    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(participant_stats)))
    axes[0].barh(range(len(participant_stats)), participant_stats["mean"],
                  xerr=participant_stats["std"], color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Mean Stress Score (1=great, 5=stressed out)")
    axes[0].set_ylabel("Participant")
    axes[0].set_title("Stress Levels by Participant")
    axes[0].set_yticks(range(len(participant_stats)))
    axes[0].set_yticklabels(participant_stats.index, fontsize=7)
    axes[0].axvline(x=3, color="gray", linestyle="--", alpha=0.5)

    # Response count per participant
    axes[1].barh(range(len(participant_stats)), participant_stats["count"],
                  color="#42A5F5", edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("Number of Responses")
    axes[1].set_ylabel("Participant")
    axes[1].set_title("EMA Response Count per Participant")
    axes[1].set_yticks(range(len(participant_stats)))
    axes[1].set_yticklabels(participant_stats.index, fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG_PATH / "04_participant_variability.png")
    plt.close()
    print("  ✓ 04_participant_variability.png")


def plot_missing_data_analysis(stress_df, sleep_df, social_df):
    """5. Missing data and response completeness analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Response rates per participant across categories
    categories = {"Stress": stress_df, "Sleep": sleep_df, "Social": social_df}
    for name, df in categories.items():
        if df is not None and not df.empty:
            counts = df.groupby("participant_id").size().reindex(
                [f"u{i:02d}" for i in range(60)], fill_value=0)
            axes[0].plot(range(len(counts)), counts.values, "o-",
                          label=name, markersize=4, alpha=0.7)

    axes[0].set_xlabel("Participant")
    axes[0].set_ylabel("Response Count")
    axes[0].set_title("EMA Response Rates by Category")
    axes[0].legend()

    # Temporal coverage: responses per day
    if not stress_df.empty:
        daily = stress_df.groupby(stress_df["timestamp"].dt.date).size()
        axes[1].plot(daily.index, daily.values, color="#D32F2F", alpha=0.7, linewidth=1)
        axes[1].fill_between(daily.index, daily.values, alpha=0.2, color="#D32F2F")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Stress Responses per Day")
        axes[1].set_title("Temporal Coverage of Stress EMA")
        axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(FIG_PATH / "05_missing_data.png")
    plt.close()
    print("  ✓ 05_missing_data.png")


def plot_sleep_analysis(sleep_df):
    """6. Sleep patterns from EMA."""
    if sleep_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sleep hours distribution
    hours = sleep_df["sleep_hours"].dropna()
    if not hours.empty:
        axes[0].hist(hours, bins=20, color="#5C6BC0", edgecolor="white", alpha=0.8)
        axes[0].axvline(x=hours.mean(), color="#D32F2F", linestyle="--",
                        label=f"Mean: {hours.mean():.1f}h")
        axes[0].axvline(x=7, color="#66BB6A", linestyle="--",
                        label="Recommended: 7h")
        axes[0].set_xlabel("Sleep Hours")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Self-Reported Sleep Duration")
        axes[0].legend()

    # Sleep quality distribution
    quality = sleep_df["sleep_quality"].dropna()
    if not quality.empty:
        q_counts = quality.value_counts().sort_index()
        q_labels = {1: "Very\ngood", 2: "Fairly\ngood", 3: "Fairly\nbad", 4: "Very\nbad"}
        q_colors = ["#43A047", "#66BB6A", "#EF5350", "#D32F2F"]
        axes[1].bar(q_counts.index, q_counts.values, color=q_colors[:len(q_counts)],
                    edgecolor="white", linewidth=1.5)
        axes[1].set_xticks(q_counts.index)
        axes[1].set_xticklabels([q_labels.get(i, str(i)) for i in q_counts.index])
        axes[1].set_xlabel("Sleep Quality")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Self-Reported Sleep Quality")

    plt.tight_layout()
    plt.savefig(FIG_PATH / "06_sleep_analysis.png")
    plt.close()
    print("  ✓ 06_sleep_analysis.png")


def plot_ema_correlation_heatmap(stress_df, sleep_df, social_df):
    """7. Cross-EMA correlation (stress vs sleep vs social)."""
    # Merge by participant + date for daily-level correlation
    if stress_df.empty:
        return

    stress_daily = stress_df.groupby(
        [stress_df["participant_id"], stress_df["timestamp"].dt.date]
    ).agg({"stress_score": "mean"}).reset_index()
    stress_daily.columns = ["participant_id", "date", "avg_stress"]

    merged = stress_daily.copy()

    if sleep_df is not None and not sleep_df.empty:
        sleep_daily = sleep_df.groupby(
            [sleep_df["participant_id"], sleep_df["timestamp"].dt.date]
        ).agg({"sleep_hours": "mean", "sleep_quality": "mean"}).reset_index()
        sleep_daily.columns = ["participant_id", "date", "avg_sleep_hours", "avg_sleep_quality"]
        merged = merged.merge(sleep_daily, on=["participant_id", "date"], how="left")

    if social_df is not None and not social_df.empty:
        social_daily = social_df.groupby(
            [social_df["participant_id"], social_df["timestamp"].dt.date]
        ).agg({"social_contacts": "mean"}).reset_index()
        social_daily.columns = ["participant_id", "date", "avg_social"]
        merged = merged.merge(social_daily, on=["participant_id", "date"], how="left")

    # Correlation matrix
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = merged[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, linewidths=1, linecolor="white")
    ax.set_title("Cross-EMA Correlation\n(Daily averages per participant)")

    plt.tight_layout()
    plt.savefig(FIG_PATH / "07_ema_correlation.png")
    plt.close()
    print("  ✓ 07_ema_correlation.png")


def plot_normalization_summary(stress_df):
    """8. Data normalization overview."""
    if stress_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original distribution
    axes[0, 0].hist(stress_df["stress_score"], bins=5, color="#5C6BC0",
                    edgecolor="white", alpha=0.8, density=True)
    axes[0, 0].set_title("Stress Score Distribution (Raw)")
    axes[0, 0].set_xlabel("Stress Score")
    axes[0, 0].set_ylabel("Density")

    # Z-score normalized
    z_scores = (stress_df["stress_score"] - stress_df["stress_score"].mean()) / stress_df["stress_score"].std()
    axes[0, 1].hist(z_scores, bins=20, color="#42A5F5", edgecolor="white", alpha=0.8, density=True)
    axes[0, 1].set_title("Z-Score Normalized")
    axes[0, 1].set_xlabel("Z-Score")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].axvline(x=0, color="red", linestyle="--", alpha=0.5)

    # Min-max normalized
    min_max = (stress_df["stress_score"] - stress_df["stress_score"].min()) / \
              (stress_df["stress_score"].max() - stress_df["stress_score"].min())
    axes[1, 0].hist(min_max, bins=20, color="#66BB6A", edgecolor="white", alpha=0.8, density=True)
    axes[1, 0].set_title("Min-Max Normalized [0, 1]")
    axes[1, 0].set_xlabel("Normalized Score")
    axes[1, 0].set_ylabel("Density")

    # QQ plot equivalent - sorted values
    sorted_scores = np.sort(stress_df["stress_score"].values)
    theoretical = np.random.normal(stress_df["stress_score"].mean(),
                                    stress_df["stress_score"].std(),
                                    len(sorted_scores))
    theoretical.sort()
    axes[1, 1].scatter(theoretical, sorted_scores, alpha=0.3, s=10, color="#7B1FA2")
    min_val = min(theoretical.min(), sorted_scores.min())
    max_val = max(theoretical.max(), sorted_scores.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)
    axes[1, 1].set_title("Q-Q Plot (vs Normal)")
    axes[1, 1].set_xlabel("Theoretical Quantiles")
    axes[1, 1].set_ylabel("Sample Quantiles")

    plt.suptitle("Data Normalization Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_PATH / "08_normalization.png")
    plt.close()
    print("  ✓ 08_normalization.png")


# ──────────────── Main ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("📊 EMA Exploratory Data Analysis")
    print("=" * 60)

    # Load data
    print("\n📂 Loading processed EMA data...")
    data = load_data()
    for name, df in data.items():
        print(f"  {name}: {len(df)} rows, {df.shape[1]} cols")

    # Get key dataframes
    stress_features = data.get("stress_features", pd.DataFrame())
    stress = data.get("stress", pd.DataFrame())
    sleep = data.get("sleep", pd.DataFrame())
    social = data.get("social", pd.DataFrame())

    # Use the richer stress_features df if available
    stress_df = stress_features if not stress_features.empty else stress
    if "timestamp" in stress_df.columns:
        stress_df["timestamp"] = pd.to_datetime(stress_df["timestamp"])
    if "timestamp" in sleep.columns:
        sleep["timestamp"] = pd.to_datetime(sleep["timestamp"])
    if "timestamp" in social.columns:
        social["timestamp"] = pd.to_datetime(social["timestamp"])

    print(f"\n📈 Key stats:")
    if not stress_df.empty:
        print(f"  Stress: {len(stress_df)} responses, "
              f"{stress_df['participant_id'].nunique()} participants")
        print(f"  Mean stress score: {stress_df['stress_score'].mean():.2f} "
              f"(1=great, 5=stressed out)")
    if not sleep.empty:
        print(f"  Sleep: {len(sleep)} responses")
    if not social.empty:
        print(f"  Social: {len(social)} responses")

    # Generate plots
    print(f"\n🎨 Generating visualizations → {FIG_PATH}/")

    if not stress_df.empty:
        plot_stress_distribution(stress_df)
        plot_stress_over_time(stress_df)
        plot_stress_by_time_of_day(stress_df)
        plot_participant_variability(stress_df)
        plot_normalization_summary(stress_df)

    plot_missing_data_analysis(stress_df, sleep, social)

    if not sleep.empty:
        plot_sleep_analysis(sleep)

    plot_ema_correlation_heatmap(stress_df, sleep, social)

    print(f"\n✅ EDA complete! {len(list(FIG_PATH.glob('*.png')))} figures saved to {FIG_PATH}/")


if __name__ == "__main__":
    main()
