"""
Sensor ↔ EMA Correlation Analysis

Investigates whether passive phone sensor data correlates with
self-reported EMA stress levels. This is the key validation that
"what your phone sees" matches "how you actually feel."

Analysis:
  1. Correlation matrix: sensor features vs stress level
  2. Statistical significance testing
  3. High-stress vs low-stress behavioral comparison
  4. Temporal lag analysis
  5. Per-participant correlation strength

Outputs: reports/figures/correlation/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# ──────────────────────── Configuration ────────────────────────
DATA_PATH = Path("data/processed")
FIG_PATH = Path("reports/figures/correlation")
FIG_PATH.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 11,
    "axes.titlesize": 14,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

# Human-readable feature names for plots
FEATURE_DISPLAY = {
    "activity_active_minutes_mean": "Physical Activity",
    "activity_unknown_minutes_mean": "Unknown Activity",
    "conversation_minutes_mean": "Conversation",
    "wifi_unique_aps_mean": "WiFi Locations",
    "dark_minutes_mean": "Phone Dark Time",
    "phonelock_minutes_mean": "Screen Time",
    "phonelock_count_mean": "Screen Unlocks",
    "phonecharge_minutes_mean": "Charging Time",
    "audio_voice_minutes_mean": "Voice (Audio)",
    "audio_noise_minutes_mean": "Ambient Noise",
    "hour_of_day": "Hour of Day",
    "is_weekend": "Weekend",
    "screen_vs_charge_ratio": "Screen/Charge Ratio",
    "voice_vs_noise_ratio": "Voice/Noise Ratio",
    "activity_ratio": "Activity Ratio",
    "dark_ratio": "Dark Time Ratio",
    "social_engagement": "Social Engagement",
}


def load_merged_data() -> pd.DataFrame:
    """Load the merged sensor-EMA dataset."""
    df = pd.read_csv(DATA_PATH / "sensor_ema_merged.csv", parse_dates=["ema_timestamp"])
    return df


def plot_sensor_stress_correlation(df):
    """1. Heatmap: correlation between sensor features and stress."""
    # Select key features
    key_features = [
        "activity_active_minutes_mean", "activity_unknown_minutes_mean",
        "conversation_minutes_mean", "wifi_unique_aps_mean",
        "dark_minutes_mean", "phonelock_minutes_mean",
        "phonelock_count_mean", "phonecharge_minutes_mean",
        "audio_voice_minutes_mean", "audio_noise_minutes_mean",
        "hour_of_day", "is_weekend",
    ]
    available = [f for f in key_features if f in df.columns]

    corr_data = df[available + ["stress_level"]].corr()
    stress_corr = corr_data["stress_level"].drop("stress_level")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Bar chart of correlations
    labels = [FEATURE_DISPLAY.get(f, f) for f in stress_corr.index]
    colors = ["#D32F2F" if v < 0 else "#43A047" for v in stress_corr.values]
    y_pos = range(len(stress_corr))

    axes[0].barh(y_pos, stress_corr.values, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Pearson Correlation with Stress Level")
    axes[0].set_title("Sensor Features vs Self-Reported Stress")
    axes[0].axvline(x=0, color="black", linewidth=0.5)

    # Add significance stars
    for i, (feat, corr_val) in enumerate(stress_corr.items()):
        r, p = stats.pearsonr(df[feat].fillna(0), df["stress_level"])
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        axes[0].text(corr_val + 0.005 * np.sign(corr_val), i,
                     f"r={corr_val:.3f}{star}", va="center", fontsize=9)

    # Full correlation heatmap
    sns.heatmap(corr_data.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=axes[1], square=True, linewidths=0.5,
                xticklabels=[FEATURE_DISPLAY.get(f, f[:15]) for f in available],
                yticklabels=[FEATURE_DISPLAY.get(f, f[:15]) for f in available])
    axes[1].set_title("Sensor Feature Inter-Correlations")

    plt.tight_layout()
    plt.savefig(FIG_PATH / "01_sensor_stress_correlation.png")
    plt.close()
    print("  ✓ 01_sensor_stress_correlation.png")


def plot_high_vs_low_stress(df):
    """2. Compare sensor patterns: high-stress vs low-stress periods."""
    # Define groups
    # Stress levels 1-2 = stressed, 4-5 = not stressed (original scale)
    high_stress = df[df["stress_level"].isin([1, 2])]  # "A little" + "Definitely" stressed
    low_stress = df[df["stress_level"].isin([4, 5])]    # "Feeling good" + "Feeling great"

    features = [
        "activity_active_minutes_mean", "conversation_minutes_mean",
        "wifi_unique_aps_mean", "dark_minutes_mean",
        "phonelock_minutes_mean", "audio_voice_minutes_mean",
        "audio_noise_minutes_mean", "phonecharge_minutes_mean",
    ]
    available = [f for f in features if f in df.columns]

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for i, feat in enumerate(available):
        if i >= len(axes):
            break
        ax = axes[i]
        data_plot = pd.DataFrame({
            "Stressed\n(Level 1-2)": high_stress[feat].dropna().values[:200],
        })
        data_plot2 = pd.DataFrame({
            "Not Stressed\n(Level 4-5)": low_stress[feat].dropna().values[:200],
        })

        # KDE plot
        if len(high_stress[feat].dropna()) > 5:
            high_stress[feat].dropna().plot.kde(ax=ax, color="#D32F2F", label="Stressed", linewidth=2)
        if len(low_stress[feat].dropna()) > 5:
            low_stress[feat].dropna().plot.kde(ax=ax, color="#43A047", label="Not Stressed", linewidth=2)

        ax.set_title(FEATURE_DISPLAY.get(feat, feat), fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlabel("")

        # Statistical test
        stat, p = stats.mannwhitneyu(
            high_stress[feat].dropna(), low_stress[feat].dropna(),
            alternative="two-sided"
        )
        sig = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.text(0.95, 0.95, sig, transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    # Hide empty axes
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Sensor Behavior: Stressed vs Not Stressed Periods",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_PATH / "02_high_vs_low_stress.png")
    plt.close()
    print("  ✓ 02_high_vs_low_stress.png")


def plot_per_participant_correlation(df):
    """3. How well do sensors predict stress for EACH participant?"""
    # For each participant, compute correlation between sensor features and stress
    key_feature = "phonelock_minutes_mean"  # Screen time as the showcase feature
    if key_feature not in df.columns:
        return

    results = []
    for pid, group in df.groupby("participant_id"):
        if len(group) < 10:
            continue
        r, p = stats.pearsonr(group[key_feature].fillna(0), group["stress_level"])
        results.append({"participant_id": pid, "correlation": r, "p_value": p, "n": len(group)})

    results_df = pd.DataFrame(results).sort_values("correlation")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Per-participant correlation bars
    colors = ["#D32F2F" if r < 0 else "#43A047" for r in results_df["correlation"]]
    axes[0].barh(range(len(results_df)), results_df["correlation"],
                  color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(range(len(results_df)))
    axes[0].set_yticklabels(results_df["participant_id"], fontsize=7)
    axes[0].set_xlabel("Pearson r (Screen Time vs Stress)")
    axes[0].set_title("Per-Participant: Screen Time ↔ Stress Correlation")
    axes[0].axvline(x=0, color="black", linewidth=0.5)

    # Histogram of correlations
    axes[1].hist(results_df["correlation"], bins=15, color="#5C6BC0",
                 edgecolor="white", alpha=0.8)
    axes[1].axvline(x=0, color="black", linewidth=0.5)
    axes[1].axvline(x=results_df["correlation"].mean(), color="#D32F2F",
                     linestyle="--", label=f"Mean r={results_df['correlation'].mean():.3f}")
    axes[1].set_xlabel("Correlation Coefficient")
    axes[1].set_ylabel("Number of Participants")
    axes[1].set_title("Distribution of Screen Time ↔ Stress Correlations")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIG_PATH / "03_per_participant_correlation.png")
    plt.close()
    print("  ✓ 03_per_participant_correlation.png")


def plot_feature_importance_comparison(df):
    """4. Model-agnostic feature importance via correlation strength."""
    features = [c for c in df.columns if c.endswith("_mean") or c.endswith("_sum")
                or c in ("hour_of_day", "is_weekend", "day_of_week")]
    available = [f for f in features if f in df.columns]

    correlations = []
    for feat in available:
        r, p = stats.pearsonr(df[feat].fillna(0), df["stress_level"])
        correlations.append({"feature": feat, "abs_correlation": abs(r), "correlation": r, "p_value": p})

    corr_df = pd.DataFrame(correlations).sort_values("abs_correlation", ascending=True)
    corr_df = corr_df.tail(20)  # Top 20

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#D32F2F" if r < 0 else "#43A047" for r in corr_df["correlation"]]
    ax.barh(range(len(corr_df)), corr_df["abs_correlation"], color=colors,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels([FEATURE_DISPLAY.get(f, f.replace("_", " ")[:30])
                        for f in corr_df["feature"]], fontsize=9)
    ax.set_xlabel("|Pearson Correlation| with Stress Level")
    ax.set_title("Top 20 Sensor Features by Stress Correlation\n"
                 "(🟢 positive = more → less stressed, 🔴 negative = more → more stressed)")

    # Add significance
    for i, row in enumerate(corr_df.itertuples()):
        star = "***" if row.p_value < 0.001 else "**" if row.p_value < 0.01 else "*" if row.p_value < 0.05 else ""
        ax.text(row.abs_correlation + 0.002, i, star, va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(FIG_PATH / "04_feature_importance_correlation.png")
    plt.close()
    print("  ✓ 04_feature_importance_correlation.png")


def generate_correlation_report(df):
    """5. Generate text summary of findings."""
    report_path = FIG_PATH.parent.parent / "results" / "sensor_ema_correlation_report.txt"

    features = [c for c in df.columns if c.endswith("_mean")]
    available = [f for f in features if f in df.columns]

    lines = [
        "=" * 60,
        "Sensor ↔ EMA Stress Correlation Report",
        "=" * 60,
        f"\nDataset: {len(df)} sensor-stress pairs from {df['participant_id'].nunique()} participants",
        f"Stress level distribution: {dict(df['stress_level'].value_counts().sort_index())}",
        f"\nNote: Stress scale 1=A little stressed, 2=Definitely stressed,",
        f"      3=Stressed out, 4=Feeling good, 5=Feeling great",
        f"\n{'─'*60}",
        "Correlation with Stress Level (Pearson r, two-tailed p-value):",
        f"{'─'*60}",
    ]

    for feat in sorted(available):
        r, p = stats.pearsonr(df[feat].fillna(0), df["stress_level"])
        sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
        display = FEATURE_DISPLAY.get(feat, feat)
        lines.append(f"  {sig} r={r:+.4f}  p={p:.4f}  {display}")

    # Key findings
    lines.extend([
        f"\n{'─'*60}",
        "Key Findings:",
        f"{'─'*60}",
    ])

    # Find strongest correlations
    corrs = [(f, *stats.pearsonr(df[f].fillna(0), df["stress_level"])) for f in available]
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    lines.append("\nStrongest correlations:")
    for feat, r, p in corrs[:5]:
        display = FEATURE_DISPLAY.get(feat, feat)
        direction = "higher → less stressed" if r > 0 else "higher → more stressed"
        lines.append(f"  • {display}: r={r:+.4f} ({direction})")

    # High vs low stress comparison
    high = df[df["stress_level"].isin([1, 2])]
    low = df[df["stress_level"].isin([4, 5])]
    lines.extend([
        f"\nBehavioral Differences (Stressed vs Not Stressed):",
        f"  Stressed periods: {len(high)} samples",
        f"  Not stressed periods: {len(low)} samples",
    ])

    for feat in available[:5]:
        h_mean = high[feat].mean()
        l_mean = low[feat].mean()
        display = FEATURE_DISPLAY.get(feat, feat)
        pct = ((h_mean - l_mean) / (l_mean + 0.001)) * 100
        lines.append(f"  • {display}: stressed={h_mean:.1f} vs calm={l_mean:.1f} ({pct:+.0f}%)")

    lines.append(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")

    report = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  ✓ Correlation report saved to {report_path}")
    print("\n" + report)


# ──────────────── Main ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("🔗 Sensor ↔ EMA Correlation Analysis")
    print("=" * 60)

    df = load_merged_data()
    print(f"📂 Loaded {len(df)} samples, {df['participant_id'].nunique()} participants")
    print(f"   Features: {len(df.columns)} columns")

    # Add engineered features for correlation analysis
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from src.analysis.modeling.stress_prediction import engineer_features
    df = engineer_features(df)

    print(f"\n🎨 Generating correlation visualizations → {FIG_PATH}/")
    plot_sensor_stress_correlation(df)
    plot_high_vs_low_stress(df)
    plot_per_participant_correlation(df)
    plot_feature_importance_comparison(df)

    print(f"\n📝 Generating correlation report...")
    generate_correlation_report(df)

    print(f"\n✅ Correlation analysis complete!")


if __name__ == "__main__":
    main()
