"""
Single User Deep Dive: Complete Behavioral Profile

Takes ONE participant and creates a comprehensive visualization of:
  - All sensor streams over time
  - EMA stress responses overlaid on sensor data
  - The "story" of one student's term through data

This is the showcase piece: "Here's what one person's digital life
looks like, and here's how their phone data predicts their stress."

Output: reports/figures/deep_dive/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

# ──────────────────────── Configuration ────────────────────────
ALIGNED_PATH = Path("data/processed/aligned")
EMA_PATH = Path("data/processed/ema")
DATA_PATH = Path("data/processed")
FIG_PATH = Path("reports/figures/deep_dive")
FIG_PATH.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (16, 10),
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

STRESS_COLORS = {1: "#FFA726", 2: "#EF5350", 3: "#D32F2F", 4: "#66BB6A", 5: "#43A047"}
STRESS_LABELS = {1: "A little stressed", 2: "Definitely stressed",
                 3: "Stressed out", 4: "Feeling good", 5: "Feeling great"}


def select_best_participant() -> str:
    """Select the participant with the most EMA responses + sensor data."""
    stress_df = pd.read_csv(EMA_PATH / "stress_with_features.csv")
    response_counts = stress_df.groupby("participant_id").size()

    # Filter to those with aligned sensor data
    aligned_files = list(ALIGNED_PATH.glob("aligned_*.csv"))
    available_pids = {f.stem.replace("aligned_", "") for f in aligned_files}
    response_counts = response_counts[response_counts.index.isin(available_pids)]

    # Pick the one with most responses (good data coverage)
    best_pid = response_counts.idxmax()
    print(f"Selected participant: {best_pid} ({response_counts[best_pid]} stress responses)")
    return best_pid


def load_participant_data(pid: str) -> tuple:
    """Load all data for one participant."""
    # Sensor data (hourly)
    sensor_path = ALIGNED_PATH / f"aligned_{pid}.csv"
    sensor_df = pd.read_csv(sensor_path)
    if "Unnamed: 0" in sensor_df.columns:
        sensor_df = sensor_df.rename(columns={"Unnamed: 0": "timestamp"})
    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])

    # Stress EMA
    stress_df = pd.read_csv(EMA_PATH / "stress_with_features.csv", parse_dates=["timestamp"])
    stress_df = stress_df[stress_df["participant_id"] == pid]

    # Sleep EMA
    sleep_df = pd.read_csv(EMA_PATH / "ema_sleep.csv", parse_dates=["timestamp"])
    sleep_df = sleep_df[sleep_df["participant_id"] == pid]

    # Social EMA
    social_path = EMA_PATH / "ema_social.csv"
    social_df = pd.DataFrame()
    if social_path.exists():
        social_df = pd.read_csv(social_path, parse_dates=["timestamp"])
        social_df = social_df[social_df["participant_id"] == pid]

    return sensor_df, stress_df, sleep_df, social_df


def plot_complete_timeline(pid, sensor_df, stress_df, sleep_df):
    """1. Complete sensor + EMA timeline for one participant."""
    fig, axes = plt.subplots(6, 1, figsize=(18, 20), sharex=True)

    date_min = sensor_df["timestamp"].min()
    date_max = sensor_df["timestamp"].max()

    # ── Panel 1: Physical Activity ──
    ax = axes[0]
    ax.fill_between(sensor_df["timestamp"],
                     sensor_df["activity_active_minutes"].fillna(0),
                     color="#FF7043", alpha=0.6, label="Active minutes")
    ax.set_ylabel("Active\nMinutes/hr")
    ax.set_title(f"📱 Complete Digital Life Profile: Participant {pid}", fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    # ── Panel 2: Screen Time ──
    ax = axes[1]
    ax.fill_between(sensor_df["timestamp"],
                     sensor_df["phonelock_minutes"].fillna(0),
                     color="#42A5F5", alpha=0.6, label="Screen time")
    ax.set_ylabel("Phone Lock\nMinutes/hr")
    ax.legend(loc="upper right", fontsize=9)

    # ── Panel 3: Audio Environment ──
    ax = axes[2]
    ax.fill_between(sensor_df["timestamp"],
                     sensor_df["audio_voice_minutes"].fillna(0),
                     color="#66BB6A", alpha=0.6, label="Voice")
    ax.fill_between(sensor_df["timestamp"],
                     sensor_df["audio_noise_minutes"].fillna(0),
                     color="#FFA726", alpha=0.4, label="Noise")
    ax.set_ylabel("Audio\nMinutes/hr")
    ax.legend(loc="upper right", fontsize=9)

    # ── Panel 4: Dark Time (Sleep Proxy) ──
    ax = axes[3]
    ax.fill_between(sensor_df["timestamp"],
                     sensor_df["dark_minutes"].fillna(0),
                     color="#5C6BC0", alpha=0.5, label="Dark time")
    ax.set_ylabel("Dark\nMinutes/hr")
    ax.legend(loc="upper right", fontsize=9)

    # ── Panel 5: WiFi / Location ──
    ax = axes[4]
    ax.fill_between(sensor_df["timestamp"],
                     sensor_df["wifi_unique_aps"].fillna(0),
                     color="#AB47BC", alpha=0.5, label="Unique WiFi APs")
    ax.set_ylabel("WiFi APs\n(locations)")
    ax.legend(loc="upper right", fontsize=9)

    # ── Panel 6: STRESS EMA overlaid ──
    ax = axes[5]
    if not stress_df.empty:
        for level in sorted(stress_df["stress_level"].unique()):
            mask = stress_df["stress_level"] == level
            ax.scatter(stress_df.loc[mask, "timestamp"],
                       stress_df.loc[mask, "stress_score"],
                       c=STRESS_COLORS.get(level, "gray"), s=50, alpha=0.8,
                       label=STRESS_LABELS.get(level, str(level)),
                       edgecolors="white", linewidth=0.5, zorder=5)

        # Rolling average
        stress_sorted = stress_df.sort_values("timestamp")
        if len(stress_sorted) > 3:
            rolling = stress_sorted.set_index("timestamp")["stress_score"].rolling("3D").mean()
            ax.plot(rolling.index, rolling.values, color="#D32F2F",
                    linewidth=2, alpha=0.8, label="3-day rolling avg")

    ax.set_ylabel("Stress Score\n(1=great, 5=stressed)")
    ax.set_ylim(0.5, 5.5)
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.axhline(y=3, color="gray", linestyle="--", alpha=0.3)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIG_PATH / f"01_timeline_{pid}.png")
    plt.close()
    print(f"  ✓ 01_timeline_{pid}.png")


def plot_daily_patterns(pid, sensor_df, stress_df):
    """2. Average daily patterns (hour of day) with stress overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sensor_df["hour"] = sensor_df["timestamp"].dt.hour

    # Panel 1: Hourly activity pattern
    hourly = sensor_df.groupby("hour")["activity_active_minutes"].mean()
    axes[0, 0].bar(hourly.index, hourly.values, color="#FF7043", edgecolor="white", alpha=0.8)
    axes[0, 0].set_title("Physical Activity by Hour")
    axes[0, 0].set_xlabel("Hour of Day")
    axes[0, 0].set_ylabel("Avg Active Minutes")

    # Panel 2: Screen time pattern
    hourly_screen = sensor_df.groupby("hour")["phonelock_minutes"].mean()
    axes[0, 1].bar(hourly_screen.index, hourly_screen.values, color="#42A5F5",
                    edgecolor="white", alpha=0.8)
    axes[0, 1].set_title("Screen Time by Hour")
    axes[0, 1].set_xlabel("Hour of Day")
    axes[0, 1].set_ylabel("Avg Lock Minutes")

    # Panel 3: Stress by hour
    if not stress_df.empty and "hour" in stress_df.columns:
        hourly_stress = stress_df.groupby("hour")["stress_score"].agg(["mean", "count"])
        axes[1, 0].bar(hourly_stress.index, hourly_stress["mean"],
                        color="#D32F2F", edgecolor="white", alpha=0.8)
        axes[1, 0].set_title("Stress by Hour of Day")
        axes[1, 0].set_xlabel("Hour of Day")
        axes[1, 0].set_ylabel("Avg Stress Score")
        axes[1, 0].axhline(y=3, color="gray", linestyle="--", alpha=0.5)

    # Panel 4: Day-of-week comparison
    sensor_df["dow"] = sensor_df["timestamp"].dt.dayofweek
    dow_screen = sensor_df.groupby("dow")["phonelock_minutes"].mean()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colors = ["#EF5350" if d < 5 else "#66BB6A" for d in dow_screen.index]
    axes[1, 1].bar(dow_screen.index, dow_screen.values, color=colors, edgecolor="white")
    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(day_names)
    axes[1, 1].set_title("Screen Time: Weekday vs Weekend")
    axes[1, 1].set_ylabel("Avg Lock Minutes")

    plt.suptitle(f"Daily Behavior Patterns — {pid}", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_PATH / f"02_daily_patterns_{pid}.png")
    plt.close()
    print(f"  ✓ 02_daily_patterns_{pid}.png")


def plot_stress_prediction_story(pid, sensor_df, stress_df):
    """3. The prediction story: show how sensor data precedes stress reports."""
    if stress_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Get merged data for this participant
    merged_path = DATA_PATH / "sensor_ema_merged.csv"
    if merged_path.exists():
        merged = pd.read_csv(merged_path, parse_dates=["ema_timestamp"])
        merged = merged[merged["participant_id"] == pid].sort_values("ema_timestamp")
    else:
        return

    if merged.empty:
        return

    # Panel 1: Sensor features over time (colored by stress level)
    ax = axes[0]
    scatter = ax.scatter(merged["ema_timestamp"],
                          merged["phonelock_minutes_mean"],
                          c=merged["stress_level"],
                          cmap="RdYlGn", s=60, alpha=0.8,
                          edgecolors="white", linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Stress Level (1=stressed, 5=great)")
    ax.set_ylabel("Screen Time (6h avg before EMA)")
    ax.set_title(f"Screen Time Before Each Stress Report — {pid}", fontsize=14)

    # Panel 2: Multiple features stacked area
    ax = axes[1]
    features = ["activity_active_minutes_mean", "phonelock_minutes_mean",
                "audio_voice_minutes_mean", "dark_minutes_mean"]
    available = [f for f in features if f in merged.columns]
    colors = ["#FF7043", "#42A5F5", "#66BB6A", "#5C6BC0"]

    for i, feat in enumerate(available):
        ax.plot(merged["ema_timestamp"], merged[feat].fillna(0),
                label=feat.replace("_mean", "").replace("_minutes", ""),
                color=colors[i], linewidth=1.5, alpha=0.7)

    # Overlay stress as background shading
    for _, row in merged.iterrows():
        color = STRESS_COLORS.get(row["stress_level"], "gray")
        ax.axvspan(row["ema_timestamp"] - pd.Timedelta(hours=1),
                   row["ema_timestamp"] + pd.Timedelta(hours=1),
                   alpha=0.1, color=color)

    ax.set_xlabel("Date")
    ax.set_ylabel("Sensor Feature Value (6h avg)")
    ax.set_title("Sensor Features + Stress Reports Timeline")
    ax.legend(loc="upper right", fontsize=9)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIG_PATH / f"03_prediction_story_{pid}.png")
    plt.close()
    print(f"  ✓ 03_prediction_story_{pid}.png")


def plot_sensor_ema_scatter(pid, sensor_df, stress_df):
    """4. Scatter plots: each sensor feature vs stress level."""
    merged_path = DATA_PATH / "sensor_ema_merged.csv"
    if not merged_path.exists():
        return

    merged = pd.read_csv(merged_path, parse_dates=["ema_timestamp"])
    merged = merged[merged["participant_id"] == pid]

    if len(merged) < 10:
        return

    features = [
        ("phonelock_minutes_mean", "Screen Time"),
        ("activity_active_minutes_mean", "Physical Activity"),
        ("audio_voice_minutes_mean", "Voice/Conversation"),
        ("dark_minutes_mean", "Dark Time (Sleep)"),
        ("wifi_unique_aps_mean", "Location Variety"),
        ("conversation_minutes_mean", "Conversation Time"),
    ]
    available = [(f, l) for f, l in features if f in merged.columns]

    n_plots = len(available)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for i, (feat, label) in enumerate(available):
        ax = axes[i]
        ax.scatter(merged[feat], merged["stress_level"],
                   c=[STRESS_COLORS.get(s, "gray") for s in merged["stress_level"]],
                   alpha=0.6, s=40, edgecolors="white", linewidth=0.5)

        # Trend line
        z = np.polyfit(merged[feat].fillna(0), merged["stress_level"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(merged[feat].min(), merged[feat].max(), 50)
        ax.plot(x_range, p(x_range), "--", color="gray", alpha=0.5)

        ax.set_xlabel(label)
        ax.set_ylabel("Stress Level")
        ax.set_title(f"{label} vs Stress")
        ax.set_yticks([1, 2, 3, 4, 5])

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Sensor Features vs Stress Level — {pid}",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_PATH / f"04_scatter_{pid}.png")
    plt.close()
    print(f"  ✓ 04_scatter_{pid}.png")


def generate_participant_summary(pid, sensor_df, stress_df, sleep_df, social_df):
    """5. Text summary of the participant's profile."""
    report_path = FIG_PATH / f"summary_{pid}.txt"

    lines = [
        "=" * 60,
        f"  Participant {pid} — Complete Behavioral Profile",
        "=" * 60,
        f"\n📊 Data Coverage:",
        f"   Sensor data: {sensor_df['timestamp'].min().strftime('%Y-%m-%d')} → "
        f"{sensor_df['timestamp'].max().strftime('%Y-%m-%d')}",
        f"   Hourly records: {len(sensor_df)}",
        f"   Stress EMA responses: {len(stress_df)}",
        f"   Sleep EMA responses: {len(sleep_df)}",
        f"   Social EMA responses: {len(social_df)}",
    ]

    if not stress_df.empty:
        lines.extend([
            f"\n😰 Stress Profile:",
            f"   Mean stress score: {stress_df['stress_score'].mean():.2f}/5 "
            f"(1=great, 5=stressed out)",
            f"   Most common: {STRESS_LABELS.get(int(stress_df['stress_level'].mode().iloc[0]), 'N/A')}",
            f"   Worst reported: {STRESS_LABELS.get(int(stress_df['stress_level'].min()), 'N/A')}",
        ])

    lines.extend([
        f"\n📱 Phone Behavior (Daily Averages):",
        f"   Screen time: {sensor_df['phonelock_minutes'].mean():.1f} min/hr",
        f"   Active minutes: {sensor_df['activity_active_minutes'].mean():.1f} min/hr",
        f"   Dark time: {sensor_df['dark_minutes'].mean():.1f} min/hr",
        f"   Voice audio: {sensor_df['audio_voice_minutes'].mean():.1f} min/hr",
        f"   WiFi APs: {sensor_df['wifi_unique_aps'].mean():.1f} unique/hr",
    ])

    if not sleep_df.empty:
        avg_sleep = sleep_df["sleep_hours"].dropna().mean()
        lines.extend([
            f"\n😴 Sleep:",
            f"   Average self-reported: {avg_sleep:.1f} hours/night",
        ])

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  ✓ summary_{pid}.txt")
    print(report)


# ──────────────── Main ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("🔬 Single User Deep Dive")
    print("=" * 60)

    # Select best participant
    pid = select_best_participant()

    # Load all data
    print(f"\n📂 Loading all data for {pid}...")
    sensor_df, stress_df, sleep_df, social_df = load_participant_data(pid)
    print(f"   Sensor: {len(sensor_df)} hourly records")
    print(f"   Stress: {len(stress_df)} EMA responses")
    print(f"   Sleep: {len(sleep_df)} EMA responses")
    print(f"   Social: {len(social_df)} EMA responses")

    # Generate visualizations
    print(f"\n🎨 Generating deep dive visualizations → {FIG_PATH}/")
    plot_complete_timeline(pid, sensor_df, stress_df, sleep_df)
    plot_daily_patterns(pid, sensor_df, stress_df)
    plot_stress_prediction_story(pid, sensor_df, stress_df)
    plot_sensor_ema_scatter(pid, sensor_df, stress_df)

    # Generate summary
    print(f"\n📝 Generating profile summary...")
    generate_participant_summary(pid, sensor_df, stress_df, sleep_df, social_df)

    print(f"\n✅ Deep dive complete! All outputs in {FIG_PATH}/")


if __name__ == "__main__":
    main()
