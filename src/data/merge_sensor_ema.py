"""
Merge Sensor Features with EMA Responses

This script creates the supervised learning dataset by aligning
passive sensor data (hourly features) with EMA stress responses.

Approach:
  For each EMA stress response at time T:
    1. Aggregate sensor features from the N hours preceding T
    2. Pair the sensor summary with the stress level → one training sample

This gives us: "Given what the phone sensed in the past N hours,
predict the student's self-reported stress level."
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# ──────────────────────── Configuration ────────────────────────
ALIGNED_PATH = Path("data/processed/aligned")
EMA_PATH = Path("data/processed/ema")
OUTPUT_PATH = Path("data/processed")

# How many hours of sensor data to aggregate before each EMA response
LOOKBACK_HOURS = 6

# Sensor columns from aligned data
SENSOR_COLS = [
    "activity_active_minutes",
    "activity_unknown_minutes",
    "conversation_minutes",
    "wifi_unique_aps",
    "dark_minutes",
    "phonelock_minutes",
    "phonelock_count",
    "phonecharge_minutes",
    "audio_voice_minutes",
    "audio_noise_minutes",
]


def load_participant_sensor_data(pid: str) -> pd.DataFrame:
    """Load hourly aligned sensor data for one participant."""
    fpath = ALIGNED_PATH / f"aligned_{pid}.csv"
    if not fpath.exists():
        return pd.DataFrame()

    df = pd.read_csv(fpath)
    # Handle unnamed index column
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def aggregate_sensor_window(sensor_df: pd.DataFrame,
                            ema_time: pd.Timestamp,
                            lookback_hours: int = LOOKBACK_HOURS) -> dict:
    """
    Aggregate sensor features from [ema_time - lookback, ema_time].

    For each sensor column, compute:
      - mean, sum, std, min, max over the window
      - count of non-zero hours (activity indicator)

    Returns a flat dict of features, or None if insufficient data.
    """
    window_start = ema_time - timedelta(hours=lookback_hours)
    window = sensor_df[
        (sensor_df["timestamp"] >= window_start) &
        (sensor_df["timestamp"] <= ema_time)
    ]

    # Require at least 2 hours of data in the window
    if len(window) < 2:
        return None

    features = {"hours_of_data": len(window)}

    for col in SENSOR_COLS:
        if col not in window.columns:
            continue
        vals = window[col].fillna(0)
        features[f"{col}_mean"] = vals.mean()
        features[f"{col}_sum"] = vals.sum()
        features[f"{col}_std"] = vals.std()
        features[f"{col}_max"] = vals.max()
        features[f"{col}_nonzero"] = (vals > 0).sum()

    return features


def build_sensor_ema_dataset(lookback_hours: int = LOOKBACK_HOURS) -> pd.DataFrame:
    """
    Build the full supervised dataset: sensor features → stress level.

    For each participant:
      1. Load their aligned sensor data
      2. Load their stress EMA responses
      3. For each stress response, aggregate sensor features
      4. Pair them into one row

    Returns
    -------
    DataFrame with sensor features + stress_level + metadata
    """
    # Load stress data
    stress_path = EMA_PATH / "stress_with_features.csv"
    if not stress_path.exists():
        stress_path = EMA_PATH / "ema_stress.csv"
    stress_df = pd.read_csv(stress_path, parse_dates=["timestamp"])

    print(f"📊 Stress responses: {len(stress_df)} from "
          f"{stress_df['participant_id'].nunique()} participants")

    all_rows = []
    participants = stress_df["participant_id"].unique()

    for pid in sorted(participants):
        # Load sensor data
        sensor_df = load_participant_sensor_data(pid)
        if sensor_df.empty:
            continue

        # Get this participant's stress responses
        pid_stress = stress_df[stress_df["participant_id"] == pid]

        matched = 0
        for _, ema_row in pid_stress.iterrows():
            ema_time = ema_row["timestamp"]
            features = aggregate_sensor_window(sensor_df, ema_time, lookback_hours)
            if features is None:
                continue

            row = {
                "participant_id": pid,
                "ema_timestamp": ema_time,
                "stress_level": ema_row["stress_level"],
                **features,
            }

            # Add stress_score if available
            if "stress_score" in ema_row:
                row["stress_score"] = ema_row["stress_score"]

            # Add time features
            row["hour_of_day"] = ema_time.hour
            row["day_of_week"] = ema_time.dayofweek
            row["is_weekend"] = int(ema_time.dayofweek >= 5)

            all_rows.append(row)
            matched += 1

        print(f"  {pid}: {matched}/{len(pid_stress)} stress responses matched with sensor data")

    result = pd.DataFrame(all_rows)
    print(f"\n✅ Final dataset: {len(result)} samples, "
          f"{result['participant_id'].nunique()} participants")
    return result


def create_train_test_split(df: pd.DataFrame,
                            test_size: float = 0.15,
                            val_size: float = 0.15,
                            random_state: int = 42) -> tuple:
    """
    Create time-aware train/val/test split.

    Strategy: Split chronologically within each participant to prevent
    data leakage (future data can't predict past stress).

    Returns (train_df, val_df, test_df)
    """
    train_parts, val_parts, test_parts = [], [], []

    for pid, group in df.groupby("participant_id"):
        group = group.sort_values("ema_timestamp")
        n = len(group)

        # Chronological split per participant
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))

        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    print(f"\n📊 Split sizes:")
    print(f"   Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def main():
    print("=" * 60)
    print("Sensor ↔ EMA Merge Pipeline")
    print("=" * 60)

    # 1. Build merged dataset
    merged = build_sensor_ema_dataset()

    if merged.empty:
        print("❌ No data merged. Check that aligned sensor data and EMA data exist.")
        return

    # 2. Save full merged dataset
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH / "sensor_ema_merged.csv", index=False)
    print(f"\n💾 Saved sensor_ema_merged.csv ({len(merged)} rows)")

    # 3. Create train/val/test split
    train_df, val_df, test_df = create_train_test_split(merged)

    train_df.to_csv(OUTPUT_PATH / "train_stress.csv", index=False)
    val_df.to_csv(OUTPUT_PATH / "val_stress.csv", index=False)
    test_df.to_csv(OUTPUT_PATH / "test_stress.csv", index=False)
    print("💾 Saved train_stress.csv, val_stress.csv, test_stress.csv")

    # 4. Print feature summary
    feature_cols = [c for c in merged.columns if c not in
                    ("participant_id", "ema_timestamp", "stress_level",
                     "stress_score", "stress_label")]
    print(f"\n📐 Feature count: {len(feature_cols)}")
    print(f"   Features: {feature_cols[:10]}...")

    # 5. Class distribution
    print(f"\n📊 Stress level distribution:")
    dist = merged["stress_level"].value_counts().sort_index()
    for level, count in dist.items():
        pct = count / len(merged) * 100
        bar = "█" * int(pct / 2)
        label = {1: "A little stressed", 2: "Definitely stressed",
                 3: "Stressed out", 4: "Feeling good", 5: "Feeling great"}.get(level, "?")
        print(f"   {level} ({label}): {count:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
