"""
EMA (Ecological Momentary Assessment) Data Loader

Parses raw StudentLife EMA JSON files into clean, analysis-ready DataFrames.
Handles the two JSON formats found in the dataset:
  - Legacy format: {"null": "<value>", "resp_time": <epoch>}
  - Structured format: {"level": "<value>", "location": "...", "resp_time": <epoch>}

EMA Categories loaded:
  - Stress (1-5 scale: 1=A little stressed → 5=Feeling great)
  - Sleep (hours + quality rating)
  - Mood / Mood 1 / Mood 2 (happy/sad ratings)
  - Social (contact count)
  - Activity / Exercise / Behavior
  - PAM (Photographic Affect Meter)

Survey data (pre/post term):
  - PHQ-9 (depression severity)
  - Perceived Stress Scale (PSS)
  - Big Five personality
  - PSQI (sleep quality index)
  - Loneliness Scale
  - PANAS (positive/negative affect)
  - Flourishing Scale
  - VR-12 (health status)
"""

import json
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional


# ──────────────────────── Configuration ────────────────────────
RAW_EMA_PATH = Path("data/raw/dataset/EMA")
RAW_SURVEY_PATH = Path("data/raw/dataset/survey")
OUTPUT_PATH = Path("data/processed/ema")

# Core EMA categories relevant to mental-health phenotyping
CORE_EMA_CATEGORIES = ["Stress", "Sleep", "Mood", "Mood 1", "Social", "PAM",
                       "Activity", "Exercise", "Behavior"]

# Stress scale mapping (from EMA_definition.json)
STRESS_LABELS = {
    "1": "A little stressed",
    "2": "Definitely stressed",
    "3": "Stressed out",
    "4": "Feeling good",
    "5": "Feeling great",
}

# Sleep quality mapping
SLEEP_QUALITY_LABELS = {
    "1": "Very good",
    "2": "Fairly good",
    "3": "Fairly bad",
    "4": "Very bad",
}

# Study date range
STUDY_START = datetime(2013, 3, 27)
STUDY_END = datetime(2013, 6, 5)


# ──────────────────── Parsing Functions ────────────────────────

def _is_gps(value: str) -> bool:
    """Check if a string value is a GPS coordinate (lat,lon)."""
    if not isinstance(value, str):
        return False
    parts = value.split(",")
    if len(parts) != 2:
        return False
    try:
        lat, lon = float(parts[0]), float(parts[1])
        return -90 <= lat <= 90 and -180 <= lon <= 180
    except ValueError:
        return False


def _extract_participant_id(filename: str) -> Optional[str]:
    """Extract participant ID (e.g., 'u00') from filename."""
    match = re.search(r'(u\d{2})', filename)
    return match.group(1) if match else None


def load_ema_category(category: str, ema_path: Path = RAW_EMA_PATH) -> pd.DataFrame:
    """
    Load all participant data for a single EMA category.

    Parameters
    ----------
    category : str
        Category name (e.g., 'Stress', 'Sleep', 'Mood 1')
    ema_path : Path
        Root path to EMA data

    Returns
    -------
    pd.DataFrame with columns: participant_id, timestamp, + category-specific columns
    """
    category_dir = ema_path / "response" / category
    if not category_dir.exists():
        print(f"  ⚠ Category directory not found: {category_dir}")
        return pd.DataFrame()

    all_records = []
    for fname in sorted(os.listdir(category_dir)):
        if not fname.endswith(".json"):
            continue

        pid = _extract_participant_id(fname)
        if pid is None:
            continue

        filepath = category_dir / fname
        try:
            with open(filepath) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  ⚠ Error loading {filepath}: {e}")
            continue

        for entry in entries:
            record = _parse_ema_entry(entry, category, pid)
            if record is not None:
                all_records.append(record)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["participant_id", "timestamp"]).reset_index(drop=True)
    return df


def _parse_ema_entry(entry: dict, category: str, pid: str) -> Optional[dict]:
    """
    Parse a single EMA JSON entry into a flat dict.

    Handles both legacy format (null/resp_time) and structured format.
    Filters out location-only entries (GPS coordinates without a response value).
    """
    resp_time = entry.get("resp_time")
    if resp_time is None:
        return None

    timestamp = datetime.fromtimestamp(resp_time)

    # ── Stress ──
    if category == "Stress":
        level = entry.get("level")
        if level is None:
            # Legacy format: "null" field may hold the level or GPS
            null_val = entry.get("null", "")
            if _is_gps(null_val):
                return None  # Location-only entry, skip
            level = null_val
        try:
            level_int = int(level)
            if level_int not in range(1, 6):
                return None
        except (ValueError, TypeError):
            return None
        return {
            "participant_id": pid,
            "timestamp": timestamp,
            "stress_level": level_int,
            "stress_label": STRESS_LABELS.get(str(level_int), "Unknown"),
            "location": entry.get("location", ""),
        }

    # ── Sleep ──
    elif category == "Sleep":
        hour = entry.get("hour")
        rate = entry.get("rate")
        if hour is None and rate is None:
            null_val = entry.get("null", "")
            if _is_gps(null_val):
                return None
            # Try to determine if it's hour or rate based on value range
            try:
                val = int(null_val)
                # Sleep hours scale goes 1-19 mapping to <3 to 12 hours
                # Sleep rate goes 1-4
                if val <= 4:
                    rate = str(val)
                else:
                    hour = str(val)
            except (ValueError, TypeError):
                return None
        try:
            sleep_hours = None
            if hour is not None:
                hour_int = int(hour)
                # Map the option index to actual hours
                hour_map = {1: 2.5, 2: 3.5, 3: 4, 4: 4.5, 5: 5, 6: 5.5,
                            7: 6, 8: 6.5, 9: 7, 10: 7.5, 11: 8, 12: 8.5,
                            13: 9, 14: 9.5, 15: 10, 16: 10.5, 17: 11,
                            18: 11.5, 19: 12}
                sleep_hours = hour_map.get(hour_int, None)
            sleep_quality = None
            if rate is not None:
                rate_int = int(rate)
                sleep_quality = rate_int  # 1=Very good, 4=Very bad
        except (ValueError, TypeError):
            return None
        if sleep_hours is None and sleep_quality is None:
            return None
        return {
            "participant_id": pid,
            "timestamp": timestamp,
            "sleep_hours": sleep_hours,
            "sleep_quality": sleep_quality,
            "location": entry.get("location", ""),
        }

    # ── Mood / Mood 1 / Mood 2 ──
    elif category.startswith("Mood"):
        happy = entry.get("happy") or entry.get("happyornot")
        sad = entry.get("sad") or entry.get("sadornot")
        if happy is None and sad is None:
            null_val = entry.get("null", "")
            if _is_gps(null_val):
                return None
            return None  # Can't determine without keys
        try:
            happy_val = int(happy) if happy else None
            sad_val = int(sad) if sad else None
        except (ValueError, TypeError):
            return None
        return {
            "participant_id": pid,
            "timestamp": timestamp,
            "mood_happy": happy_val,
            "mood_sad": sad_val,
            "location": entry.get("location", ""),
        }

    # ── Social ──
    elif category == "Social":
        number = entry.get("number")
        if number is None:
            null_val = entry.get("null", "")
            if _is_gps(null_val):
                return None
            number = null_val
        try:
            contact_count = int(number)
        except (ValueError, TypeError):
            return None
        return {
            "participant_id": pid,
            "timestamp": timestamp,
            "social_contacts": contact_count,  # Scale: 1=0-4, 2=5-9, etc.
            "location": entry.get("location", ""),
        }

    # ── PAM (Photographic Affect Meter) ──
    elif category == "PAM":
        picture_idx = entry.get("picture_ind") or entry.get("null", "")
        if _is_gps(str(picture_idx)):
            return None
        try:
            pam_val = int(picture_idx)
        except (ValueError, TypeError):
            return None
        # PAM maps to valence (1-4) and arousal (1-4) based on a 4x4 grid
        valence = ((pam_val - 1) % 4) + 1  # Column: 1=low, 4=high
        arousal = ((pam_val - 1) // 4) + 1  # Row: 1=low, 4=high
        return {
            "participant_id": pid,
            "timestamp": timestamp,
            "pam_index": pam_val,
            "pam_valence": valence,
            "pam_arousal": arousal,
            "location": entry.get("location", ""),
        }

    # ── Activity / Exercise / Behavior (generic) ──
    else:
        # Generic: collect all non-location, non-resp_time fields
        record = {"participant_id": pid, "timestamp": timestamp}
        for key, val in entry.items():
            if key in ("resp_time", "location"):
                continue
            if key == "null":
                if _is_gps(str(val)):
                    continue
                record["response"] = val
            else:
                record[key] = val
        record["location"] = entry.get("location", "")
        return record


# ──────────────── Load All Core EMA Data ───────────────────────

def load_all_ema(ema_path: Path = RAW_EMA_PATH) -> dict[str, pd.DataFrame]:
    """
    Load all core EMA categories into a dictionary of DataFrames.

    Returns
    -------
    dict mapping category name → DataFrame
    """
    print("📋 Loading EMA data...")
    results = {}
    for cat in CORE_EMA_CATEGORIES:
        print(f"  Loading {cat}...")
        df = load_ema_category(cat, ema_path)
        if not df.empty:
            results[cat] = df
            print(f"    ✓ {len(df)} records from {df['participant_id'].nunique()} participants")
        else:
            print(f"    ✗ No data found")
    return results


def load_stress_data(ema_path: Path = RAW_EMA_PATH) -> pd.DataFrame:
    """
    Load and clean stress EMA data specifically.

    Returns DataFrame with: participant_id, timestamp, stress_level (1-5),
    stress_label, date, hour, week_of_term
    """
    df = load_ema_category("Stress", ema_path)
    if df.empty:
        return df

    # Add derived time columns
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    days_since_start = (df["timestamp"] - pd.Timestamp(STUDY_START)).dt.days
    df["week_of_term"] = (days_since_start // 7) + 1
    df = df[df["week_of_term"].between(1, 10)]

    # Invert stress scale for intuitive "higher = more stressed"
    # Original: 1=A little stressed, 2=Definitely stressed, 3=Stressed out,
    #           4=Feeling good, 5=Feeling great
    # Inverted: 1=Feeling great, 5=Stressed out (higher = worse)
    df["stress_score"] = 6 - df["stress_level"]

    return df


# ──────────────── Survey Data Loader ───────────────────────────

def load_survey(name: str, survey_path: Path = RAW_SURVEY_PATH) -> pd.DataFrame:
    """Load a pre/post survey CSV."""
    filepath = survey_path / f"{name}.csv"
    if not filepath.exists():
        print(f"  ⚠ Survey not found: {filepath}")
        return pd.DataFrame()
    return pd.read_csv(filepath)


def load_phq9(survey_path: Path = RAW_SURVEY_PATH) -> pd.DataFrame:
    """
    Load PHQ-9 depression severity scores.

    Returns DataFrame with: uid, type (pre/post), phq9_score (0-27), severity
    """
    df = load_survey("PHQ-9", survey_path)
    if df.empty:
        return df

    # PHQ-9 scoring: each item is 0-3, total 0-27
    response_map = {
        "Not at all": 0,
        "Several days": 1,
        "More than half the days": 2,
        "Nearly every day": 3,
    }

    score_cols = [c for c in df.columns if c not in ("uid", "type", "Response")]
    for col in score_cols:
        df[col] = df[col].map(response_map)

    df["phq9_score"] = df[score_cols].sum(axis=1)

    # Severity categories
    def severity(score):
        if score <= 4:
            return "Minimal"
        elif score <= 9:
            return "Mild"
        elif score <= 14:
            return "Moderate"
        elif score <= 19:
            return "Moderately Severe"
        else:
            return "Severe"

    df["severity"] = df["phq9_score"].apply(severity)
    return df[["uid", "type", "phq9_score", "severity"]]


def load_pss(survey_path: Path = RAW_SURVEY_PATH) -> pd.DataFrame:
    """
    Load Perceived Stress Scale scores.

    Returns DataFrame with: uid, type (pre/post), pss_score (0-40)
    """
    df = load_survey("PerceivedStressScale", survey_path)
    if df.empty:
        return df

    response_map = {
        "Never": 0,
        "Almost never": 1,
        "Sometime": 2,
        "Fairly often": 3,
        "Very often": 4,
    }

    score_cols = [c for c in df.columns if c not in ("uid", "type")]
    for col in score_cols:
        df[col] = df[col].map(response_map)

    # Items 4, 5, 7, 8 are reverse-scored
    reverse_items = [c for c in score_cols if any(
        x in c for x in ["4.", "5.", "7.", "8."]
    )]
    for col in reverse_items:
        df[col] = 4 - df[col]

    df["pss_score"] = df[score_cols].sum(axis=1)
    return df[["uid", "type", "pss_score"]]


# ──────────────── Save Processed EMA Data ──────────────────────

def save_processed_ema(ema_data: dict[str, pd.DataFrame],
                       output_path: Path = OUTPUT_PATH):
    """Save all EMA DataFrames as CSVs."""
    output_path.mkdir(parents=True, exist_ok=True)

    for cat, df in ema_data.items():
        safe_name = cat.lower().replace(" ", "_")
        filepath = output_path / f"ema_{safe_name}.csv"
        df.to_csv(filepath, index=False)
        print(f"  💾 Saved {filepath} ({len(df)} rows)")


# ──────────────── Main Entry Point ─────────────────────────────

def main():
    """Load, parse, and save all EMA data."""
    print("=" * 60)
    print("EMA Data Loading Pipeline")
    print("=" * 60)

    # 1. Load EMA definition
    defn_path = RAW_EMA_PATH / "EMA_definition.json"
    if defn_path.exists():
        with open(defn_path) as f:
            defn = json.load(f)
        print(f"\n📖 EMA Definition: {len(defn)} categories defined")
        for item in defn:
            print(f"   • {item['name']}: {len(item.get('questions', []))} questions")

    # 2. Load all EMA categories
    print()
    ema_data = load_all_ema()

    # 3. Save processed data
    print("\n💾 Saving processed EMA data...")
    save_processed_ema(ema_data)

    # 4. Load and save stress data specifically (with derived features)
    print("\n📊 Processing stress data with derived features...")
    stress_df = load_stress_data()
    if not stress_df.empty:
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        stress_df.to_csv(OUTPUT_PATH / "stress_with_features.csv", index=False)
        print(f"  💾 Saved stress_with_features.csv ({len(stress_df)} rows)")

        # Summary stats
        print(f"\n📈 Stress Data Summary:")
        print(f"   Participants: {stress_df['participant_id'].nunique()}")
        print(f"   Date range: {stress_df['date'].min()} → {stress_df['date'].max()}")
        print(f"   Total responses: {len(stress_df)}")
        print(f"   Mean stress score: {stress_df['stress_score'].mean():.2f} (1=great, 5=stressed out)")
        print(f"   Responses per participant: {stress_df.groupby('participant_id').size().describe()}")

    # 5. Load surveys
    print("\n📋 Loading survey data...")
    phq9 = load_phq9()
    if not phq9.empty:
        phq9.to_csv(OUTPUT_PATH / "phq9_scores.csv", index=False)
        print(f"  ✓ PHQ-9: {len(phq9)} records")

    pss = load_pss()
    if not pss.empty:
        pss.to_csv(OUTPUT_PATH / "pss_scores.csv", index=False)
        print(f"  ✓ PSS: {len(pss)} records")

    print("\n✅ EMA pipeline complete!")


if __name__ == "__main__":
    main()
