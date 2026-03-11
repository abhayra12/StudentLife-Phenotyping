"""
Stress Prediction from Passive Sensor Data

Models: Random Forest & XGBoost
Target: EMA stress level (1–5 classification) and stress score (regression)

This is the core ML component: using only what the phone senses passively
(activity, audio, screen usage, WiFi, etc.) to predict self-reported stress.

Pipeline:
  1. Load merged sensor-EMA dataset (train/val/test splits)
  2. Feature engineering (add ratios, interaction features)
  3. Train Random Forest (classification + regression)
  4. Train XGBoost (classification + regression)
  5. Evaluate with accuracy, F1, MAE, and per-class metrics
  6. Feature importance analysis
  7. Save models and results
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost not installed, skipping XGBoost models")


# ──────────────────────── Configuration ────────────────────────
DATA_PATH = Path("data/processed")
MODEL_PATH = Path("models")
RESULTS_PATH = Path("reports/results")

# Columns to exclude from features
META_COLS = ["participant_id", "ema_timestamp", "stress_level", "stress_score", "stress_label"]

# Stress labels
STRESS_LABELS = {
    1: "A little stressed",
    2: "Definitely stressed",
    3: "Stressed out",
    4: "Feeling good",
    5: "Feeling great",
}


# ──────────────── Feature Engineering ──────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to improve model performance."""
    df = df.copy()

    # ── Ratio features ──
    # Screen time ratio: lock time vs charge time (phone use pattern)
    lock_sum = df.get("phonelock_minutes_sum", pd.Series(0))
    charge_sum = df.get("phonecharge_minutes_sum", pd.Series(0))
    df["screen_vs_charge_ratio"] = lock_sum / (charge_sum + 1)

    # Voice vs noise ratio (social engagement vs environment)
    voice_sum = df.get("audio_voice_minutes_sum", pd.Series(0))
    noise_sum = df.get("audio_noise_minutes_sum", pd.Series(0))
    df["voice_vs_noise_ratio"] = voice_sum / (noise_sum + 1)

    # Activity ratio (active vs total time)
    active_sum = df.get("activity_active_minutes_sum", pd.Series(0))
    df["activity_ratio"] = active_sum / (df.get("hours_of_data", 1) * 60 + 1)

    # Dark time ratio (potential sleep/rest)
    dark_sum = df.get("dark_minutes_sum", pd.Series(0))
    df["dark_ratio"] = dark_sum / (df.get("hours_of_data", 1) * 60 + 1)

    # ── Interaction features ──
    # Late night activity (hour 0-5 combined with phone usage)
    if "hour_of_day" in df.columns:
        df["is_late_night"] = (df["hour_of_day"] < 6).astype(int)
        df["late_night_phone"] = df["is_late_night"] * lock_sum

    # Conversation + social indicator
    conv_sum = df.get("conversation_minutes_sum", pd.Series(0))
    df["social_engagement"] = voice_sum + conv_sum

    # WiFi diversity (location variety indicator)
    wifi_mean = df.get("wifi_unique_aps_mean", pd.Series(0))
    wifi_max = df.get("wifi_unique_aps_max", pd.Series(0))
    df["wifi_variability"] = wifi_max - wifi_mean

    return df


# ──────────────── Model Training ───────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    """Get list of feature columns (everything except metadata)."""
    return [c for c in df.columns if c not in META_COLS]


def prepare_data(train_df: pd.DataFrame, val_df: pd.DataFrame,
                 test_df: pd.DataFrame) -> tuple:
    """Prepare X, y arrays with feature engineering and scaling."""

    # Engineer features
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    feature_cols = get_feature_cols(train_df)

    X_train = train_df[feature_cols].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values

    # Classification target: stress level (1-5)
    y_train_cls = train_df["stress_level"].values
    y_val_cls = val_df["stress_level"].values
    y_test_cls = test_df["stress_level"].values

    # Regression target: stress score (inverted, higher = more stressed)
    if "stress_score" in train_df.columns:
        y_train_reg = train_df["stress_score"].values
        y_val_reg = val_df["stress_score"].values
        y_test_reg = test_df["stress_score"].values
    else:
        y_train_reg = 6 - y_train_cls
        y_val_reg = 6 - y_val_cls
        y_test_reg = 6 - y_test_cls

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, X_val, X_test,
            y_train_cls, y_val_cls, y_test_cls,
            y_train_reg, y_val_reg, y_test_reg,
            feature_cols, scaler)


def train_random_forest(X_train, y_train_cls, y_train_reg,
                        X_val, y_val_cls, y_val_reg):
    """Train Random Forest for both classification and regression."""
    print("\n🌲 Training Random Forest...")

    # Classification
    rf_cls = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_cls.fit(X_train, y_train_cls)
    val_acc = accuracy_score(y_val_cls, rf_cls.predict(X_val))
    val_f1 = f1_score(y_val_cls, rf_cls.predict(X_val), average="weighted")
    print(f"  Classification: Val Accuracy={val_acc:.3f}, Val F1={val_f1:.3f}")

    # Regression
    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_reg.fit(X_train, y_train_reg)
    val_mae = mean_absolute_error(y_val_reg, rf_reg.predict(X_val))
    val_r2 = r2_score(y_val_reg, rf_reg.predict(X_val))
    print(f"  Regression:     Val MAE={val_mae:.3f}, Val R²={val_r2:.3f}")

    return rf_cls, rf_reg


def train_xgboost(X_train, y_train_cls, y_train_reg,
                  X_val, y_val_cls, y_val_reg):
    """Train XGBoost for both classification and regression."""
    if not HAS_XGBOOST:
        return None, None

    print("\n🚀 Training XGBoost...")

    # Classification (multi-class)
    xgb_cls = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softmax",
        num_class=5,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    # XGBoost expects 0-indexed classes
    xgb_cls.fit(X_train, y_train_cls - 1,
                eval_set=[(X_val, y_val_cls - 1)],
                verbose=False)
    val_pred = xgb_cls.predict(X_val) + 1
    val_acc = accuracy_score(y_val_cls, val_pred)
    val_f1 = f1_score(y_val_cls, val_pred, average="weighted")
    print(f"  Classification: Val Accuracy={val_acc:.3f}, Val F1={val_f1:.3f}")

    # Regression
    xgb_reg = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_reg.fit(X_train, y_train_reg,
                eval_set=[(X_val, y_val_reg)],
                verbose=False)
    val_mae = mean_absolute_error(y_val_reg, xgb_reg.predict(X_val))
    val_r2 = r2_score(y_val_reg, xgb_reg.predict(X_val))
    print(f"  Regression:     Val MAE={val_mae:.3f}, Val R²={val_r2:.3f}")

    return xgb_cls, xgb_reg


# ──────────────── Evaluation ───────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test, task: str = "classification",
                   offset: int = 0) -> dict:
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)
    if offset:
        y_pred = y_pred + offset

    if task == "classification":
        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average="weighted")
        f1_m = f1_score(y_test, y_pred, average="macro")
        print(f"\n{'='*50}")
        print(f"📊 {name} — Test Results")
        print(f"{'='*50}")
        print(f"  Accuracy:       {acc:.3f}")
        print(f"  F1 (weighted):  {f1_w:.3f}")
        print(f"  F1 (macro):     {f1_m:.3f}")
        print(f"\n  Classification Report:")
        report = classification_report(y_test, y_pred,
                                       target_names=[STRESS_LABELS[i] for i in sorted(STRESS_LABELS)],
                                       zero_division=0)
        print(report)

        cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])
        print(f"  Confusion Matrix:")
        print(f"  {cm}")

        return {
            "model": name,
            "task": "classification",
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1_w, 4),
            "f1_macro": round(f1_m, 4),
        }
    else:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"\n{'='*50}")
        print(f"📊 {name} — Test Results (Regression)")
        print(f"{'='*50}")
        print(f"  MAE:   {mae:.3f}")
        print(f"  RMSE:  {rmse:.3f}")
        print(f"  R²:    {r2:.3f}")

        return {
            "model": name,
            "task": "regression",
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        }


def get_feature_importance(model, feature_cols: list, name: str,
                           top_n: int = 15) -> pd.DataFrame:
    """Extract and display feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(f"\n🔍 Top {top_n} Features ({name}):")
    for i, row in fi.head(top_n).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"  {row['importance']:.3f} {bar} {row['feature']}")

    return fi


# ──────────────── Main Pipeline ────────────────────────────────

def main():
    print("=" * 60)
    print("🧠 Stress Prediction from Passive Sensor Data")
    print("   Random Forest + XGBoost")
    print("=" * 60)

    # 1. Load data
    print("\n📂 Loading data...")
    train_df = pd.read_csv(DATA_PATH / "train_stress.csv")
    val_df = pd.read_csv(DATA_PATH / "val_stress.csv")
    test_df = pd.read_csv(DATA_PATH / "test_stress.csv")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Prepare features
    print("\n⚙️ Preparing features...")
    (X_train, X_val, X_test,
     y_train_cls, y_val_cls, y_test_cls,
     y_train_reg, y_val_reg, y_test_reg,
     feature_cols, scaler) = prepare_data(train_df, val_df, test_df)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train shape: {X_train.shape}")

    # 3. Train models
    rf_cls, rf_reg = train_random_forest(
        X_train, y_train_cls, y_train_reg,
        X_val, y_val_cls, y_val_reg
    )

    xgb_cls, xgb_reg = train_xgboost(
        X_train, y_train_cls, y_train_reg,
        X_val, y_val_cls, y_val_reg
    )

    # 4. Evaluate on test set
    all_results = []

    # Random Forest
    results_rf_cls = evaluate_model("Random Forest", rf_cls, X_test, y_test_cls)
    results_rf_reg = evaluate_model("Random Forest", rf_reg, X_test, y_test_reg, "regression")
    all_results.extend([results_rf_cls, results_rf_reg])

    # XGBoost
    if xgb_cls is not None:
        results_xgb_cls = evaluate_model("XGBoost", xgb_cls, X_test, y_test_cls, offset=1)
        results_xgb_reg = evaluate_model("XGBoost", xgb_reg, X_test, y_test_reg, "regression")
        all_results.extend([results_xgb_cls, results_xgb_reg])

    # 5. Feature importance
    fi_rf = get_feature_importance(rf_cls, feature_cols, "Random Forest")
    if xgb_cls is not None:
        fi_xgb = get_feature_importance(xgb_cls, feature_cols, "XGBoost")

    # 6. Cross-validation (on combined train+val)
    print("\n📊 Cross-validation (5-fold)...")
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train_cls, y_val_cls])

    cv_scores_rf = cross_val_score(
        RandomForestClassifier(n_estimators=200, max_depth=15,
                               class_weight="balanced", random_state=42, n_jobs=-1),
        X_trainval, y_trainval, cv=5, scoring="accuracy"
    )
    print(f"  RF 5-fold CV: {cv_scores_rf.mean():.3f} ± {cv_scores_rf.std():.3f}")

    if HAS_XGBOOST:
        cv_scores_xgb = cross_val_score(
            XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                          random_state=42, n_jobs=-1, verbosity=0,
                          objective="multi:softmax", num_class=5),
            X_trainval, y_trainval - 1, cv=5, scoring="accuracy"
        )
        print(f"  XGB 5-fold CV: {cv_scores_xgb.mean():.3f} ± {cv_scores_xgb.std():.3f}")

    # 7. Save results
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Save models
    with open(MODEL_PATH / "rf_stress_classifier.pkl", "wb") as f:
        pickle.dump(rf_cls, f)
    with open(MODEL_PATH / "rf_stress_regressor.pkl", "wb") as f:
        pickle.dump(rf_reg, f)
    if xgb_cls is not None:
        with open(MODEL_PATH / "xgb_stress_classifier.pkl", "wb") as f:
            pickle.dump(xgb_cls, f)
        with open(MODEL_PATH / "xgb_stress_regressor.pkl", "wb") as f:
            pickle.dump(xgb_reg, f)
    with open(MODEL_PATH / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n💾 Models saved to {MODEL_PATH}/")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_PATH / "stress_prediction_results.csv", index=False)

    # Save feature importance
    if not fi_rf.empty:
        fi_rf.to_csv(RESULTS_PATH / "feature_importance_rf.csv", index=False)
    if xgb_cls is not None and not fi_xgb.empty:
        fi_xgb.to_csv(RESULTS_PATH / "feature_importance_xgb.csv", index=False)

    # Save summary JSON
    summary = {
        "dataset": {
            "total_samples": len(train_df) + len(val_df) + len(test_df),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "participants": int(train_df["participant_id"].nunique()),
            "features": len(feature_cols),
            "lookback_hours": 6,
        },
        "results": all_results,
        "feature_cols": feature_cols[:20],
    }
    with open(RESULTS_PATH / "stress_prediction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📊 Results saved to {RESULTS_PATH}/")
    print("\n✅ Stress prediction pipeline complete!")


if __name__ == "__main__":
    main()
