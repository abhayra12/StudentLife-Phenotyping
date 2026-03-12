"""
SOTA Stress Prediction — State-of-the-Art Tabular ML

Pushes beyond the 10-algorithm baseline in stress_prediction.py using
three complementary advances:

  1. CatBoost + Optuna Bayesian HPO
       - Native class-imbalance handling (auto_class_weights)
       - Ordered boosting (unlike standard GBDT, prevents target leakage)
       - Optuna searches 60 trials over the full hyperparameter space
       - Expected lift: +2–4% accuracy over untuned CatBoost

  3. Two-Level Stacked Ensemble → Soft-Voting Ensemble
       - Trains RF, XGBoost, LightGBM, CatBoost, ExtraTrees on full training set
       - Averages predicted class probabilities (soft voting)
       - Robust to temporal distribution shift (no OOF leakage across time boundary)
       - Expected to match or exceed best single model on chronological splits

  3. SHAP Feature Importance
       - TreeSHAP for exact Shapley values (not permutation approximations)
       - Identifies which sensor signals drive stress predictions
       - Publication-ready summary and bar plots

Pipeline:
  1. Load merged sensor-EMA dataset
  2. Feature engineering (ratios, interactions, cyclical encoding)
  3. CatBoost with Optuna HPO → save best config + model
  4. Stacked ensemble (OOF meta-features) → meta-learner prediction
  5. Compare all SOTA models vs baseline (from stress_prediction.py)
  6. SHAP analysis on best model
  7. Save results, models, and figures

Dataset: 2,154 samples × 54 features → 5-class stress classification
         Stress distribution: 45% level-1, 21% level-4, 16% level-2,
                              12% level-3, 6% level-5 (imbalanced)
"""

import json
import pickle
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Optional imports ────────────────────────────────────────────────────────

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠  XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠  LightGBM not available")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("⚠  CatBoost not available — install with: pip install catboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠  Optuna not available — install with: pip install optuna")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠  SHAP not available — install with: pip install shap")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ── Paths ───────────────────────────────────────────────────────────────────

DATA_PATH    = Path("data/processed")
MODEL_PATH   = Path("models")
RESULTS_PATH = Path("reports/results")
FIGURES_PATH = Path("reports/figures/modeling")

for p in [MODEL_PATH, RESULTS_PATH, FIGURES_PATH]:
    p.mkdir(parents=True, exist_ok=True)

META_COLS = [
    "participant_id", "ema_timestamp",
    "stress_level", "stress_score", "stress_label",
]

RANDOM_STATE = 42
N_SPLITS     = 5      # StratifiedKFold splits
OPTUNA_TRIALS = 60    # Bayesian HPO trials


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Loading & Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load merged sensor-EMA data and return (df, X, y, feature_names)."""

    csv_path = DATA_PATH / "sensor_ema_merged.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Merged dataset not found at {csv_path}. "
            "Run: python src/data/merge_sensor_ema.py"
        )

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} samples × {df.shape[1]} cols "
          f"from {csv_path.name}")

    # Target
    y = df["stress_level"].values  # 1–5

    # Feature columns (exclude metadata and targets)
    meta_cols_present = [c for c in META_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols_present]

    X_df = df[feature_cols].copy()

    # ── Feature engineering ──────────────────────────────────────────────────

    # Cyclical time encoding
    if "hour_of_day" in X_df.columns:
        X_df["hour_sin"] = np.sin(2 * np.pi * X_df["hour_of_day"] / 24)
        X_df["hour_cos"] = np.cos(2 * np.pi * X_df["hour_of_day"] / 24)

    if "day_of_week" in X_df.columns:
        X_df["dow_sin"] = np.sin(2 * np.pi * X_df["day_of_week"] / 7)
        X_df["dow_cos"] = np.cos(2 * np.pi * X_df["day_of_week"] / 7)

    # Activity ratio (active vs total) — captures relative mobility
    act_sum = X_df.get("activity_active_minutes_sum", pd.Series(0, index=X_df.index))
    unk_sum = X_df.get("activity_unknown_minutes_sum", pd.Series(0, index=X_df.index))
    total = act_sum + unk_sum + 1e-8
    X_df["active_ratio"] = act_sum / total

    # Conversation density per hour of data
    conv_sum = X_df.get("conversation_minutes_sum", pd.Series(0, index=X_df.index))
    hours    = df.get("hours_of_data", pd.Series(1, index=df.index)).clip(lower=1)
    X_df["conv_per_hour"] = conv_sum / hours.values

    # Phone lock intensity (lock count / lock minutes) — fragmented usage proxy
    lock_count = X_df.get("phonelock_count_sum", pd.Series(0, index=X_df.index))
    lock_mins  = X_df.get("phonelock_minutes_sum", pd.Series(1, index=X_df.index)).clip(lower=1)
    X_df["lock_intensity"] = lock_count / lock_mins

    # Audio social ratio: voice / (voice + noise)
    voice = X_df.get("audio_voice_minutes_sum", pd.Series(0, index=X_df.index))
    noise = X_df.get("audio_noise_minutes_sum", pd.Series(0, index=X_df.index))
    X_df["social_audio_ratio"] = voice / (voice + noise + 1e-8)

    # Fill NaN with median
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float32)

    print(f"  Feature matrix: {X.shape[0]:,} × {X.shape[1]} "
          f"(+{len(feature_names) - len(feature_cols)} engineered features)")
    return df, X, y, feature_names


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """Chronological train / val / test split (no shuffle to prevent leakage)."""
    n = len(X)
    test_start = int(n * (1 - test_frac))
    val_start  = int(n * (1 - test_frac - val_frac))

    X_train = X[:val_start];  y_train = y[:val_start]
    X_val   = X[val_start:test_start]; y_val = y[val_start:test_start]
    X_test  = X[test_start:]; y_test = y[test_start:]

    print(f"  Split → train {len(X_train):,}  val {len(X_val):,}  "
          f"test {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 2. CatBoost with Optuna Bayesian HPO
# ══════════════════════════════════════════════════════════════════════════════

def optuna_catboost_hpo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = OPTUNA_TRIALS,
) -> dict:
    """
    Search CatBoost hyperparameter space with Optuna (TPE sampler + median pruner).
    Returns best hyperparameter dict.
    """
    if not HAS_CAT or not HAS_OPTUNA:
        print("  ⚠ Skipping Optuna HPO (CatBoost or Optuna not installed)")
        return {}

    # Use cached HPO results if already run (saves ~15 min on reruns)
    params_path = RESULTS_PATH / "catboost_best_params.json"
    if params_path.exists():
        with open(params_path) as f:
            cached = json.load(f)
        print(f"  ✓ Loaded cached HPO params from {params_path}")
        print(f"    depth={cached.get('depth')}, lr={cached.get('learning_rate', 0):.4f}, "
              f"iter={cached.get('iterations')}")
        return cached

    print(f"\n[Optuna] Searching {n_trials} trials over CatBoost HPO space ...")

    classes = np.unique(y_train)
    n_classes = len(classes)

    def objective(trial: optuna.Trial) -> float:
        bootstrap_type = trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        )
        params = {
            "iterations":        trial.suggest_int("iterations", 200, 600),
            "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "depth":             trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "min_data_in_leaf":  trial.suggest_int("min_data_in_leaf", 1, 40),
            "random_strength":   trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "bootstrap_type":    bootstrap_type,
            "auto_class_weights": "Balanced",
            "eval_metric":       "Accuracy",
            "random_seed":       RANDOM_STATE,
            "verbose":           0,
            "thread_count":      -1,
        }
        # subsample is only valid for Bernoulli / MVS; Bayesian uses bagging_temperature
        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0.0, 1.0
            )
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            model = CatBoostClassifier(**params)
            model.fit(X_train[tr_idx], y_train[tr_idx],
                      eval_set=(X_train[va_idx], y_train[va_idx]),
                      early_stopping_rounds=30)
            preds = model.predict(X_train[va_idx])
            scores.append(accuracy_score(y_train[va_idx], preds))

            trial.report(np.mean(scores), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score  = study.best_value
    print(f"  Best CV accuracy: {best_score:.4f}  |  "
          f"params: depth={best_params.get('depth')}, "
          f"lr={best_params.get('learning_rate', 0):.4f}, "
          f"iter={best_params.get('iterations')}")

    # Persist best params
    params_path = RESULTS_PATH / "catboost_best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved HPO results → {params_path}")

    return best_params


def train_tuned_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: dict,
) -> tuple[float, float, object]:
    """Train CatBoost with Optuna-tuned params and evaluate."""

    if not HAS_CAT:
        return 0.0, 0.0, None

    params = {
        **best_params,
        "auto_class_weights": "Balanced",
        "random_seed":        RANDOM_STATE,
        "verbose":            0,
        "thread_count":       -1,
    }
    # Remove optuna-specific keys not in CatBoost API
    params.pop("eval_metric", None)

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test).flatten()
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average="weighted")

    # CV score
    cv_model = CatBoostClassifier(**params)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for tr, va in skf.split(X_train, y_train):
        cv_model.fit(X_train[tr], y_train[tr])
        cv_scores.append(accuracy_score(y_train[va], cv_model.predict(X_train[va]).flatten()))

    print(f"  CatBoost (Optuna) — Test acc: {acc:.4f}  "
          f"F1: {f1:.4f}  CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    return acc, f1, model


# ══════════════════════════════════════════════════════════════════════════════
# 3. Two-Level Stacked Ensemble
# ══════════════════════════════════════════════════════════════════════════════

def build_base_models() -> dict:
    """Return L1 base learners — diverse, class-balanced."""
    models = {
        "rf":  RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "et":  ExtraTreesClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    if HAS_LGB:
        models["lgb"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE, verbose=-1,
        )
    if HAS_CAT:
        models["cat"] = CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=7,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE, verbose=0, thread_count=-1,
        )
    return models


def stacked_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, dict, None]:
    """
    Soft-voting ensemble: train all base learners on the full training set
    and average their predicted class probabilities.

    Chosen over OOF two-level stacking because the dataset uses a chronological
    train/test split (temporally ordered).  OOF stacking generates meta-features
    by randomly shuffling folds of the training period, which are OOD relative
    to the test period → the meta-learner learns fold artifacts that don't
    transfer, yielding sub-random test accuracy.

    Soft voting has no such leakage: each base model sees the full training
    time-window and its probability output is directly averaged, making the
    ensemble robust to temporal distribution shift.

    Returns (accuracy, f1, trained_models_dict, None).
    """
    from sklearn.base import clone as sk_clone

    base_models = build_base_models()
    n_classes   = len(np.unique(y_train))

    # Encode labels to 0-indexed (required by XGBoost; consistent for all models)
    le          = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    print(f"\n[Ensemble] Soft-voting over: {list(base_models.keys())}")

    avg_probs = np.zeros((len(X_test), n_classes))
    trained   = {}

    for name, model in base_models.items():
        m = sk_clone(model)
        m.fit(X_train, y_train_enc)
        trained[name] = m

        # Reindex probabilities in case a class is absent from training set
        trained_cls = getattr(m, "classes_", np.arange(n_classes))
        raw  = m.predict_proba(X_test)
        full = np.zeros((len(X_test), n_classes))
        for j, cls in enumerate(trained_cls):
            full[:, int(cls)] = raw[:, j]

        avg_probs += full / len(base_models)
        print(f"    ✓ {name}", end="\r")

    print(f"    All {len(base_models)} base models trained.       ")

    # Decode predictions back to original label space
    preds_enc = np.argmax(avg_probs, axis=1)
    preds     = le.inverse_transform(preds_enc)
    y_test_enc = le.transform(y_test)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="weighted")

    # Estimate variance via bootstrap (no cross-val needed for soft voting)
    rng = np.random.default_rng(RANDOM_STATE)
    boot_accs = []
    for _ in range(200):
        idx = rng.choice(len(y_test), len(y_test), replace=True)
        boot_accs.append(accuracy_score(y_test[idx], preds[idx]))
    boot_mean = float(np.mean(boot_accs))
    boot_std  = float(np.std(boot_accs))

    print(f"\n  Soft Voting Ensemble — Test acc: {acc:.4f}  "
          f"F1: {f1:.4f}  Bootstrap: {boot_mean:.4f} ± {boot_std:.4f}")

    return acc, f1, trained, None


# ══════════════════════════════════════════════════════════════════════════════
# 4. SHAP Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

def run_shap_analysis(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    model_name: str = "CatBoost",
) -> None:
    """
    Feature attribution via CatBoost's native SHAP implementation.

    Uses CatBoost's C++ get_feature_importance(type='ShapValues') directly —
    same algorithm as TreeSHAP but without routing through the shap Python
    library, which segfaults on multiclass CatBoost models in some envs.
    Falls back to built-in feature importance if SHAP computation fails.
    """
    if not HAS_PLOT:
        return

    print(f"\n[SHAP] Computing feature attribution for {model_name} ...")

    mean_abs_shap: np.ndarray | None = None
    shap_matrix:   np.ndarray | None = None  # (n_samples, n_features)

    # ── Strategy 1: CatBoost native SHAP (no shap library) ──────────────────
    if HAS_CAT:
        try:
            from catboost import Pool  # already imported by model; safe re-import
            n_samples = min(150, len(X_test))
            pool      = Pool(X_test[:n_samples])
            raw_shap  = model.get_feature_importance(pool, type="ShapValues")
            # Shapes CatBoost returns:
            #   binary    → (n_samples, n_features + 1)
            #   multiclass → (n_classes, n_samples, n_features + 1)
            if raw_shap.ndim == 3:                         # multiclass
                # drop bias (last col), average |SHAP| across classes
                shap_matrix   = np.abs(raw_shap[:, :, :len(feature_names)]).mean(axis=0)
                mean_abs_shap = shap_matrix.mean(axis=0)
                shap_matrix   = raw_shap[0, :, :len(feature_names)]  # class-0 for scatter
            elif raw_shap.ndim == 2:                       # binary or regression
                shap_matrix   = raw_shap[:, :len(feature_names)]
                mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
            else:
                raise ValueError(f"Unexpected SHAP shape: {raw_shap.shape}")

            print(f"  ✓ CatBoost native SHAP: {n_samples} samples × "
                  f"{len(feature_names)} features")

        except Exception as e:
            print(f"  ⚠ Native SHAP failed ({type(e).__name__}: {e})")
            mean_abs_shap = None
            shap_matrix   = None

    # ── Strategy 2: Built-in feature importance (PredictionValuesChange) ────
    if mean_abs_shap is None:
        try:
            importances   = np.array(model.get_feature_importance())
            mean_abs_shap = importances[:len(feature_names)]
            print("  ✓ Using built-in CatBoost feature importance (PredictionValuesChange)")
        except Exception as e2:
            print(f"  ⚠ Feature importance also failed: {e2}")
            return

    # ── Build Series and plots ───────────────────────────────────────────────
    feature_importance = pd.Series(
        mean_abs_shap, index=feature_names[:len(mean_abs_shap)]
    ).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left — Top-20 bar chart
    top20 = feature_importance.head(20)
    axes[0].barh(top20.index[::-1], top20.values[::-1], color="#2196F3")
    axes[0].set_xlabel("Mean |SHAP| value (contribution to prediction)")
    axes[0].set_title(f"Top-20 Features — {model_name} (TreeSHAP)", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)

    # Right — scatter of SHAP values for top-10 (if matrix available)
    top10_names = list(feature_importance.head(10).index)
    if shap_matrix is not None:
        top10_idx = [feature_names.index(n) for n in top10_names if n in feature_names]
        rng = np.random.default_rng(42)
        for i, col_idx in enumerate(top10_idx):
            vals = shap_matrix[:, col_idx]
            jitter = rng.standard_normal(len(vals)) * 0.06
            axes[1].scatter(vals, np.full(len(vals), i) + jitter,
                            alpha=0.35, s=12, c="#FF5722")
        axes[1].set_yticks(range(len(top10_idx)))
        axes[1].set_yticklabels(top10_names[:len(top10_idx)])
        axes[1].set_xlabel("SHAP value (direction and magnitude)")
        axes[1].set_title("SHAP Distribution — Top-10 Features", fontweight="bold")
        axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].grid(axis="x", alpha=0.3)
    else:
        # Fallback: second bar chart for top-10
        top10 = feature_importance.head(10)
        axes[1].barh(top10.index[::-1], top10.values[::-1], color="#FF5722")
        axes[1].set_xlabel("Feature Importance Score")
        axes[1].set_title("Top-10 Feature Importance", fontweight="bold")
        axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_PATH / "sota_shap_importance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature attribution plot → {out_path}")
    print(f"  Top-5 features: {', '.join(feature_importance.head(5).index.tolist())}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Comparison Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_sota_comparison(results: list[dict]) -> None:
    """Bar chart comparing all SOTA models vs baselines."""
    if not HAS_PLOT or not results:
        return

    df = pd.DataFrame(results).sort_values("accuracy", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = ["#4CAF50" if "SOTA" in r or "Stacked" in r or "Optuna" in r
              else "#2196F3" for r in df["model"]]

    # Accuracy
    bars = axes[0].barh(df["model"], df["accuracy"] * 100, color=colors)
    axes[0].set_xlabel("Test Accuracy (%)")
    axes[0].set_title("Stress Prediction Accuracy — SOTA vs Baseline", fontweight="bold")
    axes[0].axvline(x=20, color="red", linestyle="--", alpha=0.5, label="Random (5-class)")
    axes[0].legend()
    axes[0].grid(axis="x", alpha=0.3)
    for bar, row in zip(bars, df.itertuples()):
        axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{row.accuracy*100:.1f}%", va="center", fontsize=9)

    # F1
    bars2 = axes[1].barh(df["model"], df["f1"] * 100, color=colors)
    axes[1].set_xlabel("Weighted F1 Score (%)")
    axes[1].set_title("Weighted F1 Score — SOTA vs Baseline", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)
    for bar, row in zip(bars2, df.itertuples()):
        axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{row.f1*100:.1f}%", va="center", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="SOTA (this module)"),
        Patch(facecolor="#2196F3", label="Baseline (stress_prediction.py)"),
    ]
    axes[0].legend(handles=legend_elements + [
        plt.Line2D([0], [0], color="red", linestyle="--", label="Random (20%)")
    ])

    plt.tight_layout()
    out = FIGURES_PATH / "sota_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  SOTA comparison chart → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Save Results
# ══════════════════════════════════════════════════════════════════════════════

def to_native(val):
    """Convert numpy scalars to native Python types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    return val


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    return to_native(obj)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Main Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "═" * 65)
    print("  SOTA Stress Prediction — CatBoost HPO + Stacked Ensemble")
    print("═" * 65)

    # ── Load & split ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading data ...")
    df, X, y, feature_names = load_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X_scaled, y)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    n_classes = len(np.unique(y))
    print(f"  Classes: {sorted(np.unique(y))}  |  "
          f"Imbalance: {np.bincount(y)[1:].max() / np.bincount(y)[1:].min():.1f}x")

    # ── Baseline: plain CatBoost (no HPO) ────────────────────────────────────
    results = []

    if HAS_CAT:
        print("\n[1b] Baseline CatBoost (default params) ...")
        cat_base = CatBoostClassifier(
            iterations=300, learning_rate=0.1, depth=7,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE, verbose=0, thread_count=-1,
        )
        cat_base.fit(X_train_full, y_train_full)
        preds_base = cat_base.predict(X_test).flatten()
        acc_base = accuracy_score(y_test, preds_base)
        f1_base  = f1_score(y_test, preds_base, average="weighted")
        print(f"  CatBoost (default) — acc: {acc_base:.4f}  f1: {f1_base:.4f}")
        results.append({"model": "CatBoost (default)", "accuracy": acc_base, "f1": f1_base})

    # ── Optuna HPO ───────────────────────────────────────────────────────────
    print("\n[2/4] CatBoost + Optuna HPO ...")
    t0 = time()
    best_params = optuna_catboost_hpo(X_train_full, y_train_full, n_trials=OPTUNA_TRIALS)
    hpo_time    = time() - t0

    best_catboost_model = None
    if best_params and HAS_CAT:
        acc_hpo, f1_hpo, best_catboost_model = train_tuned_catboost(
            X_train_full, y_train_full, X_test, y_test, best_params
        )
        results.append({"model": "CatBoost + Optuna HPO (SOTA)", "accuracy": acc_hpo, "f1": f1_hpo})
        print(f"  HPO search time: {hpo_time/60:.1f} min")

    # ── Stacked Ensemble ─────────────────────────────────────────────────────
    print("\n[3/4] Soft Voting Ensemble (RF + ET + XGB + LGB + CatBoost) ...")
    t0 = time()
    acc_stack, f1_stack, ensemble_models, _ = stacked_ensemble(
        X_train_full, y_train_full, X_test, y_test
    )
    stack_time = time() - t0
    results.append({"model": "Soft Voting Ensemble (SOTA)", "accuracy": acc_stack, "f1": f1_stack})
    print(f"  Ensemble training time: {stack_time/60:.1f} min")

    # ── SHAP analysis (on tuned CatBoost) ────────────────────────────────────
    print("\n[4/4] SHAP feature importance ...")
    if best_catboost_model is not None:
        run_shap_analysis(
            best_catboost_model, X_train_full, X_test,
            feature_names, model_name="CatBoost (Optuna)"
        )

    # ── Print final comparison ────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  SOTA Results Summary")
    print("─" * 65)
    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    for _, row in results_df.iterrows():
        marker = " ★ BEST" if row["accuracy"] == results_df["accuracy"].max() else ""
        print(f"  {row['model']:<40}  acc: {row['accuracy']:.4f}  "
              f"f1: {row['f1']:.4f}{marker}")
    print("─" * 65)

    # Previous baseline (from stress_prediction.py) for comparison
    baseline_results = [
        {"model": "MLP Neural Network",  "accuracy": 0.4120, "f1": 0.3890},
        {"model": "Random Forest",       "accuracy": 0.3971, "f1": 0.3745},
        {"model": "Extra Trees",         "accuracy": 0.3971, "f1": 0.3812},
        {"model": "XGBoost",             "accuracy": 0.3876, "f1": 0.3621},
        {"model": "LightGBM",            "accuracy": 0.3829, "f1": 0.3598},
    ]
    all_results = baseline_results + results

    # ── Save models ───────────────────────────────────────────────────────────
    if best_catboost_model is not None:
        model_path = MODEL_PATH / "catboost_optuna_stress.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_catboost_model, f)
        print(f"\n  Saved CatBoost model → {model_path}")

    if ensemble_models is not None:
        model_path = MODEL_PATH / "soft_voting_ensemble_stress.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(ensemble_models, f)
        print(f"  Saved Soft Voting Ensemble → {model_path}")

    # Save scaler
    scaler_path = MODEL_PATH / "sota_stress_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # ── Save results JSON ────────────────────────────────────────────────────
    best_row  = results_df.iloc[0]
    best_base = max(baseline_results, key=lambda r: r["accuracy"])

    summary = {
        "sota_results": clean_for_json(results),
        "best_sota_model": {
            "name":     best_row["model"],
            "accuracy": float(best_row["accuracy"]),
            "f1":       float(best_row["f1"]),
        },
        "improvement_over_baseline_mlp": {
            "accuracy_delta": float(best_row["accuracy"] - 0.4120),
            "pct_improvement": float((best_row["accuracy"] - 0.4120) / 0.4120 * 100),
        },
        "random_baseline": 1.0 / n_classes,
        "n_samples": int(len(y_test)),
        "n_features": int(X.shape[1]),
        "n_classes": int(n_classes),
        "optuna_trials": OPTUNA_TRIALS,
        "hpo_time_min": round(float(hpo_time / 60), 2),
        "stack_time_min": round(float(stack_time / 60), 2),
    }

    results_path = RESULTS_PATH / "sota_stress_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full results saved → {results_path}")

    # ── Save results CSV ──────────────────────────────────────────────────────
    csv_path = RESULTS_PATH / "sota_stress_comparison.csv"
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"  Comparison CSV    → {csv_path}")

    # ── Generate comparison plot ──────────────────────────────────────────────
    plot_sota_comparison(all_results)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    best_acc  = best_row["accuracy"]
    delta     = best_acc - 0.4120
    pct       = delta / 0.4120 * 100
    random_b  = 1.0 / n_classes

    print(f"  Best SOTA model:  {best_row['model']}")
    print(f"  Test accuracy:    {best_acc:.4f} ({best_acc*100:.1f}%)")
    print(f"  vs MLP baseline:  {'+' if delta >= 0 else ''}{delta*100:.2f}pp "
          f"({'+' if pct >= 0 else ''}{pct:.1f}%)")
    print(f"  vs random (1/5):  {best_acc / random_b:.2f}x better")
    print("═" * 65)


if __name__ == "__main__":
    main()
