"""
Stress Prediction from Passive Sensor Data

Comprehensive model comparison: 10 supervised + unsupervised clustering
Target: EMA stress level (1-5 classification) and stress score (regression)

This is the core ML component: using only what the phone senses passively
(activity, audio, screen usage, WiFi, etc.) to predict self-reported stress.

Supervised Models:
  1. Random Forest           6. AdaBoost
  2. XGBoost                 7. Logistic Regression
  3. LightGBM                8. SVM (RBF kernel)
  4. Extra Trees             9. K-Nearest Neighbors
  5. Gradient Boosting       10. MLP Neural Network

Unsupervised Models:
  - K-Means Clustering (k=3, k=5)
  - Gaussian Mixture Model (k=3, k=5)
  - DBSCAN (density-based)

Pipeline:
  1. Load merged sensor-EMA dataset (train/val/test splits)
  2. Feature engineering (ratios, interactions)
  3. Train all supervised models (classification + regression)
  4. Run unsupervised clustering analysis
  5. Evaluate with accuracy, F1, MAE, per-class metrics
  6. Feature importance analysis
  7. Save models, results, and comparison figures
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# Optional imports
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost not installed, skipping")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠ LightGBM not installed, skipping")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# ──────────────────────── Configuration ────────────────────────
DATA_PATH = Path("data/processed")
MODEL_PATH = Path("models")
RESULTS_PATH = Path("reports/results")
FIGURES_PATH = Path("reports/figures/modeling")

META_COLS = [
    "participant_id", "ema_timestamp",
    "stress_level", "stress_score", "stress_label",
]

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

    lock_sum = df.get("phonelock_minutes_sum", pd.Series(0))
    charge_sum = df.get("phonecharge_minutes_sum", pd.Series(0))
    df["screen_vs_charge_ratio"] = lock_sum / (charge_sum + 1)

    voice_sum = df.get("audio_voice_minutes_sum", pd.Series(0))
    noise_sum = df.get("audio_noise_minutes_sum", pd.Series(0))
    df["voice_vs_noise_ratio"] = voice_sum / (noise_sum + 1)

    active_sum = df.get("activity_active_minutes_sum", pd.Series(0))
    df["activity_ratio"] = active_sum / (df.get("hours_of_data", 1) * 60 + 1)

    dark_sum = df.get("dark_minutes_sum", pd.Series(0))
    df["dark_ratio"] = dark_sum / (df.get("hours_of_data", 1) * 60 + 1)

    if "hour_of_day" in df.columns:
        df["is_late_night"] = (df["hour_of_day"] < 6).astype(int)
        df["late_night_phone"] = df["is_late_night"] * lock_sum

    conv_sum = df.get("conversation_minutes_sum", pd.Series(0))
    df["social_engagement"] = voice_sum + conv_sum

    wifi_mean = df.get("wifi_unique_aps_mean", pd.Series(0))
    wifi_max = df.get("wifi_unique_aps_max", pd.Series(0))
    df["wifi_variability"] = wifi_max - wifi_mean

    return df


# ──────────────── Data Preparation ─────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META_COLS]


def prepare_data(train_df, val_df, test_df):
    """Prepare X, y arrays with feature engineering and scaling."""
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    feature_cols = get_feature_cols(train_df)

    X_train = train_df[feature_cols].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values

    y_train_cls = train_df["stress_level"].values
    y_val_cls = val_df["stress_level"].values
    y_test_cls = test_df["stress_level"].values

    if "stress_score" in train_df.columns:
        y_train_reg = train_df["stress_score"].values
        y_val_reg = val_df["stress_score"].values
        y_test_reg = test_df["stress_score"].values
    else:
        y_train_reg = 6 - y_train_cls
        y_val_reg = 6 - y_val_cls
        y_test_reg = 6 - y_test_cls

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, X_val, X_test,
            y_train_cls, y_val_cls, y_test_cls,
            y_train_reg, y_val_reg, y_test_reg,
            feature_cols, scaler)


# ──────────────── Supervised Model Definitions ─────────────────

def get_supervised_models():
    """Return dict of {name: (classifier, regressor)} for all algorithms."""
    models = {}

    models["Random Forest"] = (
        RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1),
        RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1),
    )

    if HAS_XGBOOST:
        models["XGBoost"] = (
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                objective="multi:softmax", num_class=5,
                eval_metric="mlogloss", random_state=42,
                n_jobs=-1, verbosity=0),
            XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbosity=0),
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = (
            LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", num_leaves=31,
                random_state=42, n_jobs=-1, verbose=-1),
            LGBMRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                num_leaves=31, random_state=42, n_jobs=-1, verbose=-1),
        )

    models["Extra Trees"] = (
        ExtraTreesClassifier(
            n_estimators=200, max_depth=15,
            class_weight="balanced", random_state=42, n_jobs=-1),
        ExtraTreesRegressor(
            n_estimators=200, max_depth=15,
            random_state=42, n_jobs=-1),
    )

    models["Gradient Boosting"] = (
        GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42),
        GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42),
    )

    models["AdaBoost"] = (
        AdaBoostClassifier(
            n_estimators=100, learning_rate=0.5,
            estimator=DecisionTreeClassifier(max_depth=3),
            random_state=42),
        AdaBoostRegressor(
            n_estimators=100, learning_rate=0.5, random_state=42),
    )

    models["Logistic Regression"] = (
        LogisticRegression(
            max_iter=1000, class_weight="balanced",
            multi_class="multinomial", solver="lbfgs",
            random_state=42),
        Ridge(alpha=1.0),
    )

    models["SVM"] = (
        SVC(kernel="rbf", C=10, gamma="scale",
            class_weight="balanced", random_state=42),
        SVR(kernel="rbf", C=10, gamma="scale"),
    )

    models["KNN"] = (
        KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
        KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1),
    )

    models["MLP"] = (
        MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            random_state=42),
        MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            random_state=42),
    )

    return models


# ──────────────── Supervised Training & Eval ───────────────────

def train_and_evaluate_supervised(models, X_train, y_train_cls, y_train_reg,
                                  X_val, y_val_cls, y_val_reg,
                                  X_test, y_test_cls, y_test_reg):
    """Train all supervised models and evaluate on test set."""
    all_results = []
    trained_models = {}

    for name, (cls_model, reg_model) in models.items():
        print(f"\n{'─'*55}")
        print(f"  Training: {name}")
        print(f"{'─'*55}")

        # ── Classification ──
        try:
            if name == "XGBoost":
                cls_model.fit(X_train, y_train_cls - 1,
                              eval_set=[(X_val, y_val_cls - 1)], verbose=False)
            elif name == "LightGBM":
                cls_model.fit(X_train, y_train_cls,
                              eval_set=[(X_val, y_val_cls)])
            else:
                cls_model.fit(X_train, y_train_cls)

            if name == "XGBoost":
                y_pred_cls = cls_model.predict(X_test) + 1
            else:
                y_pred_cls = cls_model.predict(X_test)

            acc = accuracy_score(y_test_cls, y_pred_cls)
            f1_w = f1_score(y_test_cls, y_pred_cls, average="weighted")
            f1_m = f1_score(y_test_cls, y_pred_cls, average="macro")

            print(f"  Classification → Acc={acc:.3f}, F1(w)={f1_w:.3f}, F1(m)={f1_m:.3f}")

            all_results.append({
                "model": name, "task": "classification",
                "accuracy": round(acc, 4),
                "f1_weighted": round(f1_w, 4),
                "f1_macro": round(f1_m, 4),
            })
            trained_models[f"{name}_cls"] = cls_model

        except Exception as e:
            print(f"  ⚠ Classification failed: {e}")

        # ── Regression ──
        try:
            if name == "XGBoost":
                reg_model.fit(X_train, y_train_reg,
                              eval_set=[(X_val, y_val_reg)], verbose=False)
            elif name == "LightGBM":
                reg_model.fit(X_train, y_train_reg,
                              eval_set=[(X_val, y_val_reg)])
            else:
                reg_model.fit(X_train, y_train_reg)

            y_pred_reg = reg_model.predict(X_test)
            mae = mean_absolute_error(y_test_reg, y_pred_reg)
            rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
            r2 = r2_score(y_test_reg, y_pred_reg)

            print(f"  Regression     → MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

            all_results.append({
                "model": name, "task": "regression",
                "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4),
            })
            trained_models[f"{name}_reg"] = reg_model

        except Exception as e:
            print(f"  ⚠ Regression failed: {e}")

    return all_results, trained_models


# ──────────────── Unsupervised Analysis ────────────────────────

def run_unsupervised_analysis(X_train, y_train_cls, X_test, y_test_cls,
                              feature_cols):
    """Run clustering and evaluate cluster-stress alignment."""
    print(f"\n{'='*55}")
    print("  Unsupervised Learning: Clustering Analysis")
    print(f"{'='*55}")

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train_cls, y_test_cls])
    results = []

    # ── PCA ──
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all)
    print(f"\n  PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # ── K-Means ──
    for k in [3, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(X_all)
        sil = silhouette_score(X_all, clusters)

        cluster_map = {}
        for c in range(k):
            mask = clusters == c
            if mask.sum() > 0:
                cluster_map[c] = pd.Series(y_all[mask]).mode().iloc[0]

        mapped = np.array([cluster_map[c] for c in clusters])
        test_preds = mapped[len(X_train):]
        acc = accuracy_score(y_test_cls, test_preds)

        print(f"\n  K-Means (k={k}): Silhouette={sil:.3f}, Mapped Acc={acc:.3f}")
        results.append({
            "model": f"K-Means (k={k})", "task": "clustering",
            "silhouette": round(sil, 4), "mapped_accuracy": round(acc, 4),
        })

    # ── GMM ──
    for k in [3, 5]:
        gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200)
        gmm.fit(X_all)
        clusters = gmm.predict(X_all)
        sil = silhouette_score(X_all, clusters)

        cluster_map = {}
        for c in range(k):
            mask = clusters == c
            if mask.sum() > 0:
                cluster_map[c] = pd.Series(y_all[mask]).mode().iloc[0]

        mapped = np.array([cluster_map[c] for c in clusters])
        test_preds = mapped[len(X_train):]
        acc = accuracy_score(y_test_cls, test_preds)

        print(f"  GMM (k={k}):     Silhouette={sil:.3f}, Mapped Acc={acc:.3f}, BIC={gmm.bic(X_all):.0f}")
        results.append({
            "model": f"GMM (k={k})", "task": "clustering",
            "silhouette": round(sil, 4), "mapped_accuracy": round(acc, 4),
            "bic": round(gmm.bic(X_all), 2),
        })

    # ── DBSCAN ──
    for eps in [1.5, 2.0, 3.0]:
        db = DBSCAN(eps=eps, min_samples=10)
        clusters = db.fit_predict(X_all)
        n_clusters = len(set(clusters) - {-1})
        n_noise = (clusters == -1).sum()
        sil = None
        if n_clusters >= 2:
            mask = clusters != -1
            sil = silhouette_score(X_all[mask], clusters[mask])

        sil_str = f"{sil:.3f}" if sil is not None else "N/A"
        print(f"  DBSCAN (eps={eps}): Clusters={n_clusters}, Noise={n_noise}, Silhouette={sil_str}")
        results.append({
            "model": f"DBSCAN (eps={eps})", "task": "clustering",
            "silhouette": round(sil, 4) if sil else None,
            "n_clusters": n_clusters, "noise_points": n_noise,
        })

    return results, X_pca, y_all


# ──────────────── Feature Importance ───────────────────────────

def get_feature_importance(model, feature_cols, name, top_n=15):
    """Extract and display feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = (np.abs(model.coef_).mean(axis=0)
                       if model.coef_.ndim > 1 else np.abs(model.coef_))
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature": feature_cols, "importance": importances,
    }).sort_values("importance", ascending=False)

    print(f"\n🔍 Top {top_n} Features ({name}):")
    for _, row in fi.head(top_n).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"  {row['importance']:.3f} {bar} {row['feature']}")

    return fi


# ──────────────── Plotting ─────────────────────────────────────

def plot_model_comparison(results_df, save_path):
    """Create model comparison bar charts."""
    if not HAS_PLOTTING:
        return
    save_path.mkdir(parents=True, exist_ok=True)

    # Classification accuracy
    cls_df = results_df[results_df["task"] == "classification"].sort_values("accuracy", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(cls_df)))
    bars = axes[0].barh(cls_df["model"], cls_df["accuracy"], color=colors)
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Classification Accuracy (5-class stress)")
    axes[0].axvline(x=0.2, color="red", linestyle="--", alpha=0.5, label="Random (20%)")
    for bar, val in zip(bars, cls_df["accuracy"]):
        axes[0].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{val:.1%}", va="center", fontsize=9)
    axes[0].legend()

    x_pos = np.arange(len(cls_df))
    w = 0.35
    axes[1].barh(x_pos - w/2, cls_df["f1_weighted"], w, label="F1 (weighted)", color="#2196F3")
    axes[1].barh(x_pos + w/2, cls_df["f1_macro"], w, label="F1 (macro)", color="#FF9800")
    axes[1].set_yticks(x_pos)
    axes[1].set_yticklabels(cls_df["model"])
    axes[1].set_xlabel("F1 Score")
    axes[1].set_title("F1 Scores by Model")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path / "stress_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Regression comparison
    reg_df = results_df[results_df["task"] == "regression"]
    if not reg_df.empty:
        reg_df = reg_df.sort_values("mae", ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(reg_df)))
        bars = axes[0].barh(reg_df["model"], reg_df["mae"], color=colors)
        axes[0].set_xlabel("MAE")
        axes[0].set_title("Regression MAE (lower is better)")
        for bar, val in zip(bars, reg_df["mae"]):
            axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                         f"{val:.3f}", va="center", fontsize=9)

        r2_sorted = reg_df.sort_values("r2", ascending=True)
        bars = axes[1].barh(r2_sorted["model"], r2_sorted["r2"],
                           color=plt.cm.viridis(np.linspace(0.3, 0.9, len(r2_sorted))))
        axes[1].set_xlabel("R²")
        axes[1].set_title("Regression R² (higher is better)")
        axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, r2_sorted["r2"]):
            axes[1].text(max(val + 0.01, 0.01), bar.get_y() + bar.get_height()/2,
                         f"{val:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path / "stress_regression_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  📊 Plots saved to {save_path}/")


def plot_clustering(X_pca, y_all, save_path):
    """PCA scatter plot colored by stress level."""
    if not HAS_PLOTTING:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_all,
                        cmap="RdYlGn", alpha=0.5, s=15, edgecolors="none")
    plt.colorbar(scatter, ax=ax, label="Stress Level (1=stressed → 5=great)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of Sensor Features (colored by stress)")
    plt.tight_layout()
    plt.savefig(save_path / "stress_pca_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, name, save_path, offset=0):
    """Confusion matrix for the best model."""
    if not HAS_PLOTTING:
        return
    y_pred = model.predict(X_test)
    if offset:
        y_pred = y_pred + offset

    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Little\nstressed", "Def.\nstressed", "Stressed\nout",
              "Feeling\ngood", "Feeling\ngreat"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(save_path / "stress_confusion_matrix_best.png", dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────── Main Pipeline ────────────────────────────────

def main():
    print("=" * 60)
    print("🧠 Stress Prediction from Passive Sensor Data")
    print("   Comprehensive Model Comparison")
    print("   10 Supervised + Unsupervised Clustering")
    print("=" * 60)

    # 1. Load data
    print("\n📂 Loading data...")
    train_df = pd.read_csv(DATA_PATH / "train_stress.csv")
    val_df = pd.read_csv(DATA_PATH / "val_stress.csv")
    test_df = pd.read_csv(DATA_PATH / "test_stress.csv")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Prepare features
    print("\n⚙️  Preparing features...")
    (X_train, X_val, X_test,
     y_train_cls, y_val_cls, y_test_cls,
     y_train_reg, y_val_reg, y_test_reg,
     feature_cols, scaler) = prepare_data(train_df, val_df, test_df)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train shape: {X_train.shape}")

    # 3. Train all supervised models
    models = get_supervised_models()
    print(f"\n🏋️  Training {len(models)} supervised models...")

    sup_results, trained_models = train_and_evaluate_supervised(
        models, X_train, y_train_cls, y_train_reg,
        X_val, y_val_cls, y_val_reg,
        X_test, y_test_cls, y_test_reg
    )

    # 4. Unsupervised analysis
    unsup_results, X_pca, y_all = run_unsupervised_analysis(
        X_train, y_train_cls, X_test, y_test_cls, feature_cols
    )

    # 5. Summary table
    cls_results = [r for r in sup_results if r["task"] == "classification"]
    reg_results = [r for r in sup_results if r["task"] == "regression"]

    print(f"\n{'='*70}")
    print("  📊 FINAL RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1(w)':>10} {'F1(m)':>10}")
    print(f"  {'─'*55}")
    for r in sorted(cls_results, key=lambda x: x["accuracy"], reverse=True):
        print(f"  {r['model']:<25} {r['accuracy']:>9.1%} {r['f1_weighted']:>10.3f} {r['f1_macro']:>10.3f}")
    print(f"  {'─'*55}")
    print(f"  {'Random baseline':<25} {'20.0%':>10}")

    print(f"\n  {'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print(f"  {'─'*55}")
    for r in sorted(reg_results, key=lambda x: x["mae"]):
        print(f"  {r['model']:<25} {r['mae']:>10.3f} {r['rmse']:>10.3f} {r['r2']:>10.3f}")

    print(f"\n  Unsupervised:")
    print(f"  {'Model':<25} {'Silhouette':>10} {'Mapped Acc':>10}")
    print(f"  {'─'*55}")
    for r in unsup_results:
        s = f"{r['silhouette']:.3f}" if r.get('silhouette') else "N/A"
        a = f"{r['mapped_accuracy']:.1%}" if r.get('mapped_accuracy') else "N/A"
        print(f"  {r['model']:<25} {s:>10} {a:>10}")

    # 6. Feature importance
    best_cls = sorted(cls_results, key=lambda x: x["accuracy"], reverse=True)
    best_name = best_cls[0]["model"] if best_cls else "Random Forest"
    best_model = trained_models.get(f"{best_name}_cls")

    fi_best = pd.DataFrame()
    if best_model:
        fi_best = get_feature_importance(best_model, feature_cols, best_name)

    fi_rf = pd.DataFrame()
    if "Random Forest_cls" in trained_models:
        fi_rf = get_feature_importance(
            trained_models["Random Forest_cls"], feature_cols, "Random Forest")

    # 7. Cross-validation
    print(f"\n📊 Cross-validation (5-fold) on best model ({best_name})...")
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train_cls, y_val_cls])

    if best_name in models:
        cv_cls = models[best_name][0].__class__(**models[best_name][0].get_params())
        if best_name == "XGBoost":
            cv_scores = cross_val_score(cv_cls, X_trainval, y_trainval - 1,
                                        cv=5, scoring="accuracy")
        else:
            cv_scores = cross_val_score(cv_cls, X_trainval, y_trainval,
                                        cv=5, scoring="accuracy")
        print(f"  5-fold CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 8. Plots
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(sup_results)
    plot_model_comparison(results_df, FIGURES_PATH)
    plot_clustering(X_pca, y_all, FIGURES_PATH)
    if best_model:
        offset = 1 if best_name == "XGBoost" else 0
        plot_confusion_matrix(best_model, X_test, y_test_cls, best_name,
                             FIGURES_PATH, offset)

    # 9. Save
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Save top 3 models
    for r in best_cls[:3]:
        name = r["model"]
        safe = name.lower().replace(" ", "_")
        for suffix in ["cls", "reg"]:
            key = f"{name}_{suffix}"
            if key in trained_models:
                fname = f"{safe}_stress_{'classifier' if suffix == 'cls' else 'regressor'}.pkl"
                with open(MODEL_PATH / fname, "wb") as f:
                    pickle.dump(trained_models[key], f)

    with open(MODEL_PATH / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n💾 Models saved to {MODEL_PATH}/")

    # Save results
    all_results_df = pd.DataFrame(sup_results + unsup_results)
    all_results_df.to_csv(RESULTS_PATH / "stress_prediction_results.csv", index=False)

    if not fi_best.empty:
        fi_best.to_csv(RESULTS_PATH / "feature_importance_best.csv", index=False)
    if not fi_rf.empty:
        fi_rf.to_csv(RESULTS_PATH / "feature_importance_rf.csv", index=False)

    # JSON-safe converter for numpy types
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def clean_for_json(data):
        if isinstance(data, dict):
            return {k: clean_for_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [clean_for_json(i) for i in data]
        return to_native(data)

    summary = clean_for_json({
        "dataset": {
            "total_samples": len(train_df) + len(val_df) + len(test_df),
            "train": len(train_df), "val": len(val_df), "test": len(test_df),
            "participants": int(train_df["participant_id"].nunique()),
            "features": len(feature_cols), "lookback_hours": 6,
        },
        "best_model": best_cls[0] if best_cls else {},
        "supervised_results": sup_results,
        "unsupervised_results": unsup_results,
        "top_features": fi_best.head(10).to_dict("records") if not fi_best.empty else [],
    })
    with open(RESULTS_PATH / "stress_prediction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📊 Results saved to {RESULTS_PATH}/")
    print(f"\n✅ Comprehensive stress prediction pipeline complete!")
    print(f"   Best model: {best_name} (Accuracy={best_cls[0]['accuracy']:.1%})")


if __name__ == "__main__":
    main()
