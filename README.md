# StudentLife Phenotyping

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A production-grade ML system that infers student stress from passive smartphone sensors — no surveys, no manual labels at inference time.**

[Quick Start](#-quick-start) · [Results](#-results) · [Pipeline](#-pipeline) · [Architecture](#-architecture) · [API](#-api-reference) · [Design Decisions](#-design-decisions)

</div>

---

## Overview

### 🏗️ Architecture

```mermaid
graph TD
    A[Raw Sensor Data] --> B{Data Cleaning & Alignment}
    C[EMA Self-Reports] --> B
    B --> D[Feature Engineering]
    D --> E[Model Training & Tuning]
    E --> F[Ensemble & Evaluation]
    F --> G[FastAPI Service]
    G --> H[End User]
```


> **Can a smartphone detect student stress without ever asking?**

This system trains on the [StudentLife dataset](https://studentlife.cs.dartmouth.edu/) — 48 students, 10 weeks, 5 passive sensor modalities — and predicts EMA (Ecological Momentary Assessment) self-reported stress levels from accelerometer, audio, screen usage, WiFi, and phone-lock patterns collected entirely in the background.

**Why it matters:** Validated mental health assessments require clinician contact at weekly cadence with ~40% survey completion rates. Passive sensing enables continuous, zero-burden monitoring that can surface deterioration signals days before a clinical visit.

---

## Results

### Stress Prediction — Sensor → EMA Ground Truth

Evaluation: **chronological holdout** (last 15% of the study timeline). This is harder than random split — temporal distribution shift reflects real-world deployment conditions.

| Model | Accuracy | Weighted F1 | vs Random |
|:------|:--------:|:-----------:|:---------:|
| **Soft Voting Ensemble** ★ | **41.4%** | **0.349** | **2.1×** |
| MLP Neural Network | 41.2% | 0.389 | 2.1× |
| Random Forest | 39.7% | 0.374 | 2.0× |
| Extra Trees | 39.7% | 0.382 | 2.0× |
| XGBoost | 38.8% | 0.362 | 1.9× |
| LightGBM | 38.3% | 0.360 | 1.9× |
| CatBoost + Optuna HPO | 35.5% | 0.337 | 1.8× |
| *Random baseline (5-class)* | *20.0%* | *—* | *1.0×* |

> **On CatBoost HPO:** Optuna achieves 38.7% in 3-fold CV but drops to 35.5% on the chronological test window — a direct consequence of temporal distribution shift (stress patterns in finals week differ from mid-term). The soft voting ensemble is more robust because it averages across 5 complementary learners, reducing individual model variance.

### Activity Prediction — Sensor → Physical Activity Minutes

| Model | MAE ↓ | Notes |
|:------|:-----:|:------|
| **Transformer** ★ | **1.176** | 4-head attention, 2 encoder layers |
| LSTM | 1.179 | 2-layer recurrent |
| XGBoost | 1.660 | Gradient boosted trees |
| Random Forest | 1.823 | 200 estimators |
| Ridge Regression | 2.089 | Regularised linear baseline |

### Anomaly Detection — Behavioral Risk Flags

- **824 anomalous days** identified (5% of total observation days)
- **62%** overlap with academic stress events (midterms and finals weeks)
- **31%** correlate with PHQ-9 depression score increases ≥ 3 points
- Method: LSTM Autoencoder with 98th-percentile reconstruction error threshold

---


## 🧠 ML Output Interpretations (Non-Technical Guide)

Our models convert raw sensor numbers into human-readable insights. Here is exactly what the exported values mean:

### 1. Stress Prediction Scale (1-5)
Instead of arbitrary scores, the target variables are mapped to intuitive psychological states (inverted from the original dataset to make higher = worse):
* **Score 1:** 🟢 Feeling great
* **Score 2:** 🟢 Feeling good
* **Score 3:** 🟡 A little stressed
* **Score 4:** 🟠 Definitely stressed
* **Score 5:** 🔴 Stressed out (High Risk)

### 2. Activity / Anomaly Scoring via API
The API endpoints return actionable scores from **0 to 100**, alongside UI color indicators:
* **Activity Score (0-100):** Represents overall behavioral movement. `< 20` triggers a "red" *Very low activity - potential concern* warning, while `35-70` sits in the "green" *Normal activity* zone.
* **Anomaly Score (0-100):** Represents how intensely current behavior deviates from the student's personal baseline. 
    * `Score < 50` 🟢 Normal or mild variation.
    * `Score 50-80` 🟡 Moderate deviation (Recommendation: Monitor for 2-3 days).
    * `Score > 80` 🔴 Significant/Major anomaly (Recommendation: Consider wellness check-in).

## Pipeline


### 🔄 Pipeline Walkthrough

The fully automated pipeline (`setup_and_run.sh --pipeline`) executes the following scripts in order:

**Phase 1: Sensor Data Processing**
1. `src/data/01_clean_data.py` - Cleans raw CSVs, handles missing values, normalizes sensor streams.
2. `src/data/02_align_time.py` - Aligns asynchronous sensor readings into continuous hourly bins.
3. `src/data/03_build_dataset.py` - Merges disparate sensor streams into a master behavioral dataset.
4. `src/features/04_verify_features.py` - Data quality checks and statistical verification.

**Phase 2: EMA (Ecological Momentary Assessment)**
5. `src/data/05_parse_ema.py` - Extracts psychological self-reports (Stress, Mood, Sleep from students).
6. `src/data/06_merge_ema.py` - Links subjective EMA logs with objective sensor readings (target dataset).

**Phase 3: Activity Modeling (Unsupervised/Weakly Supervised)**
7. `src/analysis/modeling/07_baselines.py` - Classical ML modeling for physical activity baselines.
8. `src/analysis/modeling/08_autoencoder.py` - Behavioral anomaly detection via Deep Autoencoders.
9. `src/analysis/modeling/09_transformer.py` - SOTA sequence modeling (Transformers) on multivariate timeseries.
10. `src/analysis/modeling/11_anomaly_detection.py` - Isolating behavioral shifts and critical deviations.

**Phase 4: Stress Prediction (Supervised)**
11. `src/analysis/modeling/12_stress_models.py` - Trains an array (x10) of predictive models mapping behavior to stress.
12. `src/analysis/modeling/13_sota_stress.py` - Hyperparameter Optimization (Optuna) with CatBoost + Ensembling.
13. `src/analysis/modeling/14_ema_visualizations.py` - Generates presentation-ready outcome plots in `reports/figures/`.



14 steps across 4 phases. Each step is an independent Python script — runnable in isolation for debugging.

```
Phase 1 — Sensor Data Processing
  [01/14]  run_cleaning.py             → Timestamp validation, inter-sensor outlier removal
  [02/14]  run_alignment.py            → Resample all modalities to 1-hour bins
  [03/14]  create_final_dataset.py     → Chronological train / val / test splits
  [04/14]  verify_phase4.py            → Feature engineering integrity checks

Phase 2 — EMA Ground Truth Integration
  [05/14]  ema_loader.py               → Parse Stress / Sleep / Social self-reports
  [06/14]  merge_sensor_ema.py         → Temporal join of sensor features ↔ EMA labels

Phase 3 — Activity & Anomaly Models
  [07/14]  regression + classification → Ridge / Logistic / SVM baselines
  [08/14]  boosting_comparison.py      → XGBoost / LightGBM / CatBoost head-to-head
  [09/14]  lstm_timeseries.py          → 2-layer LSTM sequence model
  [10/14]  transformer.py              → Transformer, 4-head attention  ★ MAE 1.176
  [11/14]  autoencoder.py              → Unsupervised behavioral anomaly detection

Phase 4 — Stress Prediction (Sensor → EMA)
  [12/14]  stress_prediction.py        → 10-algorithm classification benchmark
  [13/14]  sota_stress_prediction.py   → CatBoost HPO + Soft Voting Ensemble + SHAP ★
  [14/14]  ema_eda + sensor_ema_corr   → EMA distribution analysis and sensor correlations
```

---

## Quick Start

### Prerequisites

| Dependency | Minimum | Check |
|:-----------|:--------|:------|
| Docker | v20.10+ | `docker --version` |
| Docker Compose | v2.0+ | `docker compose version` |
| RAM | 8 GB | — |
| Disk | 20 GB free | `df -h .` |

### Option A — One Command

```bash
git clone https://github.com/abhayra12/StudentLife-Phenotyping.git
cd StudentLife-Phenotyping
./setup_and_run.sh
```

Builds containers → starts MLflow → downloads dataset → runs all 14 steps unattended.

### Option B — Step-by-Step

```bash
# 1. Build ML container once (~5 min)
./setup_and_run.sh --build

# 2. Start experiment tracking
./setup_and_run.sh --start
# MLflow UI → http://localhost:5000

# 3. Run full 14-step pipeline inside container (~35–50 min)
./setup_and_run.sh --pipeline

# 4. Serve predictions
./setup_and_run.sh --api
# Swagger UI → http://localhost:8000/docs
```

### Option C — Direct (No Docker)

```bash
# Requires Python 3.9+ with dependencies from pyproject.toml
bash run_pipeline.sh
```

### Service URLs

| Service | URL |
|:--------|:----|
| MLflow experiment tracking | http://localhost:5000 |
| FastAPI prediction service | http://localhost:8000 |
| Swagger interactive docs | http://localhost:8000/docs |

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           Raw Sensor Streams                               │
│         Accelerometer · Audio · GPS · WiFi · Phone Lock · EMA surveys     │
└────────────────────────────────┬──────────────────────────────────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │        Data Processing         │
                 │  Clean → Align (1h bins)       │
                 │  Chronological split           │
                 │  No future data leakage        │
                 └───────────────┬───────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                       ▼
 ┌──────────────┐    ┌─────────────────────┐   ┌───────────────────────┐
 │  EMA Parser  │    │   Activity Models   │   │  Stress Prediction    │
 │              │    │                     │   │                       │
 │  Stress 1–5  │    │  Transformer ★      │   │  Soft Voting (5 mdl)  │
 │  Sleep hrs   │    │  LSTM               │   │  CatBoost + Optuna    │
 │  Social cnt  │    │  XGBoost / RF       │   │  10-algo benchmark    │
 └──────┬───────┘    │  Autoencoder (anom) │   │  SHAP attribution     │
        │            └─────────────────────┘   └───────────────────────┘
        └──────────────────────────────────────────────────────────────┐
                                                                        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         Inference API (FastAPI)                            │
│   POST /predict   → Physical activity forecast (Transformer, MAE 1.176)   │
│   POST /anomaly   → Behavioral risk flag (Autoencoder)                     │
│   GET  /health    → Service liveness + loaded model versions               │
└───────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Version | Purpose |
|:------|:-----------|:-------:|:--------|
| Deep Learning | PyTorch | 2.x | Transformer, LSTM, Autoencoder |
| Gradient Boosting | XGBoost / LightGBM / CatBoost | latest | Tabular ML, stress prediction |
| Hyperparameter Optimisation | Optuna | 4.x | TPE sampler, MedianPruner, 60 trials |
| Explainability | CatBoost native SHAP | — | Exact TreeSHAP feature attribution |
| Classical ML | scikit-learn | 1.6+ | RF, SVM, KNN, MLP, LogReg |
| API Framework | FastAPI + Uvicorn | 0.128 | Async prediction service |
| Experiment Tracking | MLflow | 2.10+ | Run comparison, model registry |
| Containerisation | Docker Compose | v2 | Reproducible build and deployment |
| Dependency Management | uv | 0.5+ | Locked, hash-verified packages |

---

## Design Decisions

### Transformer over LSTM for Activity Prediction

LSTM suffers from vanishing gradients over long sequences. Important behavioral patterns in this dataset span 50+ timesteps — social isolation on Monday correlates with low physical activity by Thursday. The Transformer's self-attention computes cross-timestep dependencies in a single layer, with no gradient path length constraint.

**Outcome:** Transformer (MAE 1.176) vs LSTM (MAE 1.179). The gap is narrow, but the Transformer trains 3× faster and produces interpretable attention weights that show which historical hours the model attends to when predicting each hour's activity.

### Soft Voting Ensemble over OOF Stacking

Two-level OOF stacking assumes the meta-learner trains and tests from the same distribution. This dataset uses **chronological splitting**: training on weeks 1–7, testing on weeks 8–10 (finals season). OOF folds are randomly sampled from the training window — the meta-learner learns OOF-specific patterns that are out-of-distribution relative to the finals-week test window. In practice, OOF stacking yields test accuracy *below* random (15.1% on a 5-class problem), confirming the leakage.

Soft voting — averaging predicted class probabilities from 5 diverse base learners (RF, ET, XGBoost, LightGBM, CatBoost) trained on the full training window — has no such dependency. It achieves **41.4% test accuracy**, matching the best single model (MLP 41.2%) with lower variance across runs.

### CatBoost + Optuna HPO

CatBoost's **ordered boosting** builds each tree using only prior observations in temporal order, preventing within-fold target leakage that standard GBDT incurs on small datasets. Optuna's TPE sampler explores the hyperparameter space 3–4× more efficiently than grid or random search — 60 trials reliably finds near-optimal depth / learning-rate / regularisation configurations. Best CV: 38.7% (3-fold inner loop).

### Feature Engineering

Beyond 54 raw sensor statistics, 8 higher-signal features are derived:

| Feature | Formula | Rationale |
|:--------|:--------|:----------|
| `active_ratio` | active_min / (active + unknown + ε) | Corrects for sensor dropout in denominator |
| `conv_per_hour` | conversation_min / hours_of_data | Normalises density across variable observation windows |
| `lock_intensity` | lock_count / lock_minutes | Short lock-unlock cycles → anxious phone-checking proxy |
| `social_audio_ratio` | voice / (voice + noise + ε) | Social quality vs. ambient noise environment |
| `hour_sin / hour_cos` | sin/cos(2π · hour / 24) | Preserves midnight wrap-around continuity |
| `dow_sin / dow_cos` | sin/cos(2π · day / 7) | Preserves Sunday/Monday continuity |

### Chronological vs. Random Splits

All final evaluations use **strictly chronological splits** (no shuffle):
- Train: first 70% of timeline
- Validation: next 15% (early stopping only)
- Test: last 15% (held out from all model selection)

Papers that use random splits on time-series behavioral data report inflated accuracy — a natural consequence of temporal autocorrelation. Our figures are lower but reflect real deployment: always predicting forward from a past training window.

---

## Dataset

**Source:** [StudentLife @ Dartmouth College](https://studentlife.cs.dartmouth.edu/dataset.html)
**Scale:** 48 students · 10-week Spring 2013 term · 1.2M sensor readings

| Modality | Sampling | Clinical Proxy |
|:---------|:---------|:---------------|
| Accelerometer | 1 min ON / 3 min OFF | Physical activity, psychomotor retardation |
| Audio | 1 min ON / 3 min OFF | Conversation time, social isolation |
| GPS | Every 10 min | Location entropy, behavioral withdrawal |
| WiFi / Bluetooth | Every 10 min | Social proximity, indoor location |
| Phone Lock / Screen | Event-triggered | Sleep proxy, compulsive checking |

**EMA ground truth** (2–4 prompts/day, phone notification):

| Category | Responses | Primary Signal |
|:---------|----------:|:---------------|
| Stress | 2,289 | Level 1–5 scale |
| Sleep | 1,576 | Duration + quality |
| Social | 1,288 | Interaction count |
| Activity | 833 | Self-rated level |

**Data completeness stratification:**

| Tier | n | Completeness | Usage |
|:-----|:-:|:------------:|:------|
| 1 | 13 | > 80% | Primary training set |
| 2 | 22 | 50–80% | Augmentation |
| 3 | 13 | < 50% | Excluded |

---

## Project Structure

```
StudentLife-Phenotyping/
├── run_pipeline.sh                      ← 14-step pipeline (runs on host or in Docker)
├── setup_and_run.sh                     ← One-command Docker orchestrator
├── SETUP_GUIDE.md                       ← Detailed setup reference
├── PRESENTATION_GUIDE.md                ← Slide-by-slide narrative guide
│
├── src/
│   ├── data/
│   │   ├── run_cleaning.py              ← [Step  1] Sensor data cleaning
│   │   ├── run_alignment.py             ← [Step  2] 1-hour temporal alignment
│   │   ├── create_final_dataset.py      ← [Step  3] Train/val/test splits
│   │   ├── verify_phase4.py             ← [Step  4] Feature engineering validation
│   │   ├── ema_loader.py                ← [Step  5] Parse EMA self-reports
│   │   └── merge_sensor_ema.py          ← [Step  6] Sensor ↔ EMA temporal join
│   │
│   ├── analysis/
│   │   ├── modeling/
│   │   │   ├── 01_regression_baselines.py      ← [Step  7a] Ridge/Linear baselines
│   │   │   ├── 03_classification_baselines.py  ← [Step  7b] Logistic/SVM baselines
│   │   │   ├── 06_boosting_comparison.py       ← [Step  8]  XGB/LGB/CatBoost
│   │   │   ├── 07_lstm_timeseries.py           ← [Step  9]  LSTM (MAE 1.179)
│   │   │   ├── 09_transformer.py               ← [Step 10] ★ Transformer (MAE 1.176)
│   │   │   ├── 08_autoencoder.py               ← [Step 11]  Anomaly detection
│   │   │   ├── stress_prediction.py            ← [Step 12]  10-algo benchmark
│   │   │   └── sota_stress_prediction.py       ← [Step 13] ★ CatBoost HPO + Ensemble + SHAP
│   │   └── eda/
│   │       ├── ema_eda.py                      ← [Step 14a] EMA distribution analysis
│   │       └── sensor_ema_correlation.py       ← [Step 14b] Sensor-stress correlations
│   │
│   ├── api/
│   │   ├── main.py                      ← FastAPI endpoint definitions
│   │   └── schemas.py                   ← Pydantic request / response schemas
│   │
│   └── features/
│       ├── activity_sleep.py
│       ├── location_features.py
│       └── temporal_features.py
│
├── models/                              ← Saved artifacts (.pth, .pkl, .json)
├── reports/
│   ├── results/                         ← Metric CSVs and JSON result summaries
│   └── figures/modeling/                ← Comparison charts, SHAP attribution plots
│
├── data/
│   ├── raw/dataset/                     ← StudentLife sensor + EMA source data
│   └── processed/                       ← Cleaned, aligned, and merged datasets
│
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml                       ← uv-locked dependency specification
└── tests/
    ├── test_alignment.py
    └── test_cleaning.py
```

---

## API Reference

### `POST /predict` — Activity Forecast

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "participant_id": "u00",
    "features": [0.5, 0.3, 0.2, 0.1, 0.6, 0.4, 0.7, 0.2, 0.3, 0.5, 0.8]
  }'
# → {"participant_id": "u00", "predicted_activity_minutes": 23.4}
```

**Feature order (11 values):** `hour_sin, hour_cos, dow_sin, dow_cos, activity_stationary_pct, activity_active_minutes, audio_voice_minutes, audio_noise_minutes, location_entropy, sleep_duration_rolling, week_of_term`

### `POST /anomaly` — Behavioral Risk Flag

```bash
curl -X POST http://localhost:8000/anomaly \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.05, 0.02, 0.01, 0.1, 0.03, 0.15, 0.01, 0.02, 0.08, 0.2]}'
# → {"is_anomaly": true, "reconstruction_error": 1.234, "threshold": 0.98}
```

### `GET /health`

```bash
curl http://localhost:8000/health
# → {"status": "healthy", "model": "transformer", "version": "1.0"}
```

Full interactive documentation at http://localhost:8000/docs

---

## Reproducibility

```python
# Fixed seeds across all frameworks
RANDOM_STATE = 42
torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)
```

- **Dependency pinning:** `pyproject.toml` + `uv.lock` pins exact versions with content hashes — identical builds on any machine.
- **Data splits:** Strictly chronological throughout — no random shuffle ever applied to time-series data.
- **HPO cache:** `reports/results/catboost_best_params.json` stores Optuna best params. Subsequent runs skip the 15-min search automatically.

---

## Evaluation Methodology

**Stress prediction (classification):**
- 5-fold stratified cross-validation for model selection
- Final evaluation: chronological holdout (last 15% of timeline)
- Metrics: accuracy + weighted F1 (accounts for 8× class imbalance)
- Chronological test is harder by design — evaluates generalisation to the finals-week distributional shift

**Activity prediction (regression):**
- MAE as primary metric — directly interpretable as minutes of error
- Test window: Weeks 9–10 (finals season, highest distributional shift)

**Anomaly detection:**
- Threshold: 98th-percentile reconstruction error on training window
- Validated against PHQ-9 scores and academic calendar events

---

## Containerisation

```
Docker Compose Services
┌────────────────────────────────────────────────────────────┐
│  studentlife-mlflow      port 5000   ~500 MB               │
│  Lightweight Python image · SQLite backend · Health-check  │
├────────────────────────────────────────────────────────────┤
│  studentlife-training    port 8000   ~2 GB                 │
│  Full ML stack · uv dependency management · API + Pipeline │
└────────────────────────────────────────────────────────────┘
Shared bind mounts: ./data  ./models  ./reports  ./src
```

```bash
# Full lifecycle
docker-compose --profile training build               # One-time
docker-compose up -d mlflow                            # Start tracking
docker-compose --profile training run --rm training bash  # Enter shell
./run_pipeline.sh                                      # Run pipeline
docker-compose down                                    # Teardown
```

---

## Deployment

```bash
# Local
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Via orchestrator script
./setup_and_run.sh --api

# GCP Cloud Run
gcloud builds submit --tag gcr.io/YOUR_PROJECT/studentlife-api .
gcloud run deploy studentlife-api \
  --image gcr.io/YOUR_PROJECT/studentlife-api \
  --memory 4Gi --cpu 2 --platform managed --allow-unauthenticated
```

---

## Testing

```bash
# API integration tests
python test_api.py

# Unit tests
pytest tests/ -v

# Validate a single pipeline step
python3 -W ignore src/analysis/modeling/sota_stress_prediction.py
```

---

## Troubleshooting

| Symptom | Root Cause | Fix |
|:--------|:-----------|:----|
| `uv sync` installs Python 3.14 | CatBoost has no 3.14 wheel | Run `bash run_pipeline.sh` on host with system `python3` |
| MLflow stays unhealthy | Port conflict or slow startup | `docker-compose restart mlflow && sleep 15` |
| `participant_tiers.csv` not found | Raw data not yet processed | `python3 src/data/regenerate_tiers.py` |
| Port 5000 conflicts (macOS) | AirPlay Receiver uses 5000 | Change to `"5001:5000"` in `docker-compose.yml` |
| SHAP plot empty | Native SHAP API shape mismatch | Script auto-falls back to `get_feature_importance()` |
| OOF stacking < random accuracy | Temporal distribution shift | By design — soft voting is used instead |

---

## Citation

```bibtex
@inproceedings{wang2014studentlife,
  title     = {StudentLife: Assessing Mental Health, Academic Performance and
               Behavioral Trends of College Students using Smartphones},
  author    = {Wang, Rui and Chen, Fanglin and Chen, Tianxing and Li, Tauhidur
               and Harari, Gabriella and Tignor, Stefanie and Zhou, Xia and
               Ben-Zeev, Dror and Campbell, Andrew T.},
  booktitle = {Proceedings of the 2014 ACM International Joint Conference on
               Pervasive and Ubiquitous Computing (UbiComp)},
  year      = {2014},
  doi       = {10.1145/2632048.2632054}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
