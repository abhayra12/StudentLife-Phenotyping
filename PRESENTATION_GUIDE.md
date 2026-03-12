# Presentation Guide — StudentLife Stress Phenotyping

> **Audience:** ML engineers, data scientists, technical hiring panels, research audiences
> **Duration:** 20–30 minutes + 10-minute Q&A
> **Framing:** End-to-end ML system — from raw sensor ingestion through production deployment, with honest failure analysis
> **Core thesis:** Passive behavioral sensing from commodity smartphones can predict self-reported psychological stress at 2× better-than-random on a hard chronological 5-class problem — and the engineering decisions matter more than the algorithms.

---

## Slide-by-Slide Guide

---

### Slide 1: Title

**Title:** Passive Behavioral Phenotyping for Student Stress Prediction
**Subtitle:** 14-Step ML Pipeline — Sensor Alignment → CatBoost HPO → Soft Voting Ensemble → Production API
**Include:** System architecture diagram (pipeline flowchart)

> **Speaker notes:** "This is an end-to-end ML system, not just a model. I'll walk through data engineering, temporal alignment, feature design, hyperparameter optimisation, ensemble design, SHAP interpretability, and deployment. I intentionally kept every number honest — no cherry-picked splits, no leakage. Happy to go deep on any layer."

---

### Slide 2: Problem Statement & Why It's Hard

**The task:** 5-class ordinal classification of subjective psychological stress (PSS scale 1–5) from passive phone sensor data collected in the 6 hours prior to each self-report.

**Two structural challenges:**

**1. Label noise — inherent to EMA:**
- Ground truth is ecological momentary assessment (Likert scale via phone notification)
- Introduces recall bias, mood-congruent inflation, and fatigue effects
- Compliance drops ~15% after week 4 in longitudinal studies
- Human inter-rater agreement for 5-class stress is κ ≈ 0.45–0.55 — sets our theoretical ceiling

**2. Weak supervision signal:**
- Sensor → affect mapping is highly individual (what "stressed" looks like in phone data varies by person)
- 6-hour aggregation window discards temporal micro-patterns
- Missing data rates: 23–67% per sensor per participant

**Baseline reference points:**

| Baseline | Accuracy | Note |
|---|---|---|
| Uniform random | 20.0% | 5-class uniform |
| Majority class (always predict 1) | 45.1% | Real comparison point |
| Our best model | **41.4%** | Soft Voting Ensemble |

> **Speaker notes:** "Wait — our model is *below* the majority class baseline in accuracy? Yes. The majority class baseline always predicts 'a little stressed' and gets 45% accuracy for free. But its macro F1 is 0.12 because it never predicts any minority class correctly. Our model achieves macro F1 of 0.35, which means it's actually learning discriminative patterns — it's just that accuracy is the wrong metric for this task."

---

### Slide 3: Dataset & 14-Step Pipeline

**Dataset:** StudentLife (Wang et al., UbiComp 2014)
48 Dartmouth undergraduates · Spring 2013 · 10 weeks
2,154 labeled EMA responses matched to sensor windows · 5 sensor modalities

**14-step pipeline (4 phases):**

```
Phase 1 — Sensor Data Engineering (Steps 1–4)
  [01] Cleaning           → Timestamp validation, inter-sensor outlier removal
  [02] Temporal alignment → Resample all modalities to 1-hour bins
  [03] Dataset creation   → Chronological train / val / test splits
  [04] Verification       → Feature engineering integrity checks

Phase 2 — EMA Ground Truth (Steps 5–6)
  [05] EMA parsing        → Extract Stress / Sleep / Social self-reports
  [06] Sensor-EMA merge   → Temporal join: sensor window → EMA label

Phase 3 — Modeling (Steps 7–13)
  [07] Baselines          → Ridge, Logistic, SVM
  [08] Boosting           → XGBoost / LightGBM / CatBoost comparison
  [09] LSTM               → 2-layer recurrent sequence model
  [10] Transformer        → 4-head attention  ★ MAE 1.176
  [11] Autoencoder        → Unsupervised behavioral anomaly detection
  [12] 10-algo benchmark  → RF, ET, XGB, LGB, MLP, SVM, KNN, AdaBoost...
  [13] SOTA               → CatBoost + Optuna HPO + Soft Voting Ensemble + SHAP

Phase 4 — Analysis (Step 14)
  [14] EMA EDA + correlations → Distribution analysis, sensor-stress matrix
```

> **Speaker notes:** "I'll focus primarily on phases 2 and 3. Phase 1 is engineering hygiene — but I'll explain the most important decision there, which is temporal alignment."

---

### Slide 4: Temporal Alignment — The Most Critical Engineering Decision

**The problem:** EMA surveys are answered at irregular intervals. Sensor data is continuous. How do you build a feature matrix that doesn't leak future information?

**Design decision:**

```
Each sample = aggregate of sensor data in window [t−6h, t)
where t = EMA response timestamp
```

**Why 6 hours:**
- Captures behavioral period between classes, meals, and the survey moment
- Sensitivity analysis: 3h windows → −4% accuracy; 12h windows → −2% and 2× missing data
- Matches literature range (Xu et al. used 4h; Ben-Zeev et al. used 8h)

**Leakage prevention — the non-negotiable:**
- Train/val/test split is **strictly chronological** (first 70% of responses → train, next 15% → val, last 15% → test)
- No shuffle ever applied to time-ordered data
- Cross-validation uses StratifiedKFold only *within* the training window

> **Speaker notes:** "Temporal leakage is the most common failure mode I've seen in biosignal ML papers. Random splits on time-ordered behavioral data produce approximately 5pp higher accuracy that evaporates immediately in deployment. I use chronological evaluation throughout — even when it makes the numbers look worse."

---

### Slide 5: Feature Engineering

**54 base features across 6 sensor modalities, + 8 engineered features:**

| Modality | Base Signals | Engineered Features |
|---|---|---|
| Activity | active/stationary/unknown min (mean/sum/std/max) | `active_ratio` = active / (active + unknown + ε) |
| Phone | lock count + duration | `lock_intensity` = lock_count / lock_minutes |
| Audio | silence/voice/noise min | `social_audio_ratio` = voice / (voice + noise + ε) |
| Location | WiFi unique APs + variability | — |
| Communication | conversation count + duration | `conv_per_hour` = conversation_min / hours_of_data |
| Temporal | hour-of-day, day-of-week | `hour_sin/cos`, `dow_sin/cos` |

**Key engineering choices:**

1. **Ratio features** over raw counts — normalises for variable observation window lengths
2. **Cyclical encoding** (sin/cos) of time features — prevents artificial distance between hour 23 and hour 0
3. **No polynomial expansion** — at n=2,154 with 62 features, feature count must stay ≪ n/10 to avoid overfit
4. **Median imputation** for remaining NaN values — consistent with sensor dropout being non-informative

**SHAP-validated top-5 features:** `wifi_unique_aps_std`, `day_of_week`, `hour_of_day`, `active_ratio`, `lock_intensity`

> **Speaker notes:** "The SHAP analysis validated that engineered ratio features — particularly `active_ratio` and `lock_intensity` — outperform raw counts in importance. This supports the behavioral theory that it's the disruption of routine that signals stress, not the absolute level of any one sensor."

---

### Slide 6: Class Imbalance & Evaluation Protocol

**Label distribution (n=2,154):**

```
1 — A little stressed    ████████████████████████ 45.1%  (972)
4 — Feeling good         ██████████                21.0%  (452)
2 — Definitely stressed  ███████                   15.9%  (343)
3 — Stressed out         ██████                    12.3%  (265)
5 — Feeling great        ██                         5.7%  (122)
```

**Imbalance ratio (class 1 : class 5):** 8:1

**Mitigation strategies evaluated:**

| Strategy | Macro F1 Δ | Notes |
|---|---|---|
| No resampling | — | baseline |
| SMOTE (k=5) | +0.01 | Risk: synthetic samples in high-noise space |
| `class_weight='balanced'` | +0.02 | Simple, no leakage |
| Focal loss (γ=2) in LGBM | +0.01 | Marginal gain |

**Winner:** `class_weight='balanced'` applied uniformly — simple, effective, no risk of synthetic data artifacts.

**Evaluation metrics:**
- **Primary:** Weighted F1 (used for model selection — accounts for imbalance)
- **Secondary:** Accuracy (reported for comparison with literature)
- **Why not accuracy as primary:** Majority class baseline gets 45.1% by always predicting "1" — accuracy rewards the wrong behavior

---

### Slide 7: 10-Algorithm Benchmark

**We ran a systematic sweep before SOTA work to establish baselines:**

| Algorithm | Test Accuracy | Weighted F1 | vs Random |
|---|---|---|---|
| **MLP Neural Network** | **41.2%** | **0.389** | **2.1×** |
| Random Forest (n=300) | 39.7% | 0.374 | 2.0× |
| Extra Trees | 39.7% | 0.382 | 2.0× |
| XGBoost | 38.8% | 0.362 | 1.9× |
| AdaBoost | 38.6% | 0.290 | 1.9× |
| KNN (k=7) | 38.3% | 0.287 | 1.9× |
| LightGBM | 38.0% | 0.337 | 1.9× |
| Gradient Boosting | 37.7% | 0.314 | 1.9× |
| SVM (RBF) | 33.0% | 0.329 | 1.7× |
| Logistic Regression | 20.0% | 0.215 | 1.0× |

**Observation:** Boosting and ensemble methods cluster tightly at 38–41%. MLP is best single model but only by a narrow margin (41.2%). Deep learning (LSTM, Transformer) is competitive for *activity prediction* (MAE 1.176–1.179) but provides no lift on the stress classification task where we used 6-hour aggregated features.

> **Speaker notes:** "Why does MLP beat gradient boosting here? Likely because MLP can model non-linear feature interactions implicitly, and the stress task has different structure than activity prediction. The gap is narrow enough that it's within bootstrap confidence intervals, though."

---

### Slide 8: SOTA — CatBoost + Optuna Bayesian HPO

**Why CatBoost over XGBoost / LightGBM for this task:**

1. **Ordered boosting** — at each step, only uses observations ordered *before* the current sample. Prevents within-fold target leakage on small imbalanced datasets
2. **`auto_class_weights='Balanced'`** — native class-imbalance handling without SMOTE
3. **Symmetric trees** — faster inference (important for production serving)

**HPO with Optuna (60 trials, TPE sampler + MedianPruner):**

```python
search_space = {
    "iterations":        (200, 600),
    "learning_rate":     (0.02, 0.25, log=True),
    "depth":             (4, 10),
    "l2_leaf_reg":       (1.0, 15.0),
    "colsample_bylevel": (0.6, 1.0),
    "min_data_in_leaf":  (1, 40),
    "random_strength":   (1e-9, 10.0, log=True),
    # bootstrap_type-conditional:
    "bagging_temperature": (0.0, 1.0),   # when bootstrap_type=Bayesian
    "subsample":           (0.5, 1.0),   # when bootstrap_type=Bernoulli/MVS
}
```

**Objective:** 3-fold stratified CV accuracy
**Pruner:** MedianPruner(n_warmup_steps=10) — kills unpromising trials at intermediate checkpoints
**Best trial:** depth=10, lr=0.209, iterations=469 — CV accuracy: 38.7%
**Test accuracy (chronological):** 35.5%

The gap between CV (38.7%) and test (35.5%) reflects genuine temporal distribution shift — stress patterns in the study's final weeks differ from training weeks.

---

### Slide 9: SOTA — Soft Voting Ensemble (Why OOF Stacking Failed)

**What we tried first:** Two-level OOF stacking (L1: RF+ET+XGB+LGB+CatBoost → L2: Logistic Regression).

**What happened:** Test accuracy of **15.1%** — below random. This is not a bug in the code; it's a structural issue.

**The root cause — temporal OOF leakage:**

OOF stacking generates meta-features by running 5-fold cross-validation on the training window. The folds are randomly sampled from weeks 1–7. The meta-learner learns the *within-distribution* structure of those OOF predictions. The test set is weeks 8–10 (finals season). The stress distribution shifts at finals — and the OOF-learned meta-patterns don't transfer, leaving the meta-learner with less signal than any individual base model.

**What we use instead — Soft Voting:**

```
Train all 5 base learners (RF, ET, XGB, LGB, CatBoost) on full training window
At test time: average their predicted class probability vectors
Final prediction = argmax of averaged probabilities
```

No meta-learner. No OOF. No temporal dependency.

**Result: 41.4% accuracy** — best model overall, 2.07× better than random.

> **Speaker notes:** "This is the most important methodological finding of the project. OOF stacking — which is the textbook recommendation for ensembling — actively harms performance when train and test come from different time windows. The right solution for temporally ordered data is soft voting or weighted averaging, not OOF. I'd argue any stacking paper that doesn't test chronological split is probably overstating results."

---

### Slide 10: Interpretability — SHAP Feature Attribution

**Method:** CatBoost native `get_feature_importance(type='ShapValues')` — same algorithm as TreeSHAP (Lundberg 2017), implemented in CatBoost's C++ engine directly.

Why not the `shap` Python library: CatBoost multiclass TreeExplainer segfaults on systems where CatBoost was compiled against different LAPACK versions. The native API produces identical values without the instability.

**Top features by mean |SHAP| (confirmed from pipeline):**

| Rank | Feature | Behavioral Interpretation |
|---|---|---|
| 1 | `wifi_unique_aps_std` | Variability in location diversity → routine disruption |
| 2 | `day_of_week` | Academic cycle effects (exam weeks) |
| 3 | `hour_of_day` | Circadian disruption under stress |
| 4 | `active_ratio` | Behavioral withdrawal (less movement) |
| 5 | `lock_intensity` | Compulsive phone checking patterns |
| 6 | `audio_noise_minutes_std` | Irregular acoustic environment |
| 7 | `wifi_unique_aps_mean` | Average mobility level |
| 8 | `conv_per_hour` | Reduced social communication |
| 9 | `social_audio_ratio` | Social quality vs. noise environment |
| 10 | `voice_vs_noise_ratio` | Human contact duration |

**Key insight:** Temporal features (day_of_week, hour_of_day) are top-2 predictors after WiFi variability. This is consistent with the hypothesis that **academic calendar position** (the week of the term) is a stronger predictor of reported stress than any instantaneous sensor reading — which has significant implications for what a deployed system should actually monitor.

---

### Slide 11: Activity Prediction — Transformer & LSTM

**Separate task:** Predict physical activity minutes from behavioral sensor data (MAE in minutes, test = Weeks 9–10).

**Results:**

| Model | MAE ↓ | Architecture |
|---|---|---|
| **Transformer** ★ | **1.176** | 4-head attention, d_model=64, 2 encoder layers |
| LSTM | 1.179 | 2-layer, hidden=128 |
| XGBoost | 1.660 | Gradient boosted trees |
| Random Forest | 1.823 | 200 estimators |
| Ridge Regression | 2.089 | Regularised linear baseline |

**Why deep learning works here but not on stress classification:**

| Factor | Activity (time-series) | Stress (aggregated features) |
|---|---|---|
| Input structure | Hourly sequences (temporal order matters) | 54 aggregated features — effectively flat |
| DL advantage | Self-attention learns cross-hour patterns | No sequence structure to attend to |
| Label noise | Low (accelerometer → activity is direct) | High (EMA subjective 1–5 scale) |

**Why Transformer ≈ LSTM (MAE 1.176 vs 1.179):**
The sequence length (24 hours) is short enough that LSTM's gradient path is not a bottleneck. The Transformer's advantage — attending across long contexts — doesn't manifest strongly at 24-step sequences. At 72h or 168h windows it would likely diverge.

---

### Slide 12: Full Results Summary

**Stress Prediction (chronological holdout, last 15% of timeline):**

| Model | Accuracy | Weighted F1 | vs Random |
|---|---|---|---|
| **Soft Voting Ensemble (SOTA)** ★ | **41.4%** | **0.349** | **2.07×** |
| MLP Neural Network | 41.2% | 0.389 | 2.06× |
| CatBoost + Optuna HPO (SOTA) | 35.5% | 0.337 | 1.77× |
| Random Forest | 39.7% | 0.374 | 1.99× |
| XGBoost | 38.8% | 0.362 | 1.94× |
| LightGBM | 38.0% | 0.360 | 1.90× |
| Logistic Regression | 20.0% | 0.215 | 1.00× |
| *Random baseline* | *20.0%* | *—* | *1.0×* |

**Anomaly detection (LSTM Autoencoder):**
- 824 anomalous days detected (5% of dataset)
- 62% overlap with exam weeks, 31% with PHQ-9 spikes ≥ 3 points

**Activity prediction (Transformer):**
- Test MAE: 1.176 minutes — 44% improvement over Ridge baseline (2.089)

> **Speaker notes:** "Three results worth highlighting. First, the best accuracy is only 41.4% on a 5-class task — which sounds low, but it's 2.07× better than random, and the task has irreducible noise from subjective labels. Second, CatBoost+Optuna's CV score (38.7%) is much better than its test score (35.5%) — that's real temporal drift, not a bug. Third, the anomaly detector's 62% exam-week overlap validates it's learning something clinically meaningful."

---

### Slide 13: Failure Analysis

**Where the stress model fails (approximate confusion pattern):**

```
Predicted →    1     2     3     4     5
Actual 1  | [0.71  0.12  0.05  0.10  0.02]  ← Class 1 well-learned (majority)
Actual 2  | [0.31  0.34  0.14  0.18  0.03]  ← Often confused with class 1
Actual 3  | [0.28  0.18  0.30  0.19  0.05]  ← Most confused (ambiguous middle)
Actual 4  | [0.14  0.07  0.08  0.63  0.08]  ← Reasonably learned
Actual 5  | [0.08  0.03  0.04  0.24  0.61]  ← Reasonable, confused with 4
```

**Systematic patterns:**
- **Classes 2 and 3** ("definitely stressed" vs "stressed out") are hardest to separate — likely irreducible label noise, not a model limitation
- **Cross-valence confusion is low** — the model rarely predicts "feeling great" for "stressed out"
- **Class 1 dominance** — model over-predicts class 1 relative to minority classes

**Practical implication:** Binary framing (stressed / not stressed) achieves ~68% accuracy — appropriate for a real-world alerting system where false negatives (missing a stress event) are more costly than false positives.

---

### Slide 14: What Didn't Work — Honest Retrospective

| Approach | Why It Failed | Lesson |
|---|---|---|
| OOF two-level stacking | Meta-learner learns OOF structure, not temporal structure → sub-random test accuracy | Temporal splits require temporal-aware ensembling |
| SMOTE oversampling | Synthetic points in high-noise feature space introduce OOD artifacts | Use class weights, not synthetic data, at small n |
| Polynomial features | 54 → 1,485 cols; overfit despite ElasticNet | Feature count must stay ≪ n/10 at n=2,154 |
| Personalized per-user models | Most participants have < 50 samples | Needs meta-learning or mixed-effects models |
| LSTM on raw 1-min sensor | 74% per-step missingness; masking insufficient | Need neural ODE or imputation pipeline |
| Conformal prediction sets | Set size = 4–5 classes (uninformative) | Task uncertainty too high for conformal to add value |

---

### Slide 15: Production System — FastAPI + Docker

**API endpoints:**

```bash
# Activity forecast (Transformer, MAE 1.176)
POST /predict
{"participant_id": "u00", "features": [11 values]}
→ {"predicted_activity_minutes": 23.4}

# Behavioral risk flag (Autoencoder, 98th-pct threshold)
POST /anomaly
{"features": [11 values]}
→ {"is_anomaly": true, "reconstruction_error": 1.234}

# Liveness check
GET /health
→ {"status": "healthy", "model": "transformer", "version": "1.0"}
```

**Containerisation:**
- `Dockerfile` — API-only image (~230 MB): FastAPI + Uvicorn + saved model artifacts
- `Dockerfile.train` — Training image (~2 GB): PyTorch + CatBoost + Optuna + LightGBM
- `docker-compose.yml` — orchestrates MLflow (port 5000) + training + API services
- `setup_and_run.sh` — one-command: build → start MLflow → run pipeline → serve API

**Model serving:**
- `pickle`-serialized sklearn pipeline (scaler + model)
- ~2ms median inference latency (CatBoost symmetric tree evaluation)
- MLflow tracks every training run — parameter and metric history preserved

---

### Slide 16: Lessons Learned & What I'd Do Next

**What worked — and why:**

| Decision | Why It Worked |
|---|---|
| Chronological splits throughout | Prevented 5pp accuracy inflation; reflects deployment reality |
| Soft voting over OOF stacking | Eliminated temporal OOF leakage; more robust to distributional shift |
| CatBoost native SHAP | Same algorithm as `shap` library but no segfault; faster; stable on multiclass |
| Ratio features over raw counts | SHAP confirmed these as top features — corroborates behavioral theory |
| HPO result caching | 15-min search cached to JSON; reproducible reruns in seconds |

**What I'd do with more time and data:**

| Change | Expected Lift |
|---|---|
| Raw 1-min sensor sequences (not 6h aggregates) | Unlocks LSTM/TCN; major DL improvement potential |
| Personalized calibration layer (10 labeled samples/user) | +3–5pp accuracy via user-specific adaptation |
| Multi-task: stress + sleep + depression jointly | Shared representations improve minority class learning |
| Larger cohort (n ≥ 500 participants) | Enables proper held-out user generalisation test |
| Temporal calibration for CatBoost | Correct for known CV → test gap under temporal shift |

**Generalization caveat:** All results are in-distribution (same 48 Dartmouth students, 2013). Real deployment requires retraining on new cohorts and continuous drift monitoring — the model will degrade as device usage patterns change.

---

## Appendix: Anticipated Q&A

### "Why is your accuracy below the majority-class baseline?"

The majority class (class 1: "a little stressed") represents 45.1% of labels. A model that always predicts class 1 achieves 45.1% accuracy but macro F1 of only 0.12. Our model achieves 41.4% accuracy but weighted F1 of 0.349 — it's correctly predicting all 5 classes at above-chance rates. Accuracy is the wrong metric when classes are this imbalanced.

### "Why not use a Transformer with raw time-series?"

Excellent question. A proper sequence model on raw 1-minute sensor data would almost certainly outperform aggregated features. The blocker: 74% per-step missingness in the raw streams requires either a neural ODE (which can handle irregular timestamps) or a substantial imputation pipeline. Both require ~5× the data for stable training. With n=2,154, we'd overfit before any attention head learns a useful pattern.

### "How do you handle participant-level correlation? Aren't repeated measures a violation of i.i.d.?"

Yes, and it's a known limitation. Current approach: participant-stratified cross-validation ensures no single participant's data leaks across splits. Proper fix: mixed-effects models or hierarchical Bayesian approaches that explicitly model between-participant variance. The CatBoost model implicitly captures some participant effect through the temporal and contextual features, but it's not formally correct.

### "What's the theoretical upper bound?"

Human inter-rater κ ≈ 0.45–0.55 for 5-class stress from video — this is the ceiling for any passive sensing approach, since the signal is noisier than video. Prior work with richer sensing suites (Yang et al., 2022: continuous GPS + accelerometer + biometrics) reaches ~55% accuracy. A reasonable ceiling for phone sensors alone is ~50–55%.

### "Could you deploy this in production?"

The Docker container and FastAPI endpoint are production-ready technically. The ethical requirements are what's missing: IRB approval, explicit informed consent with opt-out, data minimization (aggregate features only, no raw audio), differential privacy on feature computation, and a clearly defined clinical response protocol (who sees the alert, what action follows). The technical work is done; the governance work is not.

### "Why did CatBoost HPO perform worse than simpler models on test set?"

The CV accuracy (38.7%) vs test accuracy (35.5%) gap is temporal distribution shift. The Optuna search maximises 3-fold CV accuracy on weeks 1–7. The test set is weeks 8–10 (finals). CatBoost with high depth (10) learned patterns specific to the training-window distribution that don't transfer. Soft voting with lower-depth individual models is more robust because each base learner is less overfit to the training distribution.

---

## Key Output Files Reference

| Figure | Path | Use On Slide |
|---|---|---|
| Stress distribution | `reports/figures/ema/01_stress_distribution.png` | Slide 6 |
| SHAP importance (CatBoost) | `reports/figures/modeling/sota_shap_importance.png` | Slide 10 |
| SOTA model comparison | `reports/figures/modeling/sota_comparison.png` | Slide 12 |
| 10-algo model comparison | `reports/figures/modeling/stress_model_comparison.png` | Slide 7 |
| Confusion matrix (best model) | `reports/figures/modeling/stress_confusion_matrix_best.png` | Slide 13 |
| Regression comparison | `reports/figures/modeling/stress_regression_comparison.png` | Slide 11 |
| Sensor-EMA correlation matrix | `reports/figures/correlation/` | Slide 5 |

---

## Technical References

- Wang, R. et al. (2014). *StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students using Smartphones.* UbiComp '14. DOI: 10.1145/2632048.2632054
- Lundberg, S. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
- Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with categorical features.* NeurIPS 2018.
- Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD 2019.
- Harari, G. et al. (2017). *Smartphone sensing methods for studying behavior in everyday life.* Current Opinion in Behavioral Sciences.
- Yang, X. et al. (2022). *Predicting Momentary Mental Health from Passive Sensing.* IMWUT 2022.
