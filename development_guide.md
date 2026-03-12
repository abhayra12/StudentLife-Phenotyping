# Developer Guide — StudentLife Stress Phenotyping

> This is the primary technical reference for contributors. It covers architecture decisions, coding standards, testing, and extension patterns.

---

## Quick Start (Local, No Docker)

```bash
# Prerequisites: Python 3.9+, pip
git clone https://github.com/abhayra12/StudentLife-Phenotyping.git
cd StudentLife-Phenotyping

# Install all dependencies
pip3 install scikit-learn xgboost lightgbm catboost optuna shap torch \
             pandas numpy matplotlib seaborn pyarrow mlflow fastapi uvicorn

# Run the full 14-step pipeline
bash run_pipeline.sh
```

For Docker-based setup, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

---

## Architecture

### Data Flow

```
data/raw/dataset/sensing/          ← Read-only input
         ↓ [Step 1] src/data/run_cleaning.py
data/processed/cleaned/            ← Per-sensor cleaned CSVs
         ↓ [Step 2] src/data/run_alignment.py
data/processed/aligned/            ← Hourly-grid aligned CSVs (49 participants)
         ↓ [Step 3] src/data/create_final_dataset.py
data/processed/
  sensor_ema_merged.csv            ← 2,154 labeled samples (54 features + label)
  train_stress.csv / val / test    ← Chronological 70/15/15 split
  train.parquet / val / test       ← Hourly grid (for LSTM/Transformer)
         ↓ [Steps 4–13] src/analysis/
reports/results/                   ← JSON/CSV metric outputs
reports/figures/                   ← PNG visualizations
models/                            ← Serialized model artifacts
```

### Module Map

| Directory | Purpose |
|---|---|
| `src/data/` | ETL: cleaning, alignment, dataset assembly |
| `src/features/` | Feature extractors: activity/sleep, location, temporal |
| `src/analysis/eda/` | Exploratory analysis scripts |
| `src/analysis/modeling/` | All model training scripts (numbered 01–SOTA) |
| `src/api/` | FastAPI prediction service |
| `tests/` | pytest unit tests for data pipeline |
| `scripts/` | Utility scripts |

---

## Development Workflow

### Adding a Feature or Model

1. **Branch**: `git checkout -b feat/short-description`
2. **Implement**: Code goes in `src/`. Scripts over notebooks (see below).
3. **Test locally**: `python3 src/analysis/modeling/your_script.py`
4. **Unit tests**: Add to `tests/` if adding data pipeline logic
5. **Run full pipeline**: `bash run_pipeline.sh` — confirms no regressions
6. **Commit**: Use conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`)

### Adding a New Sensor Modality

1. `src/data/cleaning.py` — add validation and outlier handling for new sensor
2. `src/data/alignment.py` — add aggregation logic (e.g., `sum` for event counts, `mean` for continuous signals)
3. `src/data/create_final_dataset.py` — include new aggregated columns in feature matrix
4. Update feature count in `README.md` and `PRESENTATION_GUIDE.md`
5. Re-run steps 1–3 to regenerate all derived datasets

### Adding a New Model

1. Create `src/analysis/modeling/NN_model_name.py` (follow numbering)
2. Add step to `run_pipeline.sh` with correct `[N/14]` label
3. Outputs: write metrics JSON to `reports/results/`, figures to `reports/figures/modeling/`
4. Update step count in `setup_and_run.sh` header and `SETUP_GUIDE.md` pipeline table

---

## Coding Standards

### Paths — Always Use `pathlib`, Always Relative

```python
# ✅ Correct
from pathlib import Path
DATA_DIR = Path('data/processed')
output_path = DATA_DIR / 'results.csv'

# ❌ Wrong — breaks Docker, Windows, and any other developer's machine
DATA_DIR = '/home/user/project/data'
```

All scripts must run correctly from the project root: `python3 src/analysis/modeling/01_regression_baselines.py`

### Python Version

Use `python3` for all invocations. The project requires Python 3.9+. The system Python at `/usr/bin/python3` is the supported runtime. The `uv` venv (Python 3.14) is for Docker only and does not support CatBoost 1.2.8.

### Scripts Over Notebooks

All analysis is in `src/analysis/` Python scripts for reproducibility and git-diff readability. Notebooks are permitted only in `notebooks/` for exploratory scratch work — never import from them.

### Type Hints & Docstrings

```python
def make_features(df: pd.DataFrame, window_hours: int = 6) -> pd.DataFrame:
    """
    Compute feature vector for each EMA response.

    Args:
        df: Aligned sensor DataFrame with timestamp index.
        window_hours: Lookback window before each EMA response.

    Returns:
        Feature DataFrame with one row per EMA response.
    """
```

### Metrics Standard

- **Primary**: Macro F1 (stress classification tasks)
- **Secondary**: Accuracy, AUC
- **Regression auxiliary tasks**: MAE (stress is a 1–5 ordinal scale, not continuous)
- Never report accuracy alone on imbalanced datasets without majority-class baseline comparison

### Randomness & Reproducibility

```python
SEED = 42
np.random.seed(SEED)
# Pass random_state=SEED to all sklearn estimators
# Pass seed=SEED to CatBoost, XGBoost, LightGBM
```

---

## Testing

```bash
# Run all unit tests
python3 -m pytest tests/ -v

# Run a specific test file
python3 -m pytest tests/test_alignment.py -v

# Run with coverage
python3 -m pytest tests/ --cov=src --cov-report=term-missing
```

Tests cover:
- `test_alignment.py` — sensor window extraction, temporal leakage prevention
- `test_cleaning.py` — outlier removal, timestamp validation

When adding new data pipeline steps, add corresponding tests.

---

## Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `PermissionError` writing to `reports/` | Root-owned files from Docker run | `sudo chmod -R 666 reports/ models/` |
| `ModuleNotFoundError: catboost` | Running in uv venv (Python 3.14) | Use system `python3` — `which python3` should show `/usr/bin/python3` |
| `ModuleNotFoundError: torch` | Not installed on system Python | `pip3 install torch --index-url https://download.pytorch.org/whl/cpu` |
| LSTM/Transformer underperform gradient boosting | Expected behavior — see PRESENTATION_GUIDE.md | Not a bug |
| `uv sync` triggered during pipeline | `python` not on PATH (only `python3`) | Ensure `run_pipeline.sh` uses `python3` throughout |
| Missing `pyarrow` for `.parquet` | Not in base install | `pip3 install pyarrow` |

---

## Project History

### Current: SOTA Phase (this session)
- CatBoost + Optuna HPO (60-trial Bayesian search, MedianPruner)
- Two-level OOF stacking: RF + ET + XGB + LGB + CatBoost → Logistic Regression meta-learner
- SHAP TreeExplainer for multi-class feature attribution
- Pipeline extended from 13 → 14 steps
- README, SETUP_GUIDE, PRESENTATION_GUIDE rewritten to MAANG-grade technical quality

### Phase 7: Deep Learning (Jan 2026)
- LSTM (bidirectional, hidden=128): MAE ~1.18 on regression task
- Transformer (4-head, d_model=64): MAE ~1.18 (matched LSTM)
- Key finding: both underperform gradient boosting on aggregated features

### Phase 6: Advanced ML (Jan 2026)
- XGBoost regression: MAE 1.66, AUC 0.75
- LightGBM / CatBoost comparison
- SHAP analysis: `audio_voice_minutes` identified as top driver
- Autoencoder anomaly detection: 47 anomalous samples flagged

### Phase 5: Baseline Modeling (Jan 2026)
- Regression baselines: Ridge R²=0.11
- Classification baselines: Random Forest AUC 0.75
- Key insight: classification framing outperforms regression

### Phase 4: Feature Engineering (Jan 2026)
- Location features: DBSCAN mobility clusters
- Activity/sleep features: step count heuristics
- Temporal features: cyclical sin/cos encoding
- All notebooks converted to `src/analysis/` scripts

### Phase 2–3: Data Acquisition & EDA (Jan 2026)
- StudentLife dataset downloaded and extracted (48 participants, 10 sensors)
- Participant quality tiers defined (A/B/C by data completeness)
- Pivoted to sensing-only scope (no social media, academic records)

### Phase 1: Setup (Jan 2026)
- Project scaffolded with `pyproject.toml`, `uv` config
- Docker multi-stage build: training image + API image
- CI pipeline: pytest on push
