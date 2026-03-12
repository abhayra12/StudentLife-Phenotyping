# Setup Guide — StudentLife Phenotyping

> **Goal:** Get the full ML environment running in under 10 minutes with zero manual intervention.

---

## Prerequisites

Verify before starting:

```bash
docker --version        # Needs v20.10+
docker compose version  # Needs v2.0+
df -h .                 # Needs 20GB+ free space
```

---

## Quick Start (3 Steps)

### Step 1: Environment Setup (~1 min)

```bash
cd ~/projects/StudentLife-Phenotyping

# Create required output directories
mkdir -p models reports data/raw data/processed notebooks
```

### Step 2: Build Containers (~3–5 min)

```bash
docker-compose --profile training build
```

What gets built:
- **`studentlife-mlflow`** — Lightweight experiment tracking server (~500MB)
- **`studentlife-training`** — ML environment with all dependencies (~2GB)

Dependencies (PyTorch, XGBoost, LightGBM, CatBoost, Optuna, SHAP, etc.) are installed
on first `./run_pipeline.sh` via `uv sync` inside the container.

### Step 3: Start Services and Run Pipeline (~30–45 min)

```bash
# Start MLflow tracking server
docker-compose up -d mlflow
sleep 15  # Wait for health check to pass
docker-compose ps   # Confirm: studentlife-mlflow (healthy)

# Enter training container
docker-compose --profile training run --rm training bash

# Inside container — run the full 14-step ML pipeline
./run_pipeline.sh
```

---

## One-Command Alternative

Automates all three steps above:

```bash
# Full setup: build → start MLflow → run pipeline
./setup_and_run.sh

# Or run individual stages
./setup_and_run.sh --build      # Build containers only
./setup_and_run.sh --start      # Start MLflow only
./setup_and_run.sh --pipeline   # Run 14-step pipeline (inside container)
./setup_and_run.sh --api        # Start FastAPI server (port 8000)
./setup_and_run.sh --shell      # Enter container shell
./setup_and_run.sh --status     # Check running services
./setup_and_run.sh --stop       # Stop all services
```

---

## Dataset

The StudentLife dataset must be present before running the pipeline.

### Download (Recommended)

```bash
# Inside the container
mkdir -p data/raw && cd data/raw

# From AWS S3 (~400MB compressed)
wget https://student-pheno.s3.ap-south-1.amazonaws.com/raw/dataset.tar.bz2

# Extract sensing and EMA data
tar -xjf dataset.tar.bz2 dataset/sensing dataset/EMA dataset/survey
rm dataset.tar.bz2   # Optional cleanup
cd /app
```

Verify extraction:
```bash
ls data/raw/dataset/sensing/   # activity, audio, bluetooth, conversation, ...
ls data/raw/dataset/EMA/       # Stress, Sleep, Social, Activity, ...
```

### Automated Download (Alternative)

```bash
# Inside container — handles download and extraction automatically
python src/data/download_dataset.py
```

> **Troubleshooting:** If `wget` downloads an HTML error page instead of the archive,
> the S3 bucket may not be publicly accessible. Check:
> ```bash
> file data/raw/dataset.tar.bz2  # Should say: bzip2 compressed data
> ls -lh data/raw/dataset.tar.bz2  # Should be ~400MB
> ```

---

## Pipeline Reference

The pipeline runs 14 steps across 4 phases. Inside the container:

```bash
./run_pipeline.sh
```

| Phase | Step | Script | Description |
|-------|------|--------|-------------|
| **Sensor Data** | 1 | `src/data/run_cleaning.py` | Timestamp validation, outlier removal |
| | 2 | `src/data/run_alignment.py` | Resample all sensors to 1-hour bins |
| | 3 | `src/data/create_final_dataset.py` | Build train/val/test splits |
| | 4 | `src/verify_phase4.py` | Validate feature engineering |
| **EMA Data** | 5 | `src/data/ema_loader.py` | Parse stress/sleep/social self-reports |
| | 6 | `src/data/merge_sensor_ema.py` | Join sensor features with EMA stress labels |
| **Activity Models** | 7 | `01_regression_baselines.py` + `03_classification_baselines.py` | Linear/Ridge + Logistic/SVM baselines |
| | 8 | `06_boosting_comparison.py` | XGBoost, LightGBM, CatBoost gradient boosting |
| | 9 | `07_lstm_timeseries.py` | LSTM sequence model |
| | 10 | `09_transformer.py` | Transformer with Multi-Head Attention (**best: MAE 1.176**) |
| | 11 | `08_autoencoder.py` | Unsupervised anomaly detection |
| **Stress Prediction** | 12 | `stress_prediction.py` | 10 supervised algorithms + clustering |
| | 13 | `sota_stress_prediction.py` | CatBoost+Optuna HPO, Stacked Ensemble, SHAP |
| | 14 | `ema_eda.py` + `sensor_ema_correlation.py` | EMA analysis and visualizations |

Expected runtime: **30–45 minutes** (GPU-less CPU run).

---

## Manual Dependency Installation (Inside Container)

The pipeline auto-installs dependencies, but you can also install manually:

```bash
# Inside container
export UV_HTTP_TIMEOUT=120   # Needed for large CUDA downloads

uv sync                      # Install all dependencies
uv sync --all-groups         # Include dev dependencies (Jupyter, pytest)

# Verify
uv pip list | grep -E "torch|catboost|xgboost|lightgbm|optuna|shap"
```

---

## API Server

After running the pipeline, models are saved to `models/`. To serve predictions:

```bash
# Option A — Background server (inside container)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &>/dev/null &
sleep 3

# Health check
curl http://localhost:8000/health

# Prediction endpoint (264 values = 24 timesteps × 11 features)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"participant_id": "u00", "features": [0.5, 0.3, ...]}'

# Interactive Swagger docs
open http://localhost:8000/docs
```

```bash
# Option B — Via setup_and_run.sh (from host)
./setup_and_run.sh --api
```

---

## Service URLs

| Service | URL | Notes |
|---------|-----|-------|
| **MLflow UI** | http://localhost:5000 | Experiment tracking, run comparison |
| **FastAPI** | http://localhost:8000 | Prediction endpoints |
| **Swagger UI** | http://localhost:8000/docs | Interactive API documentation |

---

## Stop Services

```bash
# Exit container
exit

# Stop all services
docker-compose down

# Full cleanup (remove volumes)
docker-compose down -v
```

---

## Troubleshooting

**MLflow shows unhealthy:**
```bash
docker logs studentlife-mlflow
docker-compose restart mlflow
```

**`participant_tiers.csv` missing:**
```bash
python src/data/regenerate_tiers.py
```

**Port already in use:**
```bash
lsof -i :5000   # or :8000
# Kill the conflicting process or change ports in docker-compose.yml
```

**MLflow 403 "Invalid Host header":**
Add `MLFLOW_SERVER_ALLOWED_HOSTS=*` to the mlflow service in `docker-compose.yml`, then:
```bash
docker-compose up -d mlflow
```

**Compose warning about `version` key:**
Safe to ignore — the `version` key is deprecated but harmless in Compose v2.

---

## Command Reference

| Task | Command |
|------|---------|
| Full automated setup | `./setup_and_run.sh` |
| Build containers | `docker-compose --profile training build` |
| Start MLflow | `docker-compose up -d mlflow` |
| Enter container | `docker-compose --profile training run --rm training bash` |
| Run pipeline | `./run_pipeline.sh` (inside container) |
| Install dependencies | `uv sync` (inside container) |
| Stop all services | `docker-compose down` |
| View MLflow logs | `docker-compose logs -f mlflow` |
| Check service status | `docker-compose ps` |

---

## Summary

| Phase | Time | What Happens |
|-------|------|--------------|
| Build containers | ~5 min | One-time Docker image build |
| Start services | ~30 sec | MLflow tracking server ready |
| First `./run_pipeline.sh` | ~5–8 min setup + 35–50 min training | Auto-installs deps, processes data, runs all 14 pipeline steps |
| Subsequent runs | ~30–45 min | Deps already installed, just retrains |

**Workflow:**
```
Build once  →  Start MLflow  →  ./run_pipeline.sh  →  View results in MLflow UI
```
