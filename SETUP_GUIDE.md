# Quick Setup Guide - Streamlined for Fast ML Development

## 🎯 Goal
Get your ML environment running in **under 10 minutes** with minimal build time.

---

## 📋 Prerequisites

Before starting, verify:
```bash
docker --version        # Needs v20.10+
docker-compose --version  # Needs v2.0+
df -h .                 # Needs 20GB+ free space
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Environment Setup (1 minute)

```bash
# Navigate to project
cd ~/projects/StudentLife-Phenotyping

# Copy environment file
cp .env.example .env

# Create required directories
mkdir -p models reports data notebooks
```

### Step 2: Build Containers (3-5 minutes)

```bash
# Build both containers (training is behind a profile)
docker-compose --profile training build

# Expected output:
# ✓ MLflow image built (~2 min)
# ✓ Training image built (~3 min)
```

**What's being built:**
- **MLflow container**: Lightweight tracking server (~1.2GB)
- **Training container**: Minimal Python environment with essential tools only (~800MB)

**What's NOT included in build** (installed later interactively):
- PyTorch, Transformers (heavy ~2GB)
- XGBoost, Light GBM
- Jupyter, Matplotlib

### Step 3: Start Services (30 seconds)

```bash
# Start MLflow tracking server
docker-compose up -d mlflow

# Wait for health check
sleep 15

# Verify MLflow is running
docker-compose ps
# Should show: studentlife-mlflow (healthy)

# Open MLflow UI in browser
open http://localhost:5000  # Mac
# OR visit: http://localhost:5000
```

---

## 🔬 Interactive ML Environment

### Enter Training Container

```bash
# Start interactive bash session
docker-compose --profile training run --rm training bash

# You're now INSIDE the container!
# Prompt will change to: root@abc123:/app#
```

### Install ML Libraries (Automatic)

The pipeline script (`run_pipeline.sh`) automatically checks and installs dependencies on first run.
```bash
# Inside container - run full pipeline (first time)
./run_pipeline.sh
```

**Manual install (optional):**
```bash
# Inside container - set timeout for large CUDA packages
export UV_HTTP_TIMEOUT=120

# Install all dependencies from pyproject.toml
uv sync

# Or with dev dependencies (Jupyter, pytest)
uv sync --all-groups
```

> **Note**: Large packages like CUDA libraries (~1GB+) may timeout with default settings. Setting `UV_HTTP_TIMEOUT=120` gives more time for downloads.

### Verify Installation

```bash
# List installed ML packages
uv pip list | grep -E "torch|mlflow|pandas|xgboost|lightgbm"
```

### Test MLflow Connection

```bash
# Check if MLflow server is reachable
curl http://mlflow:5000/health
```

---

## � Data Requirements

Before running the full pipeline, you need the StudentLife dataset:

### Option 1: With StudentLife Dataset
**Downloading data from official site was too slow so uploaded same to AWS S3 bucket**
**Step 1: Download the dataset**

```bash
# Inside container - download StudentLife dataset from S3 (~400MB compressed)
mkdir -p data/raw
cd data/raw

# Download using wget (faster from S3)
wget https://student-pheno.s3.ap-south-1.amazonaws.com/raw/dataset.tar.bz2

# OR using curl
curl -O https://student-pheno.s3.ap-south-1.amazonaws.com/raw/dataset.tar.bz2
```

**Step 2: Extract the sensing data**

```bash
# Extract only the sensing folder (saves space)
tar -xjf dataset.tar.bz2 dataset/sensing

# Clean up compressed file (optional)
rm dataset.tar.bz2

# Verify extraction
ls dataset/sensing/  # Should show: activity, audio, bluetooth, conversation, etc.
```

**Step 3: Run data preparation**

```bash
# Go back to app directory
cd /app

# Run the full pipeline (includes data prep)
./run_pipeline.sh
```

> **Note**: The pipeline now auto-regenerates `participant_tiers.csv` if it’s missing.

**Alternative: Automated Download**

```bash
# Inside container - use the download script
python src/data/download_dataset.py

# This will:
# - Download dataset.tar.bz2 (~400MB)
# - Extract sensing/ folder automatically
# - Skip if already downloaded
```

> **⚠️ Note**: If download fails or file is corrupt, check:
> 1. S3 bucket must be publicly accessible
> 2. Verify URL returns the actual file (not HTML error page)
> 3. Check downloaded file size: `ls -lh data/raw/dataset.tar.bz2`
> 4. If file is tiny (<1MB), it's likely an error page - check S3 permissions

**Troubleshooting S3 download:**
```bash
# Inside container - check what was actually downloaded
file data/raw/dataset.tar.bz2  # Should say "bzip2 compressed data"
ls -lh data/raw/dataset.tar.bz2  # Should be ~400MB

# If it's an HTML error, make S3 bucket public:
# AWS Console → S3 → Bucket → Permissions → Public Access → Allow
# Then add bucket policy for public read access
```

### Option 2: Without Dataset (Development/Testing)

If you don't have the dataset yet, you can:

**Test individual components:**
```bash
# Test MLflow connectivity
python -c "import mlflow; mlflow.set_tracking_uri('http://mlflow:5000'); print('MLflow OK')"

# Start Jupyter for development
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# Access at http://localhost:8888

# Run model scripts directly (when you have sample data)
python src/analysis/modeling/09_transformer.py
```

**Create sample data** for testing (advanced):
```bash
# Generate synthetic data matching the schema
python src/data/generate_sample_data.py  # If you create this script
```

---

## �📊 Run Your ML Pipeline (With Data)

### Option A: Run Full Pipeline

```bash
# Inside container - runs complete data prep and all models
./run_pipeline.sh

# This executes:
# 1. Data cleaning and alignment
# 2. Feature engineering
# 3. Baseline models (regression, classification)
# 4. Advanced ML (boosting, LSTM)
# 5. SOTA models (Transformer, Autoencoder)
# All results tracked in MLflow
```

### Option B: Run Individual Scripts

```bash
# Inside container - run specific steps
python src/data/run_cleaning.py
python src/data/create_final_dataset.py
python src/analysis/modeling/09_transformer.py
```

### Option C: Use Jupyter Notebooks

```bash
# Inside container, start Jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Access from your browser:
# http://localhost:8888
```

### Option C: Develop Interactively

```bash
# Inside container
python

>>> import pandas as pd
>>> import mlflow
>>> # Your code here...
```

---

## 🎨 Optional: Run FastAPI Server

If you want to serve predictions later:

```bash
# Inside container
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Access API docs: http://localhost:8000/docs
```

---

## 🛑 Stop Services

```bash
# Exit container
exit

# Stop MLflow
docker-compose down

# Clean up everything (including volumes)
docker-compose down -v
```

---

## 📊 Command Reference

| Task | Command |
|------|---------|
| Build containers | `docker-compose --profile training build` |
| Start MLflow only | `docker-compose up -d mlflow` |
| Enter container | `docker-compose --profile training run --rm training bash` |
| Install all deps | `uv sync` (inside container) |
| Install with dev deps | `uv sync --all-groups` (inside container) |
| Stop services | `docker-compose down` |
| View logs | `docker-compose logs -f mlflow` |
| Check status | `docker-compose ps` |

---

## 🐛 Troubleshooting

### MLflow shows "unhealthy"
```bash
docker logs studentlife-mlflow
docker-compose restart mlflow
```

### Can't connect to MLflow from container
```bash
# Inside container
curl http://mlflow:5000
# Should return HTML
```

### MLflow 403 “Invalid Host header”
If MLflow rejects requests, allow host headers with ports in docker-compose:
- Add `MLFLOW_SERVER_ALLOWED_HOSTS=mlflow,mlflow:*,localhost,localhost:*,127.0.0.1,127.0.0.1:*,0.0.0.0,0.0.0.0:*` to the mlflow service
- Restart MLflow: `docker-compose up -d mlflow`

### Port already in use
### participant_tiers.csv missing
```bash
# Inside container
python src/data/regenerate_tiers.py
```
```bash
# Check what's using port 5000
lsof -i :5000
# Kill the process or change port in docker-compose.yml
```

### Compose warning about `version`
If you see a warning like “the attribute `version` is obsolete”, it's safe to ignore.

---

## ✨ Summary

**Total setup time**: ~10 minutes
- Build containers: 3-5 min
- Start MLflow: 30 sec
- Install ML libs (inside container): 5-8 min

**Workflow**:
1. Build once: `docker-compose --profile training build` 
2. Start MLflow: `docker-compose up -d mlflow`
3. Enter container: `docker-compose --profile training run --rm training bash`
4. Dependencies auto-install on first `./run_pipeline.sh`
5. **Add your data** to `data/raw/dataset/sensing/` (StudentLife dataset)
6. Develop your models!

**Benefits over old approach**:
- ⚡ **23 min faster** build time (28 min → 5 min)
- 🎯 **Simpler** - only 2 containers instead of 3
- 🔧 **Flexible** - auto-installs dependencies as needed
- 💾 **Smaller** images - minimal base layers

---

## 🚀 Next Steps

1. **Get Data**: Download StudentLife dataset → place in `data/raw/dataset/sensing/`
2. **Prepare Data**: Follow your `plan.md` Phases 2-4 (data cleaning, alignment, features)
3. **Train Models**: Run `./run_pipeline.sh` inside container
4. **Experiment**: Use MLflow UI to compare runs (http://localhost:5000)
5. **Deploy**: Use FastAPI when ready for inference

**Happy coding!** 🎯
