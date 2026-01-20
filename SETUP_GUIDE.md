# Complete Setup Guide - Fresh Machine Installation

## 🎯 Scenario: Setting Up StudentLife-Phenotyping on a New Machine

This guide walks through the **complete end-to-end setup** from scratch.

---

## 📋 Prerequisites

Before starting, ensure you have:
- Docker installed (`docker --version`)
- Docker Compose installed (`docker-compose --version`)
- Git installed
- 20GB free disk space
- Internet connection

---

## 🚀 Part 1: Outside Container Setup

### Step 1: Clone Repository

```bash
# Navigate to your projects directory
cd ~/projects  # Linux/Mac
# OR
cd C:\Users\YourName\projects  # Windows

# Clone the repository
git clone https://github.com/your-username/StudentLife-Phenotyping.git
cd StudentLife-Phenotyping

# Verify you're in the right directory
ls -la  # Linux/Mac
dir     # Windows
```

**Expected Output:** You should see files like `docker-compose.yml`, `Dockerfile`, `README.md`, etc.

---

### Step 2: Environment Configuration

```bash
# Copy environment template
cp .env.example .env  # Linux/Mac
Copy-Item .env.example .env  # Windows PowerShell

# View the environment file (optional)
cat .env  # Linux/Mac
type .env  # Windows

# No changes needed for local development
```

**What's in `.env`:**
- MLflow tracking URI
- Model registry names
- Resource limits
- Various configuration options

---

### Step 3: Build All Containers (Optimized)

**🚀 We'll use Docker BuildKit for faster builds (10 min vs 25+ min)**

#### Enable BuildKit First:

**Windows PowerShell:**
```powershell
# Set environment variables for optimized builds
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1

# Verify BuildKit is enabled
Write-Host "✓ BuildKit enabled: DOCKER_BUILDKIT=$env:DOCKER_BUILDKIT" -ForegroundColor Green
```

**Linux/Mac:**
```bash
# Set environment variables
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Verify
echo "✓ BuildKit enabled: DOCKER_BUILDKIT=$DOCKER_BUILDKIT"
```

**💡 Tip:** Add these to your shell profile (`~/.bashrc`, `~/.zshrc`, or PowerShell profile) to make them permanent.

---

#### Optional: Pre-download Base Images

```bash
# Download Python base image in parallel (saves ~2 minutes)
docker pull python:3.13-slim
```

---

#### Build Containers

```bash
# Build all three containers (MLflow, Training, API)
docker-compose build

# With BuildKit enabled:
# - First build: ~10 minutes
# - Subsequent builds: ~3 minutes
# - Code-only changes: ~15 seconds
```

**Expected Output:**
```
[+] Building 600.0s (BuildKit enabled)
 ✔ Image studentlife-phenotyping-mlflow     Built   120s
 ✔ Image studentlife-phenotyping-training   Built   480s
```

**Verify images:**
```bash
docker images | grep studentlife
```

**Expected:**
```
REPOSITORY                         TAG       SIZE
studentlife-phenotyping-training   latest    14.1GB
studentlife-phenotyping-mlflow     latest    1.24GB
```

**📊 BuildKit Performance:**
- **First build:** 10 min (vs 25+ min without BuildKit)
- **Rebuild after dependency change:** 3 min (vs 25+ min)
- **Rebuild after code change:** 15 sec (vs 25+ min)

**See `BUILD_OPTIMIZATION.md` for advanced optimizations (cache mounts, multi-stage builds).**

---

### Step 4: Start MLflow Tracking Server

```bash
# Start MLflow in detached mode (background)
docker-compose up -d mlflow

# Wait for health check (10-15 seconds)
sleep 15

# Check status
docker-compose ps
```

**Expected Output:**
```
NAME                 STATUS
studentlife-mlflow   Up 15 seconds (healthy)
```

**Verify MLflow UI:**
```bash
# Open in browser
# Linux: xdg-open http://localhost:5000
# Mac: open http://localhost:5000
# Windows: start http://localhost:5000

# OR test with curl
curl http://localhost:5000
```

---

### Step 5: Verify Network Setup

```bash
# Check Docker networks
docker network ls | grep studentlife

# Inspect the network
docker network inspect studentlife-phenotyping_studentlife-network
```

**Expected:** You should see `studentlife-mlflow` container connected.

---

## 🎓 Part 2: Inside Container - Interactive Mode

### Step 6: Launch Interactive Training Container

```bash
# Start interactive bash shell
docker-compose run --rm training bash
```

**You're now INSIDE the container!** Prompt changes to something like:
```
root@abc123:/app#
```

---

### Step 7: Test Network Connectivity (Inside Container)

```bash
# Test 1: Can we reach MLflow?
curl http://mlflow:5000

# Expected: HTML output with "MLflow" text

# Test 2: Check hostname resolution
ping -c 3 mlflow

# Expected: 
# PING mlflow (172.18.0.2) 56(84) bytes of data.
# 64 bytes from studentlife-mlflow.studentlife-phenotyping_studentlife-network
```

---

### Step 8: Test MLflow Python Integration (Inside Container)

```bash
# Test Python MLflow connection
python -c "
import mlflow
print(f'MLflow version: {mlflow.__version__}')
mlflow.set_tracking_uri('http://mlflow:5000')
experiments = mlflow.search_experiments()
print(f'Found {len(experiments)} experiments')
print('✓ MLflow connection successful!')
"
```

**Expected Output:**
```
MLflow version: 2.10.0 (or higher)
Found 1 experiments
✓ MLflow connection successful!
```

---

### Step 9: Test MLflow Config Module (Inside Container)

```bash
# Test our custom MLflow configuration
python -c "
from src.mlflow_config import setup_mlflow, EXPERIMENT_NAME
setup_mlflow()
print('✓ MLflow setup complete!')
print(f'Experiment: {EXPERIMENT_NAME}')
"
```

**Expected Output:**
```
✓ MLflow initialized:
  - Tracking URI: http://mlflow:5000
  - Experiment: studentlife-phenotyping
✓ MLflow setup complete!
Experiment: studentlife-phenotyping
```

---

### Step 10: Log a Test Experiment (Inside Container)

```bash
# Create a test MLflow run
python -c "
import mlflow
from src.mlflow_config import setup_mlflow

setup_mlflow()

with mlflow.start_run(run_name='first_test_run'):
    # Log parameters
    mlflow.log_param('environment', 'docker')
    mlflow.log_param('test_mode', 'interactive')
    
    # Log metrics
    mlflow.log_metric('test_accuracy', 0.95)
    mlflow.log_metric('test_loss', 0.05)
    
    print('✓ Logged test experiment!')
    print('Check MLflow UI: http://localhost:5000')
"
```

**Verify:** Open http://localhost:5000 in browser → You should see new run!

---

### Step 11: Check Project Structure (Inside Container)

```bash
# View directory structure
ls -la

# Expected:
# drwxr-xr-x  - root root  src/
# -rw-r--r--  - root root  train.py
# drwxr-xr-x  - root root  models/
# drwxr-xr-x  - root root  data/
```

```bash
# Check if all source files are present
ls -la src/
ls -la src/analysis/modeling/

# Expected:
# 08_autoencoder.py
# 09_transformer.py
```

---

### Step 12: Check Installed Dependencies (Inside Container)

```bash
# Verify ML libraries are installed
python -c "
import torch
import mlflow
import pandas as pd
import sklearn
import lightgbm
import xgboost

print(f'PyTorch: {torch.__version__}')
print(f'MLflow: {mlflow.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'LightGBM: {lightgbm.__version__}')
print(f'XGBoost: {xgboost.__version__}')
print('✓ All dependencies installed!')
"
```

---

### Step 13: Simulate Training (Inside Container - No Data Required)

```bash
# Test training script imports
python -c "
# Test if training scripts are importable
from src.analysis.modeling import transformer_model
print('✓ Can import transformer module')

# Test MLflow integration in training
import mlflow
from src.mlflow_config import setup_mlflow, log_model_to_registry

setup_mlflow()

# Simulate a minimal training run
with mlflow.start_run(run_name='simulation_test'):
    mlflow.log_param('model', 'transformer')
    mlflow.log_param('epochs', 50)
    mlflow.log_metric('train_loss', 1.5)
    mlflow.log_metric('val_loss', 1.7)
    print('✓ Training simulation logged!')
"
```

---

### Step 14: Exit Container

```bash
# Exit the interactive shell
exit
```

**You're now OUTSIDE the container again!**

---

## 🔄 Part 3: Back Outside Container - Next Steps

### Step 15: View MLflow UI

```bash
# MLflow should still be running
docker-compose ps

# Open MLflow UI in browser
# http://localhost:5000

# You should see your test runs!
```

---

### Step 16: Prepare for Actual Training (Data Required)

**Option A: If you have the StudentLife dataset:**

```bash
# Place dataset in correct location
# data/raw/dataset/

# Then run inside container:
docker-compose run --rm training bash
# Inside: python src/data/make_dataset.py
```

**Option B: Download dataset (if server available):**

```bash
docker-compose run --rm training python src/data/download_dataset.py
```

---

### Step 17: Train Models with MLflow Tracking

```bash
# Once data is ready, train both models
docker-compose run --rm training python train.py

# This will:
# 1. Train Transformer model (~25 minutes)
# 2. Train Autoencoder model (~15 minutes)
# 3. Log all experiments to MLflow
# 4. Register models in MLflow registry
# 5. Save models to models/ directory
```

**Monitor in MLflow UI:** http://localhost:5000

---

### Step 18: Deploy API (After Training)

```bash
# Build and start API service
docker-compose up -d api

# Check status
docker-compose ps

# Expected:
# studentlife-mlflow   Up (healthy)
# studentlife-api      Up (healthy)
```

**Access API:**
- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

### Step 19: Test API

```bash
# Test health endpoint
curl http://localhost:8000/health

# Make a prediction (requires trained model)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "participant_id": "u00",
    "features": [[...]]  // 24-hour sequence
  }'
```

---

### Step 20: View All Services

```bash
# See all running services
docker-compose ps

# View logs
docker-compose logs -f mlflow  # MLflow logs
docker-compose logs -f api     # API logs

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

---

## 📊 Complete Command Reference

### Outside Container Commands

| Task | Command |
|------|---------|
| Build containers | `docker-compose build` |
| Start MLflow | `docker-compose up -d mlflow` |
| Start all services | `docker-compose up -d` |
| Stop services | `docker-compose down` |
| View logs | `docker-compose logs -f <service>` |
| Check status | `docker-compose ps` |
| Interactive shell | `docker-compose run --rm training bash` |
| Run training | `docker-compose run --rm training python train.py` |
| Rebuild image | `docker-compose build --no-cache training` |

### Inside Container Commands

| Task | Command |
|------|---------|
| Test MLflow | `curl http://mlflow:5000` |
| Check Python | `python --version` |
| List files | `ls -la` |
| Run Python | `python -c "print('test')"` |
| Install package | `pip install <package>` |
| Run training | `python train.py` |
| Test imports | `python -c "import mlflow; print('ok')"` |
| Exit | `exit` |

---

## 🎯 Success Checklist

After completing all steps, verify:

- [ ] Docker images built (14.1GB training, 1.24GB MLflow)
- [ ] MLflow UI accessible at http://localhost:5000
- [ ] MLflow container status: "healthy"
- [ ] Network connectivity verified (training → mlflow)
- [ ] Test experiment visible in MLflow UI
- [ ] All Python dependencies working
- [ ] Can start interactive container shell
- [ ] Can run Python scripts in container
- [ ] Models can be trained (when data available)
- [ ] API can be deployed (after training)

---

## 🐛 Troubleshooting

### Issue: MLflow shows "unhealthy"
```bash
docker logs studentlife-mlflow
docker-compose restart mlflow
```

### Issue: Container can't reach MLflow
```bash
docker network inspect studentlife-phenotyping_studentlife-network
docker-compose down
docker-compose up -d mlflow
```

### Issue: Build fails
```bash
docker-compose build --no-cache
docker system prune -a  # Warning: removes all unused images
```

### Issue: Port already in use
```bash
# Windows
Get-NetTCPConnection -LocalPort 5000, 8000

# Linux/Mac
lsof -i :5000
lsof -i :8000

# Change ports in docker-compose.yml
```

---

## ✨ Summary

**You now have a fully containerized ML platform!**

**From scratch to production in ~15 minutes:**
1. Clone repo (1 min)
2. Enable BuildKit (30 sec)
3. Build containers (10 min with BuildKit)
4. Start MLflow (1 min)
5. Test interactively (3 min)
6. Train models (40 min when data ready)
7. Deploy API (1 min)

**Total setup time:** ~15 minutes (excluding training)

**Works on:** Any machine with Docker (Linux/Mac/Windows, Local/Cloud)

---

## 🚀 Next Steps

1. **Get Data:** Download or provide StudentLife dataset
2. **Train Models:** `docker-compose run --rm training python train.py`
3. **Experiment:** Use MLflow UI to compare runs
4. **Deploy:** `docker-compose up -d api`
5. **Iterate:** Modify hyperparameters, retrain, compare

**Happy experimenting!** 🎯
