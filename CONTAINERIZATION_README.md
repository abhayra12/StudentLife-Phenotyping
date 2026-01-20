# StudentLife-Phenotyping: MLflow + Docker Containerization

## ✅ What's Been Implemented

This branch (`containerize`) now includes a complete containerization setup with MLflow experiment tracking:

### 1. **MLflow Integration** ✓
- ✅ Added MLflow dependencies to `pyproject.toml`
- ✅ Created `src/mlflow_config.py` for centralized MLflow configuration
- ✅ Integrated MLflow tracking into:
  - `src/analysis/modeling/09_transformer.py` (Transformer model)
  - `src/analysis/modeling/08_autoencoder.py` (Autoencoder model)
- ✅ Automatic logging of:
  - Hyperparameters
  - Training metrics (loss, MAE, MSE)
  - Model artifacts
  - Visualizations
  - Model registry with versioning

### 2. **Docker Containerization** ✓
- ✅ **`Dockerfile`** - Multi-stage production API container
- ✅ **`Dockerfile.train`** - Training container with dev dependencies
- ✅ **`Dockerfile.mlflow`** - MLflow tracking server
- ✅ **`docker-compose.yml`** - Complete orchestration setup
- ✅ **`.dockerignore`** - Optimized build context
- ✅ **`.env.example`** - Environment configuration template

### 3. **Helper Scripts** ✓
- ✅ `DOCKER_GUIDE.md` - Comprehensive usage guide
- ✅ Ready-to-use Docker Compose configuration

---

## 🚀 Quick Start

### **Option 1: Using Docker Compose (Recommended)**

```powershell
# 1. Install MLflow dependencies first (for local development)
uv sync

# 2. Start MLflow tracking server
docker-compose up -d mlflow

# 3. Access MLflow UI
# Open http://localhost:5000 in browser

# 4. Train models with MLflow tracking (optional - can run locally)
docker-compose run --rm training python train.py

# 5. Deploy API (when models are trained)
docker-compose up -d api

# 6. Access API
# Open http://localhost:8000/docs
```

### **Option 2: Local Development (Without Docker)**

```powershell
# 1. Install dependencies
uv sync

# 2. Start local MLflow server
mlflow server --host 0.0.0.0 --port 5000

# 3. Train models (in separate terminal)
python train.py

# 4. View experiments in MLflow UI
# Open http://localhost:5000
```

---

## 📁 Architecture

### Docker Services

```
┌──────────────────────────────────────────────────┐
│              Docker Compose Setup                 │
├──────────────────────────────────────────────────┤
│                                                   │
│  ┌─────────────┐  ┌──────────────┐              │
│  │   MLflow    │  │     API      │              │
│  │  Server     │  │   Service    │              │
│  │  Port 5000  │  │  Port 8000   │              │
│  └──────┬──────┘  └──────┬───────┘              │
│         │                 │                       │
│         └────┬────────────┘                       │
│              │                                    │
│    ┌─────────▼─────────┐                        │
│    │     Training      │   (On-demand)          │
│    │    Container      │                         │
│    └───────────────────┘                        │
│                                                   │
│  Volumes:                                        │
│  - mlflow_data (backend DB)                     │
│  - mlflow_artifacts (models, plots)             │
│  - ./models (local models)                      │
│  - ./data (datasets)                            │
└──────────────────────────────────────────────────┘
```

### MLflow Integration

```python
# Models automatically log to MLflow:

📊 Transformer Training:
  ├─ Hyperparameters (d_model, nhead, lr, ...)
  ├─ Metrics (train_mae, val_mae per epoch)
  ├─ Best model artifact
  └─ Model registered as "behavioral-transformer"

🔍 Autoencoder Training:
  ├─ Hyperparameters (latent_dim, epochs, ...)
  ├─ Metrics (train_mse, val_mse, thresholds)
  ├─ Visualizations (error distributions, latent space)
  ├─ Anomaly detection results
  └─ Model registered as "behavioral-autoencoder"
```

---

## 📖 Usage Examples

### Train Models Locally with MLflow

```powershell
# 1. Start MLflow UI
mlflow ui --port 5000

# 2. Run training
python train.py

# Result: All metrics logged to http://localhost:5000
```

### Train Models in Docker

```powershell
# Complete workflow
docker-compose up -d mlflow
docker-compose run --rm training python train.py

# View results
# Open http://localhost:5000
```

### Compare Experiments

```python
# MLflow automatically tracks all runs
# 1. Go to http://localhost:5000
# 2. Click on "studentlife-phenotyping" experiment
# 3. Compare metrics across runs
# 4. Select best model and transition to "Production"
```

### Load Models from MLflow Registry

```python
import mlflow

# Load latest production model
model = mlflow.pytorch.load_model("models:/behavioral-transformer/Production")

# Load specific version
model = mlflow.pytorch.load_model("models:/behavioral-transformer/1")
```

---

## 🔄 Complete Workflow

### End-to-End Training → Deployment

```powershell
# 1. Start infrastructure
docker-compose up -d mlflow

# 2. Train transformer
docker-compose run --rm training python src/analysis/modeling/09_transformer.py

# 3. Train autoencoder  
docker-compose run --rm training python src/analysis/modeling/08_autoencoder.py

# 4. Review experiments in MLflow UI
# http://localhost:5000

# 5. Promote best model to Production
# (Via MLflow UI or API)

# 6. Deploy API with production models
docker-compose up -d api

# 7. Test predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"participant_id": "u00", "features": [...]}'
```

---

## 📊 MLflow Features Available

### Experiment Tracking
- ✅ Automatic parameter logging
- ✅ Metric logging per epoch
- ✅ Artifact logging (plots, results)
- ✅ Model versioning

### Model Registry
- ✅ Register models with names
- ✅ Version management
- ✅ Stage transitions (Staging → Production)
- ✅ Model lineage tracking

### Visualization
- ✅ Metric comparison charts
- ✅ Parameter importance
- ✅ Artifact viewing (plots, CSVs)

---

## 🎯 Next Steps

### To Get Started:
1. **Read** `DOCKER_GUIDE.md` for detailed commands
2. **Run** `docker-compose up -d mlflow` to start tracking server
3. **Train** models with `python train.py` (locally or in Docker)
4. **Explore** MLflow UI at http://localhost:5000

### For Production:
1. Configure PostgreSQL backend for MLflow (instead of SQLite)
2. Set up S3/Azure/GCS for artifact storage
3. Add authentication to MLflow UI
4. Configure HTTPS/TLS for API
5. Set up monitoring and logging

---

## 📚 Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **Docker Compose Docs**: https://docs.docker.com/compose/
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/
- **Project README**: See `README.md` for full project details

---

## 🐛 Troubleshooting

### MLflow not accessible
```powershell
# Check if container is running
docker-compose ps mlflow

# Check logs
docker-compose logs mlflow

# Restart
docker-compose restart mlflow
```

### Models not loading in API
```powershell
# Verify models are trained and registered
# Check MLflow UI → Models section

# Verify models directory
ls models/

# Check API logs
docker-compose logs api
```

### Training fails
```powershell
# Check data directory exists
ls data/processed/

# Run interactively to debug
docker-compose run --rm training bash
python train.py
```

For more help, see `DOCKER_GUIDE.md` or check the logs:
```powershell
docker-compose logs -f
```

---

## 📝 Summary

**What you have now:**
- ✅ Complete MLflow integration for experiment tracking
- ✅ Docker containers for training, API, and MLflow server
- ✅ Docker Compose orchestration for easy deployment
- ✅ Model registry with versioning
- ✅ Production-ready API with health checks
- ✅ Comprehensive documentation

**Start experimenting with:**
```powershell
docker-compose up -d mlflow
python train.py
# Open http://localhost:5000
```

Enjoy your containerized ML workflow! 🚀
