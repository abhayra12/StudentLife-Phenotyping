# Docker Compose Quick Start Guide

## 🚀 Quick Start - Easiest Approach

### 1. **Start Everything (MLflow + API)**
```powershell
docker-compose up -d mlflow api
```

Access:
- **MLflow UI**: http://localhost:5000
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

### 2. **Train Models with MLflow Tracking**
```powershell
docker-compose run --rm training python train.py
```

### 3. **View Logs**
```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mlflow
docker-compose logs -f api
```

### 4. **Stop All Services**
```powershell
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

---

## 📦 What's Included

### Three Services:

1. **mlflow** - Experiment tracking server (Port 5000)
   - Stores experiments, metrics, parameters
   - Model registry
   - Artifact storage

2. **api** - FastAPI inference server (Port 8000)
   - Serves predictions
   - Loads models from MLflow
   - Auto-restarts on failure

3. **training** - On-demand training container
   - Runs training scripts
   - Logs to MLflow automatically
   - Access to all data and models

---

## 🔨 Common Commands

### Build Images
```powershell
# Build all images
docker-compose build

# Build specific service
docker-compose build mlflow
docker-compose build api
docker-compose build training
```

### Start Services
```powershell
# Start in background
docker-compose up -d

# Start with logs
docker-compose up

# Start specific services
docker-compose up -d mlflow api
```

### Stop Services
```powershell
# Stop (keeps containers)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop, remove containers and volumes
docker-compose down -v
```

### View Status
```powershell
# Service status
docker-compose ps

# Resource usage
docker stats
```

---

## 🎯 Complete Workflow

### First Time Setup
```powershell
# 1. Copy environment template
Copy-Item .env.example .env

# 2. Build all containers
docker-compose build

# 3. Start services
docker-compose up -d mlflow api

# 4. Wait for health checks (30 seconds)
Start-Sleep -Seconds 30

# 5. Check status
docker-compose ps
```

### Training Workflow
```powershell
# 1. Ensure MLflow is running
docker-compose up -d mlflow

# 2. Run training
docker-compose run --rm training python train.py

# 3. View results in MLflow UI
# Open http://localhost:5000
```

### Deployment Workflow
```powershell
# 1. Train models first
docker-compose run --rm training python train.py

# 2. Start API with trained models
docker-compose up -d api

# 3. Test API
curl http://localhost:8000/health
```

---

## 🔧 Advanced Usage

### Interactive Training Shell
```powershell
docker-compose run --rm training bash
```

### Custom Training Script
```powershell
docker-compose run --rm training python src/analysis/modeling/09_transformer.py
```

### View MLflow Artifacts
```powershell
# Artifacts are stored in named volume
docker volume inspect studentlife-phenotyping_mlflow_artifacts

# Access artifacts directory
docker-compose exec mlflow ls -la /mlflow/artifacts
```

### Scale API (Multiple Instances)
```powershell
docker-compose up -d --scale api=3
```

### Override Resource Limits
```powershell
# Edit docker-compose.yml
# Under api.deploy.resources.limits
cpus: '4'      # Increase from 2
memory: 4G     # Increase from 2G
```

---

## 🐛 Troubleshooting

### MLflow Not Starting
```powershell
# Check logs
docker-compose logs mlflow

# Restart
docker-compose restart mlflow

# Clean start
docker-compose down -v
docker-compose up -d mlflow
```

### API Can't Connect to MLflow
```powershell
# Verify network
docker network ls | grep studentlife

# Check MLflow health
docker-compose exec mlflow curl -f http://localhost:5000/health

# Restart with dependency check
docker-compose down
docker-compose up -d
```

### Training Container Fails
```powershell
# Check logs
docker-compose logs training

# Run interactively to debug
docker-compose run --rm training bash
python train.py
```

### Port Already in Use
```powershell
# Find process using port
Get-NetTCPConnection -LocalPort 5000, 8000

# Stop conflicting service or change ports in docker-compose.yml
ports:
  - "5001:5000"  # Change host port
```

---

## 📊 Monitoring

### Check Health Status
```powershell
# All services
docker-compose ps

# Specific service health
docker inspect --format='{{.State.Health.Status}}' studentlife-mlflow
docker inspect --format='{{.State.Health.Status}}' studentlife-api
```

### View Resource Usage
```powershell
# Real-time stats
docker stats studentlife-mlflow studentlife-api

# Disk usage
docker system df
```

---

## 🗂️ Data Persistence

### Volumes
- `mlflow_data` - MLflow backend database (SQLite)
- `mlflow_artifacts` - Model artifacts and logs

### Local Mounts
- `./models` - Trained model files
- `./data` - Training datasets
- `./reports` - Generated reports
- `./src` - Source code (live reload)

### Backup Volumes
```powershell
# Backup MLflow data
docker run --rm -v studentlife-phenotyping_mlflow_data:/data -v ${PWD}:/backup ubuntu tar czf /backup/mlflow_backup.tar.gz /data

# Restore
docker run --rm -v studentlife-phenotyping_mlflow_data:/data -v ${PWD}:/backup ubuntu tar xzf /backup/mlflow_backup.tar.gz -C /
```

---

## 🎓 Example Workflows

### Complete Training & Deployment
```powershell
# 1. Start MLflow
docker-compose up -d mlflow

# 2. Train transformer
docker-compose run --rm training python src/analysis/modeling/09_transformer.py

# 3. Train autoencoder
docker-compose run --rm training python src/analysis/modeling/08_autoencoder.py

# 4. Verify models in MLflow UI
# Open http://localhost:5000

# 5. Deploy API
docker-compose up -d api

# 6. Test predictions
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @test_request.json
```

### Development Mode (Hot Reload)
```powershell
# Source code is mounted, changes reflect immediately
docker-compose up api

# Edit src/api/main.py
# API auto-reloads
```

---

## 🚨 Production Checklist

- [ ] Update `.env` with production settings
- [ ] Use PostgreSQL instead of SQLite for MLflow backend
- [ ] Configure S3/Azure/GCS for artifact storage
- [ ] Set up proper secrets management (not in .env)
- [ ] Enable HTTPS/TLS
- [ ] Configure proper logging aggregation
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Implement backup strategy
- [ ] Use `--scale` for API redundancy
- [ ] Add nginx reverse proxy
