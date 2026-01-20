# Docker Build Optimization Guide

## ⚡ Reducing Build Time from 22+ Minutes to ~5 Minutes

### Current Issues:
- Training container: 14.1 GB (very large)
- Build time: 22+ minutes
- Export time: Several minutes
- Re-downloading dependencies on every rebuild

---

## 🚀 Optimization Strategies

### 1. Enable Docker BuildKit (Fastest Win!)

**BuildKit provides:**
- Parallel layer building
- Better caching
- Faster dependency resolution
- Progress visibility

**Add to your setup:**

#### Windows PowerShell:
```powershell
# Add to your PowerShell profile or run before each build
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1

# Then build
docker-compose build
```

#### Linux/Mac:
```bash
# Add to ~/.bashrc or ~/.zshrc
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Or prepend to build command
DOCKER_BUILDKIT=1 docker-compose build
```

**Time saved: ~30-40%** (7-8 minutes)

---

### 2. Use BuildKit Cache Mounts (Huge Impact!)

Update `Dockerfile.train` to use cache mounts for pip/uv:

**Before:**
```dockerfile
RUN uv sync --frozen
```

**After (with cache mount):**
```dockerfile
# Use BuildKit cache mount to persist pip cache
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv sync --frozen
```

**Benefits:**
- Downloaded packages cached between builds
- Wheels reused
- Only changed dependencies re-downloaded

**Add this to Dockerfile.train:**

```dockerfile
# Dockerfile for Training - OPTIMIZED VERSION
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies (these rarely change, cache this layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (rarely changes)
RUN pip install --no-cache-dir uv

# Copy ONLY dependency files first (better caching)
COPY pyproject.toml uv.lock ./

# Install dependencies with cache mount (OPTIMIZED!)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv sync --frozen

# Copy source code LAST (changes most frequently)
COPY src/ src/
COPY train.py ./

# Create directories
RUN mkdir -p data models mlruns reports

# Environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI="http://mlflow:5000"

EXPOSE 8888

CMD ["python", "train.py"]
```

**Time saved: ~50% on rebuilds** (11 minutes)

---

### 3. Pre-Download Base Images

```bash
# Download base images before building (parallel downloads)
docker pull python:3.13-slim

# Then build
docker-compose build
```

**Time saved: ~2-3 minutes**

---

### 4. Build Only What Changed

Instead of rebuilding everything:

```bash
# Build only specific services
docker-compose build mlflow      # ~2 min (small)
docker-compose build training    # ~7 min (large)
docker-compose build api         # ~3 min

# Or build in parallel (if you have resources)
docker-compose build --parallel
```

---

### 5. Use Docker Layer Caching Effectively

**Current Dockerfile.train order (GOOD ✓):**
1. System dependencies (rarely change)
2. uv installation (rarely change)
3. Dependency files (change occasionally)
4. Install dependencies (change occasionally)
5. Source code (change frequently)

**This is already optimal!** The problem is cache isn't being used.

---

### 6. Reduce Image Size (Faster Export)

**Current: 14.1 GB → Target: 8-10 GB**

Add these optimizations to Dockerfile.train:

```dockerfile
# After uv sync, clean up unnecessary files
RUN uv sync --frozen && \
    # Remove cached wheels
    rm -rf /root/.cache/* && \
    # Remove unnecessary files in virtual env
    find /app/.venv -name "*.pyc" -delete && \
    find /app/.venv -name "__pycache__" -delete && \
    # Remove test files
    find /app/.venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
```

**Size reduction: ~30%** (14.1 GB → ~10 GB)
**Export time saved: ~3 minutes**

---

### 7. Use Multi-Stage Build for Training (Advanced)

Create a builder stage that compiles dependencies:

```dockerfile
# STAGE 1: Builder
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv sync --frozen

# STAGE 2: Runtime
FROM python:3.13-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the virtual environment from builder
COPY --from=builder /build/.venv /app/.venv

# Copy source code
COPY src/ src/
COPY train.py ./

# Create directories
RUN mkdir -p data models mlruns reports

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI="http://mlflow:5000"

EXPOSE 8888

CMD ["python", "train.py"]
```

**Benefits:**
- Build tools not in final image
- Smaller final image (~9-10 GB vs 14.1 GB)
- Faster export

**Time saved: ~4 minutes on export**

---

### 8. Use .dockerignore Aggressively

Already optimized, but ensure these are excluded:

```
# .dockerignore additions for faster builds
**/__pycache__
**/*.pyc
**/*.pyo
**/*.pyd
.Python
*.egg-info
dist/
build/
.pytest_cache/
.coverage
htmlcov/
```

---

## 📊 Complete Optimized Build Process

### Updated Build Commands:

**Windows PowerShell:**
```powershell
# Enable BuildKit
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1

# Pre-download base image
docker pull python:3.13-slim

# Build with BuildKit cache
docker-compose build --parallel

# Or build specific service
docker-compose build training
```

**Linux/Mac:**
```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Pre-download base image
docker pull python:3.13-slim

# Build with BuildKit cache
docker-compose build --parallel
```

---

## ⏱️ Expected Build Times

### First Build (No Cache):
| Component | Without Optimization | With Optimization | Savings |
|-----------|---------------------|-------------------|---------|
| MLflow | 2 min | 1.5 min | 25% |
| Training | 20 min | 7 min | **65%** |
| API | 3 min | 2 min | 33% |
| **Total** | **25 min** | **~10 min** | **60%** |

### Rebuild (With Cache):
| Component | Without Optimization | With Optimization | Savings |
|-----------|---------------------|-------------------|---------|
| MLflow | 2 min | 10 sec | **92%** |
| Training | 20 min | 2 min | **90%** |
| API | 3 min | 20 sec | **89%** |
| **Total** | **25 min** | **~3 min** | **88%** |

### Code-Only Changes (Best Case):
With proper layer caching:
- Change Python code → **~15 seconds** (just copy layer)
- Change dependencies → **~2 minutes** (re-install only)
- Change Dockerfile → **~7 minutes** (rebuild from change point)

---

## 🎯 Recommended Quick Wins

**Implement these in order of impact:**

### Priority 1 (Immediate - No Code Changes):
```powershell
# 1. Enable BuildKit
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1

# 2. Pre-pull base image
docker pull python:3.13-slim

# 3. Build
docker-compose build
```
**Time saved: ~8 minutes** (22 min → 14 min)

### Priority 2 (5 Min Code Change):
Update `Dockerfile.train` to add cache mounts:
```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen
```
**Time saved: ~11 minutes on rebuilds** (22 min → 3 min on subsequent builds)

### Priority 3 (10 Min Code Change):
Implement multi-stage build for training container.
**Image size reduced: 14.1 GB → ~9 GB**
**Export time saved: ~4 minutes**

---

## 🔧 Add to SETUP_GUIDE.md

Add this section after **Step 3: Build All Containers**:

```markdown
### Step 3.5: Build Optimization (Optional but Recommended)

**Enable BuildKit for faster builds:**

```powershell
# Windows PowerShell
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1
```

```bash
# Linux/Mac
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

**Benefits:**
- First build: ~10 minutes (vs 25 minutes)
- Rebuilds: ~3 minutes (vs 22 minutes)
- Code-only changes: ~15 seconds

**Pre-download base image (optional):**
```bash
docker pull python:3.13-slim
```

Then proceed with build:
```bash
docker-compose build
```
```

---

## 📝 Summary

**Current State:**
- Build time: 22+ minutes
- Image size: 14.1 GB
- Rebuild time: 22+ minutes (no caching)

**With Optimizations:**
- First build: ~10 minutes (**-12 min, 55% faster**)
- Image size: ~9 GB (**-5 GB, 36% smaller**)
- Rebuild: ~3 minutes (**-19 min, 86% faster**)
- Code changes: ~15 seconds (**-22 min, 99% faster**)

**Total time saved per build: ~12-19 minutes!**

---

## 🚀 Next Steps

1. **Now:** Enable BuildKit (no code changes required)
2. **Today:** Add cache mounts to Dockerfile.train
3. **This week:** Implement multi-stage build
4. **Update:** Add optimization section to SETUP_GUIDE.md

**Start with step 1 right now - it's literally one environment variable!**
