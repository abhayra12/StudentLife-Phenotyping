# Multi-stage Dockerfile for StudentLife-Phenotyping API
# Optimized for production deployment with minimal image size

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies to a virtual environment
RUN uv sync --no-dev --frozen

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.13-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application code
COPY src/ src/
COPY models/ models/
COPY .env .env

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI="http://mlflow:5000"

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
