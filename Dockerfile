FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
# Ideally we export requirements.txt from pyproject.toml, but we can install directly if we have a tool or just list main ones
# For simplicity, let's assume we install the main packages manually or via pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy scikit-learn pydantic python-dotenv

# Copy source code and models
COPY src/ src/
COPY models/ models/
COPY .env .env

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
