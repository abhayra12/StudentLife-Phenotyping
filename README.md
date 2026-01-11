# StudentLife Digital Phenotyping - Behavioral Analytics System

## Project Overview

This project develops a machine learning system for digital phenotyping using passive sensing data from the StudentLife dataset from Dartmouth College. The goal is to analyze student behavioral patterns, detect anomalies, and build interpretable behavioral features from smartphone sensor data.

**Important**: This project uses ONLY sensing data (passive smartphone sensors), excluding self-reported surveys, EMA responses, and academic records.

**Dataset**: [StudentLife Dataset - Sensing Folder Only](https://studentlife.cs.dartmouth.edu/dataset.html)
- 49 participants (undergraduate and graduate students) - verified via data analysis
- ~10 weeks of continuous sensing data (Spring 2013)
- 10 sensor types: activity, audio, bluetooth, conversation, dark, gps, phonecharge, phonelock, wifi, wifi_location
- Total size: 2.39 GB (490 CSV files)
- Focus: Passive behavioral sensing only

## Objectives

1. **Behavioral Clustering**: Group students by similar behavioral patterns
2. **Anomaly Detection**: Identify unusual behavioral changes over time
3. **Activity and Sleep Analysis**: Extract physical activity and sleep patterns
4. **Social Engagement Metrics**: Analyze conversation and co-location patterns
5. **Term Lifecycle Analysis**: How behavior changes over 10-week academic term
6. **Feature Extraction**: Build comprehensive digital phenotyping features
7. **Deploy API**: Production-ready behavioral analytics service

## Tech Stack

**Core:**
- Python 3.13+
- uv (package management)
- FastAPI (API framework)
- PyTorch (deep learning)

**ML Libraries:**
- scikit-learn (classical ML)
- XGBoost, LightGBM, CatBoost (gradient boosting)
- Optuna (hyperparameter tuning)
- SHAP (model interpretability)
- nolds (nonlinear dynamics, entropy)
- mapie (conformal prediction)

**Deep Learning:**
- PyTorch (LSTM, Transformers, Autoencoders)
- Contrastive learning frameworks

**Deployment:**
- Docker (containerization)
- Flower (federated learning)
- Opacus (differential privacy)
- Prometheus (monitoring)

## Project Structure

```
StudntLife-Pheno/
 data/
    raw/              # Raw sensing data
    processed/        # Cleaned and aligned data
    features/         # Engineered features
 notebooks/
    02_eda/          # Exploratory data analysis
    03_preprocessing/ # Data cleaning and alignment
    04_features/      # Feature engineering
    05_modeling/      # Baseline ML models
    06_advanced_ml/   # XGBoost, LightGBM, LSTM
    07_deep_learning/ # Transformers, Autoencoders
    08_production/    # API development and deployment
 src/
    data/            # Data processing modules
    features/        # Feature engineering
    models/          # ML model implementations
    utils/           # Utility functions
 api/                 # FastAPI application
 models/              # Trained model artifacts
 tests/               # Unit and integration tests
 docs/                # Documentation
```

## Getting Started

### Prerequisites
- Python 3.13+
- uv package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/abhayra12/StudentLife-Phenotyping.git
cd StudentLife-Phenotyping
```

2. Create virtual environment and install dependencies:
```bash
uv venv
uv sync
```

3. Activate virtual environment:
```bash
# Windows
.venv\\Scripts\\activate

# Linux/Mac
source .venv/bin/activate
```

### Download Dataset

Download the StudentLife dataset (53 GB):
```bash
# Download from official source
wget https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2

# Extract only sensing folder
tar -xjf dataset.tar.bz2 dataset/sensing/
mv dataset/sensing data/raw/dataset/sensing
```

## Development Workflow

1. **Data Exploration**: Start with notebooks in `02_eda/`
2. **Preprocessing**: Clean and align sensor data
3. **Feature Engineering**: Create behavioral features from raw sensors
4. **Modeling**: Train baseline models (clustering, anomaly detection)
5. **Advanced ML**: XGBoost, LightGBM, LSTM for time series
6. **Deep Learning**: Transformers, contrastive learning, autoencoders
7. **Production**: Build FastAPI service, containerize with Docker
8. **Deploy**: Cloud deployment with federated learning support

## Key Features

### Novel Improvements Beyond Original Research

This project incorporates cutting-edge techniques not present in the original StudentLife paper (2014):

**Advanced Feature Engineering:**
- Circadian cosinor analysis (quantify rhythm strength)
- Behavioral entropy (measure routine predictability)
- Sleep fragmentation index (quality over duration)
- Multi-window aggregations (6h, 12h, 24h, 3d, 7d scales)
- Personalized baselines (individual deviation detection)

**State-of-the-Art ML:**
- XGBoost with Optuna hyperparameter optimization
- Conformal prediction (calibrated uncertainty intervals)
- Temporal SHAP (time-aware feature importance)
- LSTM for sequential patterns
- Transformer architecture with attention mechanisms

**Unsupervised Learning:**
- Contrastive learning for behavioral embeddings
- K-means, hierarchical, and DBSCAN clustering
- Isolation Forest and One-Class SVM for anomaly detection

**Production-Ready Deployment:**
- FastAPI REST API with Swagger documentation
- Docker containerization with multi-stage builds
- Federated learning for privacy preservation
- Differential privacy with Opacus
- Prometheus monitoring

## Running the API

```bash
# Development server
uv run uvicorn api.main:app --reload

# Production server
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for Swagger UI

## Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=html
```

## Docker

```bash
# Build image
docker build -t studentlife-api .

# Run container
docker run -p 8000:8000 studentlife-api

# Or use docker-compose
docker-compose up
```

## Contributing

This is a learning project following a structured curriculum. See `plan.md` for the complete learning roadmap.

## Dataset Citation

If you use this implementation:

```
Wang, R., Chen, F., Chen, Z., Li, T., Harari, G., Tignor, S., Zhou, X., Ben-Zeev, D., & Campbell, A. T. (2014).
StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones.
In Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '14).
```

## License

This project is for educational purposes. The StudentLife dataset has its own terms of use.

## Repository

https://github.com/abhayra12/StudentLife-Phenotyping

---

**Note**: This implementation focuses on behavioral analytics from passive sensing. While we can't predict depression/stress directly (no survey labels), we build a comprehensive digital phenotyping system that extracts meaningful behavioral features useful for research and future supervised learning when labels become available.
