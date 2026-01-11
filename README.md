# 🎓 StudentLife Digital Phenotyping - ML Project

## 📖 Project Overview

This project develops a machine learning system for digital phenotyping using **passive sensing data only** from the StudentLife dataset from Dartmouth College. The goal is to analyze student behavioral patterns, detect anomalies, and build interpretable behavioral features from smartphone sensor data.

**Important**: This project uses **ONLY sensing data** (passive smartphone sensors), excluding self-reported surveys, EMA responses, and academic records.

**Dataset**: [StudentLife Dataset - Sensing Folder Only](https://studentlife.cs.dartmouth.edu/dataset.html)
- 48 participants (undergraduate & graduate students)
- 10 weeks of continuous sensing data (Spring 2013)
- 10 sensor types: activity, audio, bluetooth, conversation, dark, gps, phonecharge, phonelock, wifi, wifi_location
- Focus: Passive behavioral sensing only

## 🎯 Objectives

1. **Behavioral Clustering**: Group students by similar behavioral patterns
2. **Anomaly Detection**: Identify unusual behavioral changes over time
3. **Activity & Sleep Analysis**: Extract physical activity and sleep patterns
4. **Social Engagement Metrics**: Analyze conversation and co-location patterns
5. **Term Lifecycle Analysis**: How behavior changes over 10-week academic term
6. **Feature Extraction**: Build comprehensive digital phenotyping features
7. **Deploy API**: Production-ready behavioral analytics service

## 🛠️ Tech Stack

- **Languages**: Python 3.13+
- **Package Manager**: uv (modern, fast)
- **Data**: Pandas, NumPy, SciPy
- **ML**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **DL**: PyTorch
- **Visualization**: Matplotlib, Seaborn, Plotly
- **API**: FastAPI, Uvicorn
- **Deployment**: Docker
- **Experimentation**: Optuna

## 📁 Project Structure

```
├── data/              # Dataset (raw, processed, features)
├── notebooks/         # Jupyter notebooks (EDA, modeling, evaluation)
├── src/               # Source code (reusable modules)
├── models/            # Trained models and checkpoints
├── api/               # FastAPI deployment
├── tests/             # Unit and integration tests
├── configs/           # Configuration files
├── docs/              # Documentation
├── pyproject.toml     # Dependencies and metadata
└── uv.lock            # Locked dependency versions
```

## 🚀 Getting Started

### Prerequisites
- Python 3.13 or higher
- uv (install: `pip install uv`)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd StudntLife-Pheno
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   # Sync all dependencies from uv.lock
   python -m uv sync
   
   # Verify installation
   python -m uv pip check
   ```

3. **Download dataset**
   ```bash
   # Download from: https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2
   # Extract to data/raw/studentlife/
   ```

## 📊 Usage

### Activate Virtual Environment (Optional)
```bash
# If you prefer activating venv:
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Then run commands directly:
python src/script.py
jupyter notebook
```

### Or Use `uv run` (Recommended - No Activation Needed)
```bash
# Run any command without activating venv:
uv run jupyter notebook
uv run python src/models/train_model.py --config configs/experiment_config.yaml
uv run uvicorn api.main:app --reload
uv run pytest
```

### Example Commands
```bash
# Start Jupyter for notebooks
uv run jupyter notebook

# Train a model
uv run python src/models/train_model.py

# Run API server  
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
uv run pytest tests/
```

## 📈 Model Performance

_(To be updated with results)_

## 🧑‍💻 Development

- **Progress Tracking**: See `development_guide.md` for detailed progress
- **Current Task**: See `CURRENT_TASK.md` for active task instructions
- **Learning Plan**: See `plan.md` for comprehensive 14-phase roadmap

## 📝 License

MIT License

## 🙏 Acknowledgments

- StudentLife Dataset: Wang et al., Dartmouth College
- Inspired by: [ML Zoomcamp FastAPI+UV Workshop](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/05-deployment/workshop)

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Status**: 🔄 In Development | **Started**: 2026-01-11
