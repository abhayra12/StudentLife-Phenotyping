# 🎓 StudentLife Digital Phenotyping - ML Project

## 📖 Project Overview

This project develops a machine learning system for digital phenotyping using the StudentLife dataset from Dartmouth College. The goal is to predict student mental health states, stress levels, and academic performance from smartphone sensor data.

**Dataset**: [StudentLife Dataset](https://studentlife.cs.dartmouth.edu/dataset.html)
- 48 participants (undergraduate & graduate students)
- 10 weeks of continuous sensing data (Spring 2013)
- 53 GB of sensor data
- 32,000+ self-reports
- Pre/post psychological surveys

## 🎯 Objectives

1. Predict depression risk from passive smartphone sensing
2. Forecast stress levels and mood states
3. Predict academic performance (GPA)
4. Build interpretable, clinically-relevant models
5. Deploy as a production-ready API service

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

### Run Notebooks
```bash
python -m uv run jupyter notebook
```

### Train Models
```bash
python -m uv run python src/models/train_model.py --config configs/experiment_config.yaml
```

### Run API Server
```bash
python -m uv run uvicorn api.main:app --reload
```

### Run Tests
```bash
python -m uv run pytest
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
