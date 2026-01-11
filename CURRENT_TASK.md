# 📋 CURRENT TASK: Task 1.1 - Initialize Project Structure

**Status**: 🔄 **IN PROGRESS**  
**Assigned Date**: 2026-01-11  
**Phase**: PHASE 1 - PROJECT SETUP & ENVIRONMENT

---

## 🎯 Task Overview

**Objective**: Set up a professional ML project structure with modern uv-based dependency management, proper organization, and initial documentation.

**Why This Matters**: Proper project organization from day one is crucial for maintainability, reproducibility, and collaboration. This is how production ML teams structure their projects.

**Estimated Time**: 2-3 hours

---

## ✅ Completed Steps

### Step 1: Create Virtual Environment with `uv` ✅
- Installed uv: `pip install uv`
- Created venv: `python -m uv venv`
- Activated: `.venv\Scripts\activate`
- Fixed PowerShell execution policy

### Step 2: Migrate to Modern UV Workflow ✅
- Initialized uv project: `python -m uv init --no-readme --no-pin-python`
- Created `pyproject.toml` (modern Python packaging)
- Added production dependencies: `python -m uv add pandas numpy scipy matplotlib seaborn plotly scikit-learn xgboost lightgbm catboost torch statsmodels tqdm joblib pyyaml python-dotenv fastapi uvicorn pydantic optuna`
- Added dev dependencies: `python -m uv add --dev jupyter pytest pytest-cov ipykernel`
- Created `uv.lock` for reproducibility

---

## 🔄 Remaining Steps

### Step 3: Create Project Directory Structure

Create the following directory structure using PowerShell:

```powershell
# Create main directories
mkdir data, notebooks, src, models, api, tests, configs, docs

# Create data subdirectories
mkdir data\raw, data\processed, data\features, data\external

# Create notebook subdirectories
mkdir notebooks\01_exploration, notebooks\02_preprocessing, notebooks\03_feature_engineering, notebooks\04_modeling, notebooks\05_evaluation

# Create src subdirectories
mkdir src\data, src\features, src\models, src\visualization, src\utils

# Create model subdirectories
mkdir models\saved_models, models\checkpoints
```

**Expected Result**:
```
StudntLife-Pheno/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── external/
├── notebooks/
│   ├── 01_exploration/
│   ├── 02_preprocessing/
│   ├── 03_feature_engineering/
│   ├── 04_modeling/
│   └── 05_evaluation/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── visualization/
│   └── utils/
├── models/
│   ├── saved_models/
│   └── checkpoints/
├── api/
├── tests/
├── configs/
└── docs/
```

---

### Step 4: Create README.md

Create a comprehensive README with this template:

```markdown
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
- **API**: FastAPI, Uvicorn
- **Deployment**: Docker
- **Experimentation**: Optuna

## 📁 Project Structure

\`\`\`
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
\`\`\`

## 🚀 Getting Started

### Prerequisites
- Python 3.13 or higher
- uv (install: \`pip install uv\`)

### Installation

1. **Clone the repository**
   \`\`\`bash
   git clone <your-repo-url>
   cd StudntLife-Pheno
   \`\`\`

2. **Create virtual environment and install dependencies**
   \`\`\`bash
   # Sync all dependencies from uv.lock
   python -m uv sync
   
   # Verify installation
   python -m uv pip check
   \`\`\`

3. **Download dataset**
   \`\`\`bash
   # Download from: https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2
   # Extract to data/raw/studentlife/
   \`\`\`

## 📊 Usage

### Run Notebooks
\`\`\`bash
python -m uv run jupyter notebook
\`\`\`

### Train Models
\`\`\`bash
python -m uv run python src/models/train_model.py --config configs/experiment_config.yaml
\`\`\`

### Run API Server
\`\`\`bash
python -m uv run uvicorn api.main:app --reload
\`\`\`

### Run Tests
\`\`\`bash
python -m uv run pytest
\`\`\`

## 📈 Model Performance

_(To be updated with results)_

## 🧑‍💻 Development

See \`development_guide.md\` for detailed progress tracking and setup instructions.

## 📝 License

_(Add license information)_

## 🙏 Acknowledgments

- StudentLife Dataset: Wang et al., Dartmouth College
- Inspired by: [ML Zoomcamp FastAPI+UV Workshop](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/05-deployment/workshop)

## 📧 Contact

_(Your contact information)_
```

**Customize**:
- Add your name/contact
- Adjust based on your preferences
- Keep it professional!

---

### Step 5: Verify Setup

Run these checks before submitting:

1. **Check directory structure**:
   ```bash
   tree /F  # Windows
   # or: Get-ChildItem -Recurse | Select-Object FullName
   ```

2. **Verify packages installed**:
   ```python
   python -m uv run python -c "import pandas; import numpy; import sklearn; import torch; print('✅ All imports successful!')"
   ```

3. **Check files**:
   ```bash
   # Should have:
   # - pyproject.toml
   # - uv.lock
   # - README.md
   # - development_guide.md
   # - All directories created
   ```

4. **Check Git status**:
   ```bash
   git status
   ```
   Verify that `data/`, `models/`, `.venv/` are NOT shown (should be ignored)

5. **Create first commit**:
   ```bash
   git add .
   git commit -m "feat: Initialize project with modern uv workflow and directory structure"
   ```

---

## 📤 Submission

When you've completed all steps:

1. **Take screenshots** of:
   - Your directory structure
   - Successful package imports
   - Git status showing proper ignores

2. **Update development_guide.md**:
   - Add Task 1.1 to "Completed Tasks" section
   - Document any challenges faced
   - Note lessons learned

3. **Ready for next task**: Task 1.2 - Setup Development Environment

---

## 💡 Hints & Tips

1. **Directory structure**: Already mostly created (api, configs, data, etc.), just add subdirectories
2. **README**: Use the template, customize to your style
3. **Git commit**: Use conventional commits format: `feat:`, `fix:`, `docs:`, etc.
4. **Torch still downloading?**: That's fine, it's in background

---

## ❓ Questions?

If you're stuck or unsure, ask me! Common questions:

- "How do I verify torch installed?" → `python -m uv pip show torch`
- "What if git status shows too many files?" → Check .gitignore is working
- "Should I commit uv.lock?" → Yes! It's critical for reproducibility

---

**Good luck! You're almost done with Task 1.1! 🚀**

---

_Last Updated: 2026-01-11_
