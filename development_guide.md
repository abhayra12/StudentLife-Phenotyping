# 📖 StudentLife Phenotyping - Developer Manual

**Purpose**: This guide provides the technical standards, workflows, and architecture for the StudentLife Phenotyping project. It is the primary reference for developers contributing to the codebase.

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and enter
git clone https://github.com/abhayra12/StudentLife-Phenotyping.git
cd StudentLife-Phenotyping

# Install dependencies (using uv for speed)
pip install uv
python -m uv venv
.venv\Scripts\activate
python -m uv pip install -r requirements.txt
```

### 2. Run Data Pipeline
To process raw data into the final train/test split:

```bash
# Step 1: Clean raw sensor data
python src/data/run_cleaning.py

# Step 2: Align sensors to hourly grid
python src/data/run_alignment.py

# Step 3: Create final Parquet datasets
python src/data/create_final_dataset.py
```

**Output**: `data/processed/train.parquet`, `val.parquet`, `test.parquet`

---

## 🏗️ Project Architecture

### Data Flow
1.  **Raw Data** (`data/raw/dataset/sensing/`):
    - Original CSV files from 10 sensors (activity, audio, gps, etc.).
    - *Read-only*.

2.  **Cleaning** (`src/data/cleaning.py`):
    - Validates timestamps and values.
    - Removes outliers.
    - Output: `data/processed/cleaned/` (CSV).

3.  **Alignment** (`src/data/alignment.py`):
    - Resamples all sensors to a common **hourly grid**.
    - Aggregates high-frequency data (e.g., audio features averaged per hour).
    - Output: `data/processed/aligned/` (CSV).

4.  **Final Dataset** (`src/data/create_final_dataset.py`):
    - Stacks all participants.
    - Adds temporal features (week of term, day of week).
    - Splits chronologically:
        - **Train**: Weeks 1-6
        - **Val**: Weeks 7-8
        - **Test**: Weeks 9-10
    - Output: `data/processed/*.parquet`.

---

## 🛠️ Development Workflow

### Adding a New Feature
1.  **Create Branch**: `git checkout -b feat/feature-name`
2.  **Implement**: Write code in `src/`.
3.  **Test**: Add unit tests in `tests/` and run `pytest`.
4.  **Verify**: Run the pipeline to ensure no regressions.
5.  **Commit**: `git commit -m "feat: description"`

### Adding a New Sensor
1.  Update `src/data/cleaning.py`: Add validation logic.
2.  Update `src/data/alignment.py`: Add aggregation logic (e.g., sum vs mean).
3.  Run `run_alignment.py` to regenerate aligned data.

---

## 📏 Coding Standards

### 1. Relative Paths ONLY
> [!IMPORTANT]
> **NEVER use absolute paths** (e.g., `C:\Users\...`).
> Always use `pathlib` relative to the project root.

**Correct**:
```python
from pathlib import Path
DATA_DIR = Path('data/processed')
```

**Incorrect**:
```python
DATA_DIR = 'C:/Users/abhay/project/data'
```

### 2. Python Style
- **Type Hints**: Encouraged for function signatures.
- **Docstrings**: Required for all modules and major functions.
- **Imports**: Group standard lib, third-party, and local imports.

### 3. Notebooks
- Keep notebooks in `notebooks/`.
- Do not commit large outputs or heavy plots if possible.
- Use `sys.path.append('..')` or similar to import from `src/`.

---

## 🐛 Troubleshooting
See [ISSUES_LOG.md](ISSUES_LOG.md) for a history of common issues and solutions.

---

## 📜 Project History & Timeline

### Phase 3: Data Engineering (Jan 2026)
- **Task 3.3**: Created final dataset pipeline (Cleaning -> Alignment -> Split).
- **Task 3.2**: Implemented multi-sensor time alignment.
- **Task 3.1**: Built data cleaning pipeline.
- **Verification**: Removed legacy notebooks with absolute paths.

### Phase 4: Feature Engineering (Jan 2026)
- **Task 4.1**: Implemented temporal feature engineering (cyclical time, day parts).

### Phase 2: Data Acquisition & EDA (Jan 2026)
- **Task 2.4**: Validated term lifecycle trends.
- **Task 2.3**: Analyzed participant quality and defined tiers.
- **Task 2.2**: Performed deep dive EDA on sensors.
- **Task 2.1**: Extracted sensing data and created manifest.
- **Decision**: Pivoted to **Sensing-Only** scope (passive data only).

### Phase 1: Setup (Jan 2026)
- Initialized project structure.
- Configured `uv` and `pyproject.toml`.
- Created initial documentation.
