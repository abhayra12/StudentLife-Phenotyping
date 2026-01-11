# 📖 StudentLife Phenotyping - Development Guide

**Purpose**: This guide tracks all development activities, decisions, code patterns, and lessons learned throughout the project. It serves as your personal ML handbook and reference for future projects.

**Last Updated**: 2026-01-11

---

## 📅 Project Timeline

### Task 1.1: Initialize Project Structure
**Status**: 🔄 In Progress  
**Started**: 2026-01-11  
**Completed**: _Pending_

**Objective**: Set up proper project structure for ML development

**Steps Taken**:
1. ✅ Initialized Git repository
   ```bash
   git init
   ```

2. ✅ Created comprehensive learning plan (`plan.md`)
   - 14 phases covering full ML lifecycle
   - Detailed tasks with learning objectives
   - Step-by-step guidance for each component

3. ✅ Created `.gitignore` 
   - Excluded `plan.md` (private learning guide)
   - Excluded `data/` directory (large dataset files)
   - Excluded `models/` directory (trained model artifacts)
   - Excluded environment and IDE files
   - Excluded MLflow artifacts

4. ✅ Installed `uv` (modern Python package manager)
   ```bash
   pip install uv
   ```
   
5. ✅ Created virtual environment with uv
   ```bash
   python -m uv venv  # Note: Used python -m uv due to Windows PATH issue
   ```

6. ✅ Fixed PowerShell execution policy
   ```bash
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

7. ✅ Activated virtual environment
   ```bash
   .venv\Scripts\activate
   ```

8. ✅ Created `requirements.txt` with all ML dependencies (legacy)

9. ✅ Installed all dependencies using pip (initial approach)

10. ✅ Migrated to modern uv workflow
    ```bash
    # Initialized uv project
    python -m uv init --no-readme --no-pin-python
    
    # Added all dependencies with uv
    python -m uv add pandas numpy scipy matplotlib seaborn plotly scikit-learn \
      xgboost lightgbm catboost torch statsmodels tqdm joblib pyyaml \
      python-dotenv fastapi uvicorn pydantic optuna
    
    # Added development dependencies
    python -m uv add --dev jupyter pytest pytest-cov ipykernel
    ```
    **Result**: Created `pyproject.toml` and `uv.lock` for modern dependency management

11. ✅ Verified installation
    ```bash
    python -c "import pandas; import numpy; import sklearn; import torch; print('Success!')"
    ```
    **Verified Package Versions**:
    - pandas: 2.3.1
    - numpy: 2.3.2
    - torch (PyTorch): 2.8.0+cpu
    - scikit-learn: 1.7.2
    - All major ML packages imported successfully ✅

**Challenges Faced**:
- **Challenge 1**: `uv` command not recognized after installation
  - **Cause**: Windows doesn't automatically add user Python packages to PATH
  - **Solution**: Use `python -m uv` instead of just `uv`
  
- **Challenge 2**: PowerShell blocked virtual environment activation
  - **Cause**: PowerShell execution policy restricts script execution by default
  - **Solution**: Changed execution policy to RemoteSigned for current user

### Challenge 3: uv Configuration for Virtual Environment Package Management
**Date**: 2026-01-11  
**Problem**: Initially, `python -m uv pip install` failed with "No module named uv" after activating venv. Later,when using global uv, it targeted the global Python environment instead of the venv  
**Root Cause**: `uv` installed globally (outside venv) isn't automatically available/doesn't target the venv Python when environment is activated  
**Solution**: Install `uv` inside the activated virtual environment itself:
```bash
# Inside activated venv:
pip install uv --force-reinstall

# Then uv commands target the venv Python:
python -m uv --version  # Works!
python -m uv pip list   # Shows venv packages
python -m uv pip check  # Validates venv packages
```  
**Verified Working**: `python -m uv pip check` confirms "All installed packages are compatible" ✅  
**Lesson Learned**: For uv to work with a venv on Windows:
1. Use `python -m uv venv` to **create** the venv (fast!)
2. Activate the venv: `.venv\Scripts\activate`  
3. Install uv **inside** the venv: `pip install uv`
4. Now use `python -m uv pip install` for fast package installation in that venv
5. Always use `python -m uv` (not just `uv`) due to Windows PATH issues

**Next Steps**:
- [ ] Set up directory structure
- [ ] Create requirements.txt
- [ ] Install initial dependencies
- [ ] Create README.md
- [ ] Verify setup and commit to Git

**Key Files Created**:
- `plan.md` - Comprehensive 14-phase learning plan
- `.gitignore` - Git ignore rules
- `development_guide.md` - This file
- `pyproject.toml` - Modern Python project configuration
- `uv.lock` - Dependency lock file

---

## ✅ Completed Tasks

Tasks are added here when completed. Each entry includes date, what was done, challenges faced, and lessons learned.

### Task 1.1: Initialize Project Structure (In Progress)
**Started**: 2026-01-11  
**Status**: 🔄 In Progress

**Completed So Far**:

## 🏗️ Project Structure

_(To be updated as directories are created)_

---

## 💡 Code Patterns & Snippets

### Data Loading Template
_(To be added)_

### Feature Engineering Pattern
_(To be added)_

### Model Training Workflow
_(To be added)_

---

## 🎯 Key Decisions Log

### Decision 1: Project Organization
**Date**: 2026-01-11  
**Context**: How to organize ML project for maintainability  
**Decision**: Use industry-standard structure with separate directories for:
- `data/` (raw, processed, features)
- `notebooks/` (exploration, preprocessing, modeling, evaluation)
- `src/` (reusable modules)
- `models/` (saved models)
- `api/` (FastAPI deployment)
- `tests/` (testing)

**Rationale**: This structure is widely used in production ML projects and scales well  
**Alternatives Considered**: Flat structure (rejected - doesn't scale)

### Decision 2: Version Control for Learning Plan
**Date**: 2026-01-11  
**Context**: Should learning plan be tracked in Git?  
**Decision**: Keep `plan.md` in `.gitignore`  
**Rationale**: Learning plan is personal/instructional and not part of production code  
**Alternative**: Track it (rejected - adds noise to repository)

---

## 🛠️ Technical Setup

### Environment
- **OS**: Windows (PowerShell)
- **Python Version**: 3.13.9
- **Virtual Environment**: uv (via `python -m uv venv`)
- **Package Manager**: uv 0.9.24 (10-100x faster than pip)

### Key Libraries
**Data Science Stack**:
- pandas: 2.3.1
- numpy: 2.3.2
- scipy: 1.15.3

**Machine Learning**:
- scikit-learn: 1.7.2
- xgboost: 3.1.3
- lightgbm: 4.6.0
- catboost: 1.2.8

**Deep Learning**:
- torch (PyTorch): 2.8.0+cpu

**Visualization**:
- matplotlib: 3.10.0
- seaborn: 0.13.2
- plotly: 6.5.1

**Time Series**:
- statsmodels: 0.14.6

**API Development**:
- fastapi: 0.128.0
- uvicorn: 0.40.0
- pydantic: 2.10.7

**Development Tools**:
- jupyter: 1.1.1
- pytest: 9.0.2
- optuna: 4.6.0
- tqdm: 4.67.1

---

## 📚 Learning Resources

### Papers to Read
1. Wang, R., et al. (2014). "StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones"
2. _(More to be added as project progresses)_

### Tutorials Referenced
1. [ML Zoomcamp - FastAPI + UV Workshop](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/05-deployment/workshop) - Modern ML deployment with uv
2. _(More to be added)_

### Documentation Links
1. [StudentLife Dataset](https://studentlife.cs.dartmouth.edu/dataset.html)
2. _(More to be added)_

---

## 🐛 Challenges & Solutions

### Challenge 1: uv Command Not Found on Windows
**Date**: 2026-01-11  
**Problem**: After installing `uv` with `pip install uv`, the `uv` command was not recognized in PowerShell  
**Root Cause**: Windows doesn't automatically add user Python packages to the system PATH  
**Solution**: Use `python -m uv` instead of just `uv` for all commands
- `python -m uv venv` instead of `uv venv`
- `python -m uv pip install` instead of `uv pip install`  
**Lesson Learned**: On Windows, user-installed Python packages may not be in PATH. Using `python -m <module>` is a reliable workaround

### Challenge 2: PowerShell Script Execution Blocked
**Date**: 2026-01-11  
**Problem**: Virtual environment activation failed with "running scripts is disabled on this system" error  
**Root Cause**: PowerShell's default execution policy (Restricted) prevents running any scripts  
**Solution**: Changed execution policy to RemoteSigned for current user:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```  
**Lesson Learned**: Windows PowerShell requires explicit permission to run scripts for security. RemoteSigned allows local scripts while requiring downloaded scripts to be signed

---

## ✨ Best Practices Learned

### Git Workflow
1. Always create meaningful commit messages
2. Commit frequently with atomic changes
3. Use branches for experimental features

### ML Development
_(To be added as we progress)_

### Python Coding
_(To be added as we progress)_

---

## 📊 Model Performance Tracker

_(To be added during modeling phase)_

| Model | Task | Metric | Score | Date | Notes |
|-------|------|--------|-------|------|-------|
| Baseline | - | - | - | - | - |

---

## 🎓 Lessons Learned Summary

### Week 1: Project Setup
- Proper project structure is critical from day one
- Good `.gitignore` prevents large files in repository
- Clear learning plan helps maintain focus

_(More to be added weekly)_

---

## 🔄 Iterative Improvements

### Iteration 1: Initial Setup
**Date**: 2026-01-11  
**What Changed**: Created project foundation  
**Why**: Starting from scratch  
**Result**: Clean project structure ready for development

---

## 📝 Notes & Tips

### Git Tips
- Use `git status` frequently to check what's staged
- Review changes before committing with `git diff`

### Project Setup Workflow (Step-by-Step)
**Complete setup process for reproducing this environment**:

1. **Prerequisites**:
   - Python 3.8+ installed (we used 3.13.9)
   - Git installed
   - PowerShell (Windows) or Terminal (Mac/Linux)

2. **Initial Setup**:
   ```bash
   # Navigate to project directory
   cd path/to/StudntLife-Pheno
   
   # Initialize Git repository
   git init
   
   # Create .gitignore (exclude data/, models/, .venv/, plan.md)
   ```

3. **Virtual Environment Creation**:
   ```bash
   # Install uv (fast package manager)
   pip install uv
   
   # Create virtual environment using uv
   python -m uv venv
   
   # Windows: Fix PowerShell execution policy if needed
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   
   # Activate virtual environment
   # Windows PowerShell:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   # Inside activated virtual environment
   pip install -r requirements.txt
   
   # Then configure uv for faster future installs:
   pip install uv
   
   # Verify uv works with venv:
   python -m uv pip check
   
   # Now you can use uv for package management:
   # python -m uv pip install <package>
   # python -m uv pip list
   ```

5. **Verify Installation**:
   ```bash
   # Test imports
   python -c "import pandas; import numpy; import sklearn; import torch; print('✅ Success!')"
   
   # Check package versions
   pip list
   ```

6. **Initial Git Commit**:
   ```bash
   git add .
   git commit -m "feat: Initialize project structure and dependencies"
   ```

**Total Setup Time**: ~15-20 minutes (mostly dependency download time)

### Python Tips
- Always activate virtual environment before installing packages
- Use `pip list` to see installed packages
- Use `python --version` to verify Python version
- Use `which python` (Mac/Linux) or `where python` (Windows) to verify you're using venv Python

### Modern uv Workflow (For Future Projects)
**Reference**: [ML Zoomcamp FastAPI+UV Workshop](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/05-deployment/workshop)

Modern Python projects are moving from `requirements.txt` to `pyproject.toml` + `uv`:

**Traditional Approach** (what we're using now):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Modern uv Approach** (recommended for future):
```bash
# Initialize project with pyproject.toml
uv init
rm main.py  # Remove default file

# Add dependencies (auto-updates pyproject.toml + creates uv.lock)
uv add pandas numpy scikit-learn torch
uv add fast api uvicorn pydantic
uv add --dev pytest jupyter  # Dev dependencies

# Run commands without manual activation
uv run python train.py
uv run uvicorn api.main:app

# Others can reproduce exact environment
uv sync  # Installs from uv.lock
```

**Why use modern approach?**:
- ✅ `pyproject.toml`: Standard Python packaging (PEP 621)
- ✅ `uv.lock`: Guarantees reproducible installs
- ✅ `uv add`: Auto-manages dependencies
- ✅ `uv run`: No need to activate venv manually
- ✅ Faster package installation (10-100x)

**When to use each**:
- **requirements.txt**: Learning projects, simple scripts, broad compatibility
- **pyproject.toml + uv**: Production projects, team collaboration, modern workflows

**Our approach**: Using requirements.txt now (simpler for learning), will adopt modern uv workflow in Phase 10 (API Deployment).

### ML Tips
- **Virtual environments are essential**: Never install ML packages globally
- **PyTorch CPU vs GPU**: We installed CPU version (2.8.0+cpu). For GPU support, need CUDA-compatible version
- **Package compatibility**: Python 3.13 is very new; some packages may have limited support
- **Large packages**: torch, catboost, xgboost are 100+ MB; expect longer download times
- **Using uv in venv**: After initial pip install, configure uv (`pip install uv`) for 10-100x faster subsequent package installations. Always use `python -m uv` on Windows.

---

## 🎯 Current Focus

**Current Task**: Task 1.1 - Initialize Project Structure  
**Current Subtask**: Finalize modern uv workflow migration and create README.md  
**Progress**:
- ✅ Virtual environment created with uv
- ✅ Migrated to modern uv workflow
  - ✅ Created `pyproject.toml` (replaces requirements.txt)
  - ✅ Running `uv add` for all dependencies (in progress)
  - ✅ Creating `uv.lock` for reproducibility
- ✅ Updated all documentation (CURRENT_TASK, development_guide, SETUP_GUIDE)
- 🔄 Waiting for torch download to complete (~15MB/105MB)
- 🔄 Next: Create project directory structure
- 🔄 Next: Create README.md
- 🔄 Next: First Git commit

**Blockers**: None (torch downloading in background)  
**Questions**: None

**Key Achievement**: Successfully migrated to modern Python packaging standards with `pyproject.toml` + `uv.lock`!

---

## 🎉 Completed: Migration to Modern UV Workflow

**Date**: 2026-01-11  
**Status**: ✅ Complete

### What Changed
Migrated from traditional `requirements.txt` to modern `pyproject.toml` + `uv.lock` workflow.

### Files Created
- **`pyproject.toml`** (598 bytes): Project config with 26 production dependencies + 4 dev dependencies
- **`uv.lock`** (219KB): Complete dependency lock file with exact versions for reproducibility

### Dependencies Added
**Production** (26 packages): pandas, numpy, scipy, matplotlib, seaborn, plotly, scikit-learn, xgboost, lightgbm, catboost, torch, statsmodels, tqdm, joblib, pyyaml, python-dotenv, fastapi, uvicorn, pydantic, optuna

**Development** (4 packages): jupyter, pytest, pytest-cov, ipykernel

### Commands Used
```bash
python -m uv init --no-readme --no-pin-python
python -m uv add pandas numpy scipy matplotlib seaborn plotly scikit-learn xgboost lightgbm catboost torch statsmodels tqdm joblib pyyaml python-dotenv fastapi uvicorn pydantic optuna
python -m uv add --dev jupyter pytest pytest-cov ipykernel
```

### Key Benefits
- **Reproducibility**: `uv.lock` ensures exact versions across all environments
- **Speed**: 10-100x faster than pip
- **Standards**: Following PEP 621 (modern Python packaging)
- **Dev Dependencies**: Separated from production dependencies
- **Easy Management**: `uv add package` auto-updates everything

### New Commands Available
```bash
python -m uv pip list          # List packages
python -m uv tree              # Show dependency tree
python -m uv add <package>     # Add production dependency
python -m uv add --dev <pkg>   # Add dev dependency
python -m uv run <command>     # Run without activating venv
python -m uv sync              # Sync from uv.lock (for team)
```

---

## 📋 Current Task: Complete Task 1.1

**Next Steps**:
1. ⏳ Wait for torch download to complete (background)
2. 🔄 Create project directory structure
3. 🔄 Create README.md
4. 🔄 First Git commit

**When complete**: This task summary will be added to the "Completed Tasks" section above.

---

**Note**: This guide is a living document and will be updated continuously throughout the project. Each completed task gets documented here.
