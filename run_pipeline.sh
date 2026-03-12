#!/bin/bash
# Run Full ML Pipeline - Docker Version
# This script executes the complete lifecycle: Data Prep -> Verification -> Modeling

set -e  # Exit on error

# Activate virtual environment (inside Docker only; on host we use system Python)
if [ -f "/app/.venv/bin/activate" ]; then
    source /app/.venv/bin/activate
fi

# Set timeout for large package downloads (CUDA libraries)
export UV_HTTP_TIMEOUT=120

# Check if data exists, offer to download if missing
echo "Checking for StudentLife dataset..."
if [ -d "data/raw/dataset/sensing" ]; then
    echo -e "\033[0;32mSensor data found at data/raw/dataset/sensing.\033[0m"
elif [ -d "data/raw/sensing" ]; then
    echo -e "\033[0;32mSensor data found at data/raw/sensing.\033[0m"
else
    echo -e "\033[0;33mDataset not found in data/raw/dataset/sensing or data/raw/sensing\033[0m"
    echo -e "\033[0;36mWould you like to download it now? (y/n)\033[0m"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "\033[0;32mDownloading StudentLife dataset...\033[0m"
        python3 src/data/download_dataset.py
        echo -e "\033[0;32mDataset downloaded successfully!\033[0m"
    else
        echo -e "\033[0;33mSkipping download. Pipeline will fail without data.\033[0m"
        echo -e "\033[0;33mTo download later, run: python3 src/data/download_dataset.py\033[0m"
    fi
fi

# Check for EMA data
if [ -d "data/raw/dataset/EMA" ]; then
    echo -e "\033[0;32mEMA data found at data/raw/dataset/EMA.\033[0m"
else
    echo -e "\033[0;33mEMA data not found. Attempting extraction from archive...\033[0m"
    if [ -f "data/raw/dataset.tar.bz2" ]; then
        tar xjf data/raw/dataset.tar.bz2 -C data/raw/ dataset/EMA/ dataset/survey/ 2>/dev/null || true
        echo -e "\033[0;32mEMA data extracted.\033[0m"
    else
        echo -e "\033[0;33m⚠ No EMA data and no archive found. EMA steps will be skipped.\033[0m"
    fi
fi

# Check if dependencies are installed, install if needed
echo "Checking dependencies..."
if ! python3 -c "import pandas" 2>/dev/null; then
    echo -e "\033[0;33mDependencies not found. Installing via 'uv sync'...\033[0m"
    echo -e "\033[0;33mNote: Large CUDA packages may take several minutes to download.\033[0m"
    uv sync
    echo -e "\033[0;32mDependencies installed successfully!\033[0m"
else
    echo -e "\033[0;32mDependencies already installed.\033[0m"
fi

echo "Checking participant tiers..."
if [ ! -f "data/processed/participant_tiers.csv" ]; then
    echo -e "\033[0;33mparticipant_tiers.csv not found. Regenerating...\033[0m"
    python3 src/data/regenerate_tiers.py
fi

echo -e "\n\033[0;32mStarting Full ML Pipeline...\033[0m"

# ═══════════════════════════════════════════════════════
#  Phase 1: Sensor Data Pipeline
# ═══════════════════════════════════════════════════════

# 1. Data Cleaning (Task 3.1)
echo -e "\n\033[0;36m[1/14] Running Data Cleaning...\033[0m"
python3 -W ignore src/data/run_cleaning.py

# 2. Time Alignment (Task 3.2)
echo -e "\n\033[0;36m[2/14] Running Time Alignment...\033[0m"
python3 -W ignore src/data/run_alignment.py

# 3. Final Dataset Creation (Task 3.3)
echo -e "\n\033[0;36m[3/14] Creating Final Dataset...\033[0m"
python3 -W ignore src/data/create_final_dataset.py

# 4. Feature Engineering Verification (Task 4.0)
echo -e "\n\033[0;36m[4/14] Verifying Feature Engineering...\033[0m"
python3 -W ignore src/verify_phase4.py

# ═══════════════════════════════════════════════════════
#  Phase 2: EMA Data Pipeline
# ═══════════════════════════════════════════════════════

# 5. Parse EMA data (Stress, Sleep, Mood self-reports)
echo -e "\n\033[0;36m[5/14] Parsing EMA Data (self-reports)...\033[0m"
python3 -W ignore src/data/ema_loader.py

# 6. Merge sensor features with EMA stress responses
echo -e "\n\033[0;36m[6/14] Merging Sensor ↔ EMA Data...\033[0m"
python3 -W ignore src/data/merge_sensor_ema.py

# ═══════════════════════════════════════════════════════
#  Phase 3: Activity Prediction Models (sensor-only)
# ═══════════════════════════════════════════════════════

# 7. Baseline Models (Tasks 5.1 & 5.3)
echo -e "\n\033[0;36m[7/14] Running Baseline Models (Regression & Classification)...\033[0m"
python3 -W ignore src/analysis/modeling/01_regression_baselines.py
python3 -W ignore src/analysis/modeling/03_classification_baselines.py

# 8. Advanced ML (Task 6.3 - Boosting)
echo -e "\n\033[0;36m[8/14] Running Gradient Boosting Comparison...\033[0m"
python3 -W ignore src/analysis/modeling/06_boosting_comparison.py

# 9. Deep Learning - LSTM (Task 6.4)
echo -e "\n\033[0;36m[9/14] Running LSTM Model...\033[0m"
python3 -W ignore src/analysis/modeling/07_lstm_timeseries.py

# 10. Deep Learning - Transformer (Task 7.2 - SOTA)
echo -e "\n\033[0;36m[10/14] Training Best Model (Transformer)...\033[0m"
python3 -W ignore src/analysis/modeling/09_transformer.py

# 11. Anomaly Detection (Task 7.1 - Autoencoder)
echo -e "\n\033[0;36m[11/14] Running Anomaly Detection (Autoencoder)...\033[0m"
python3 -W ignore src/analysis/modeling/08_autoencoder.py

# ═══════════════════════════════════════════════════════
#  Phase 4: Stress Prediction (sensor → EMA)
# ═══════════════════════════════════════════════════════

# 12. Stress prediction (10 supervised + unsupervised)
echo -e "\n\033[0;36m[12/14] Running Stress Prediction (10 algorithms)...\033[0m"
python3 -W ignore src/analysis/modeling/stress_prediction.py

# 13. SOTA Stress Prediction (CatBoost+Optuna HPO + Stacked Ensemble + SHAP)
echo -e "\n\033[0;36m[13/14] SOTA Stress Prediction (CatBoost HPO + Stacked Ensemble + SHAP)...\033[0m"
python3 -W ignore src/analysis/modeling/sota_stress_prediction.py

# 14. EMA Exploratory Analysis & Correlation
echo -e "\n\033[0;36m[14/14] Running EMA Analysis & Visualizations...\033[0m"
python3 -W ignore src/analysis/eda/ema_eda.py
python3 -W ignore src/analysis/eda/sensor_ema_correlation.py

echo -e "\n\033[0;32m╔══════════════════════════════════════════════════╗\033[0m"
echo -e "\033[0;32m║   Pipeline Complete — All 14 steps finished      ║\033[0m"
echo -e "\033[0;32m╚══════════════════════════════════════════════════╝\033[0m"
echo -e ""
echo -e "  Outputs:"
echo -e "  ├─ Sensor models:         models/*.pth"
echo -e "  ├─ Stress models (×10):   models/*_stress_*.pkl"
echo -e "  ├─ SOTA models:           models/catboost_optuna_stress.pkl"
  echo -e "  ├─ Soft voting ensemble:  models/soft_voting_ensemble_stress.pkl"
echo -e "  ├─ SHAP plots:            reports/figures/modeling/sota_shap_importance.png"
echo -e "  ├─ Results:               reports/results/"
echo -e "  ├─ Figures:               reports/figures/"
echo -e "  └─ MLflow UI:             http://localhost:5000"
