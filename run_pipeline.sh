#!/bin/bash
# Run Full ML Pipeline - Docker Version
# This script executes the complete lifecycle: Data Prep -> Verification -> Modeling

set -e  # Exit on error

# Activate virtual environment
source /app/.venv/bin/activate

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
        python src/data/download_dataset.py
        echo -e "\033[0;32mDataset downloaded successfully!\033[0m"
    else
        echo -e "\033[0;33mSkipping download. Pipeline will fail without data.\033[0m"
        echo -e "\033[0;33mTo download later, run: python src/data/download_dataset.py\033[0m"
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
if ! python -c "import pandas" 2>/dev/null; then
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
    python src/data/regenerate_tiers.py
fi

echo -e "\n\033[0;32mStarting Full ML Pipeline...\033[0m"

# ═══════════════════════════════════════════════════════
#  Phase 1: Sensor Data Pipeline
# ═══════════════════════════════════════════════════════

# 1. Data Cleaning (Task 3.1)
echo -e "\n\033[0;36m[1/13] Running Data Cleaning...\033[0m"
python -W ignore src/data/run_cleaning.py

# 2. Time Alignment (Task 3.2)
echo -e "\n\033[0;36m[2/13] Running Time Alignment...\033[0m"
python -W ignore src/data/run_alignment.py

# 3. Final Dataset Creation (Task 3.3)
echo -e "\n\033[0;36m[3/13] Creating Final Dataset...\033[0m"
python -W ignore src/data/create_final_dataset.py

# 4. Feature Engineering Verification (Task 4.0)
echo -e "\n\033[0;36m[4/13] Verifying Feature Engineering...\033[0m"
python -W ignore src/verify_phase4.py

# ═══════════════════════════════════════════════════════
#  Phase 2: EMA Data Pipeline
# ═══════════════════════════════════════════════════════

# 5. Parse EMA data (Stress, Sleep, Mood self-reports)
echo -e "\n\033[0;36m[5/13] Parsing EMA Data (self-reports)...\033[0m"
python -W ignore src/data/ema_loader.py

# 6. Merge sensor features with EMA stress responses
echo -e "\n\033[0;36m[6/13] Merging Sensor ↔ EMA Data...\033[0m"
python -W ignore src/data/merge_sensor_ema.py

# ═══════════════════════════════════════════════════════
#  Phase 3: Activity Prediction Models (sensor-only)
# ═══════════════════════════════════════════════════════

# 7. Baseline Models (Tasks 5.1 & 5.3)
echo -e "\n\033[0;36m[7/13] Running Baseline Models (Regression & Classification)...\033[0m"
python -W ignore src/analysis/modeling/01_regression_baselines.py
python -W ignore src/analysis/modeling/03_classification_baselines.py

# 8. Advanced ML (Task 6.3 - Boosting)
echo -e "\n\033[0;36m[8/13] Running Gradient Boosting Comparison...\033[0m"
python -W ignore src/analysis/modeling/06_boosting_comparison.py

# 9. Deep Learning - LSTM (Task 6.4)
echo -e "\n\033[0;36m[9/13] Running LSTM Model...\033[0m"
python -W ignore src/analysis/modeling/07_lstm_timeseries.py

# 10. Deep Learning - Transformer (Task 7.2 - SOTA)
echo -e "\n\033[0;36m[10/13] Training Best Model (Transformer)...\033[0m"
python -W ignore src/analysis/modeling/09_transformer.py

# 11. Anomaly Detection (Task 7.1 - Autoencoder)
echo -e "\n\033[0;36m[11/13] Running Anomaly Detection (Autoencoder)...\033[0m"
python -W ignore src/analysis/modeling/08_autoencoder.py

# ═══════════════════════════════════════════════════════
#  Phase 4: Stress Prediction (sensor → EMA)
# ═══════════════════════════════════════════════════════

# 12. Stress prediction (10 supervised + unsupervised)
echo -e "\n\033[0;36m[12/13] Running Stress Prediction (10 algorithms)...\033[0m"
python -W ignore src/analysis/modeling/stress_prediction.py

# 13. EMA Exploratory Analysis & Correlation
echo -e "\n\033[0;36m[13/13] Running EMA Analysis & Visualizations...\033[0m"
python -W ignore src/analysis/eda/ema_eda.py
python -W ignore src/analysis/eda/sensor_ema_correlation.py

echo -e "\n\033[0;32mPipeline Complete! All Models & Analyses saved.\033[0m"
echo -e "Results:"
echo -e "  - Sensor models:    models/*.pth"
echo -e "  - Stress models:    models/*_stress_*.pkl"
echo -e "  - Results:          reports/results/"
echo -e "  - Figures:          reports/figures/"
echo -e "  - Check MLflow UI at http://localhost:5000"
