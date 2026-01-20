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
    echo -e "\033[0;32mDataset found at data/raw/dataset/sensing.\033[0m"
elif [ -d "data/raw/sensing" ]; then
    echo -e "\033[0;32mDataset found at data/raw/sensing.\033[0m"
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

# 1. Data Cleaning (Task 3.1)
echo -e "\n\033[0;36m[1/10] Running Data Cleaning...\033[0m"
python -W ignore src/data/run_cleaning.py

# 2. Time Alignment (Task 3.2)
echo -e "\n\033[0;36m[2/10] Running Time Alignment...\033[0m"
python -W ignore src/data/run_alignment.py

# 3. Final Dataset Creation (Task 3.3)
echo -e "\n\033[0;36m[3/10] Creating Final Dataset...\033[0m"
python -W ignore src/data/create_final_dataset.py

# 4. Feature Engineering Verification (Task 4.0)
echo -e "\n\033[0;36m[4/10] Verifying Feature Engineering...\033[0m"
python -W ignore src/verify_phase4.py

# 5. Baseline Models (Tasks 5.1 & 5.3)
echo -e "\n\033[0;36m[5/10] Running Baseline Models (Regression & Classification)...\033[0m"
python -W ignore src/analysis/modeling/01_regression_baselines.py
python -W ignore src/analysis/modeling/03_classification_baselines.py

# 6. Advanced ML (Task 6.3 - Boosting)
echo -e "\n\033[0;36m[6/10] Running Gradient Boosting Comparison...\033[0m"
python -W ignore src/analysis/modeling/06_boosting_comparison.py

# 7. Deep Learning - LSTM (Task 6.4)
echo -e "\n\033[0;36m[7/10] Running LSTM Model...\033[0m"
python -W ignore src/analysis/modeling/07_lstm_timeseries.py

# 8. Deep Learning - Transformer (Task 7.2 - SOTA)
echo -e "\n\033[0;36m[8/10] Training Best Model (Transformer)...\033[0m"
python -W ignore src/analysis/modeling/09_transformer.py

# 9. Anomaly Detection (Task 7.1 - Autoencoder)
echo -e "\n\033[0;36m[9/10] Running Anomaly Detection (Autoencoder)...\033[0m"
python -W ignore src/analysis/modeling/08_autoencoder.py

echo -e "\n\033[0;32mPipeline Complete! All Models (Baselines -> Transformer) & Anomalies saved.\033[0m"
echo -e "Check MLflow UI at http://localhost:5000 for experiment tracking"
