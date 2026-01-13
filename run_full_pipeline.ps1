# Run Full Data Pipeline
# This script executes the complete lifecycle: Data Prep -> Verification -> Modeling (SOTA)

$ErrorActionPreference = "Stop"

Write-Host "Starting Full Data Pipeline..." -ForegroundColor Green

# 1. Data Cleaning (Task 3.1)
Write-Host "`n[1/10] Running Data Cleaning..." -ForegroundColor Cyan
python -W ignore src/data/run_cleaning.py
if ($LASTEXITCODE -ne 0) { Write-Error "Cleaning failed!"; exit 1 }

# 2. Time Alignment (Task 3.2)
Write-Host "`n[2/10] Running Time Alignment..." -ForegroundColor Cyan
python -W ignore src/data/run_alignment.py
if ($LASTEXITCODE -ne 0) { Write-Error "Alignment failed!"; exit 1 }

# 3. Final Dataset Creation (Task 3.3)
Write-Host "`n[3/10] Creating Final Dataset..." -ForegroundColor Cyan
python -W ignore src/data/create_final_dataset.py
if ($LASTEXITCODE -ne 0) { Write-Error "Dataset creation failed!"; exit 1 }

# 4. Feature Engineering Verification (Task 4.0)
Write-Host "`n[4/10] Verifying Feature Engineering..." -ForegroundColor Cyan
python -W ignore src/verify_phase4.py
if ($LASTEXITCODE -ne 0) { Write-Error "Feature Verification failed!"; exit 1 }

# 5. Baseline Models (Tasks 5.1 & 5.3)
Write-Host "`n[5/10] Running Baseline Models (Regression & Classification)..." -ForegroundColor Cyan
python -W ignore src/analysis/modeling/01_regression_baselines.py
if ($LASTEXITCODE -ne 0) { Write-Error "Regression Baselines failed!"; exit 1 }

python -W ignore src/analysis/modeling/03_classification_baselines.py
if ($LASTEXITCODE -ne 0) { Write-Error "Classification Baselines failed!"; exit 1 }

# 6. Advanced ML (Task 6.3 - Boosting)
Write-Host "`n[6/10] Running Gradient Boosting Comparison..." -ForegroundColor Cyan
python -W ignore src/analysis/modeling/06_boosting_comparison.py
if ($LASTEXITCODE -ne 0) { Write-Error "Boosting Comparison failed!"; exit 1 }

# 7. Deep Learning - LSTM (Task 6.4)
Write-Host "`n[7/10] Running LSTM Model..." -ForegroundColor Cyan
python -W ignore src/analysis/modeling/07_lstm_timeseries.py
if ($LASTEXITCODE -ne 0) { Write-Error "LSTM Training failed!"; exit 1 }

# 8. Deep Learning - Transformer (Task 7.2 - SOTA)
Write-Host "`n[8/10] Training Best Model (Transformer)..." -ForegroundColor Cyan
python -W ignore src/analysis/modeling/09_transformer.py
if ($LASTEXITCODE -ne 0) { Write-Error "Model Training failed!"; exit 1 }

# 9. Anomaly Detection (Task 7.1 - Autoencoder)
Write-Host "`n[9/10] Running Anomaly Detection (Autoencoder)..." -ForegroundColor Cyan
python -W ignore src/analysis/modeling/08_autoencoder.py
if ($LASTEXITCODE -ne 0) { Write-Error "Autoencoder failed!"; exit 1 }

Write-Host "`nPipeline Complete! All Models (Baselines -> Transformer) & Anomalies saved." -ForegroundColor Green
