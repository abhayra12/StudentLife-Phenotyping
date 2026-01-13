# Run Full Data Pipeline
# This script executes the complete lifecycle: Data Prep -> Verification -> Modeling (SOTA)

$ErrorActionPreference = "Stop"

Write-Host "Starting Full Data Pipeline..." -ForegroundColor Green

# 1. Data Cleaning (Task 3.1)
Write-Host "`n[1/6] Running Data Cleaning..." -ForegroundColor Cyan
python src/data/run_cleaning.py
if ($LASTEXITCODE -ne 0) { Write-Error "Cleaning failed!"; exit 1 }

# 2. Time Alignment (Task 3.2)
Write-Host "`n[2/6] Running Time Alignment..." -ForegroundColor Cyan
python src/data/run_alignment.py
if ($LASTEXITCODE -ne 0) { Write-Error "Alignment failed!"; exit 1 }

# 3. Final Dataset Creation (Task 3.3)
Write-Host "`n[3/6] Creating Final Dataset..." -ForegroundColor Cyan
python src/data/create_final_dataset.py
if ($LASTEXITCODE -ne 0) { Write-Error "Dataset creation failed!"; exit 1 }

# 4. Feature Engineering Verification (Task 4.0)
Write-Host "`n[4/6] Verifying Feature Engineering..." -ForegroundColor Cyan
python src/verify_phase4.py
if ($LASTEXITCODE -ne 0) { Write-Error "Feature Verification failed!"; exit 1 }

# 5. Model Training (Task 7.2 - SOTA Transformer)
Write-Host "`n[5/6] Training Best Model (Transformer)..." -ForegroundColor Cyan
python src/analysis/modeling/09_transformer.py
if ($LASTEXITCODE -ne 0) { Write-Error "Model Training failed!"; exit 1 }

# 6. Anomaly Detection (Task 7.1 - Autoencoder)
Write-Host "`n[6/6] Running Anomaly Detection (Autoencoder)..." -ForegroundColor Cyan
python src/analysis/modeling/08_autoencoder.py
if ($LASTEXITCODE -ne 0) { Write-Error "Autoencoder failed!"; exit 1 }

Write-Host "`nPipeline Complete! SOTA Model & Anomalies saved." -ForegroundColor Green
