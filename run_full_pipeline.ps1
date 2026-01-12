# Run Full Data Pipeline
# This script executes the complete data engineering pipeline from Raw Data to Final Dataset.

$ErrorActionPreference = "Stop"

Write-Host "Starting Full Data Pipeline..." -ForegroundColor Green

# 1. Data Cleaning (Task 3.1)
Write-Host "`n[1/3] Running Data Cleaning..." -ForegroundColor Cyan
python src/data/run_cleaning.py
if ($LASTEXITCODE -ne 0) { Write-Error "Cleaning failed!" }

# 2. Time Alignment (Task 3.2)
Write-Host "`n[2/3] Running Time Alignment..." -ForegroundColor Cyan
python src/data/run_alignment.py
if ($LASTEXITCODE -ne 0) { Write-Error "Alignment failed!" }

# 3. Final Dataset Creation (Task 3.3)
Write-Host "`n[3/3] Creating Final Dataset..." -ForegroundColor Cyan
python src/data/create_final_dataset.py
if ($LASTEXITCODE -ne 0) { Write-Error "Dataset creation failed!" }

Write-Host "`nPipeline Complete! Output saved to data/processed/" -ForegroundColor Green
