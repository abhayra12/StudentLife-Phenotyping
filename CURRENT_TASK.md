# 📋 CURRENT TASK: Task 1.3 - Download and Organize Dataset

**Status**: 🔄 **READY TO START**  
**Assigned Date**: 2026-01-11  
**Phase**: PHASE 1 - PROJECT SETUP & ENVIRONMENT

---

## 🎯 Task Overview

**Objective**: Download the StudentLife dataset, organize it properly in the project structure, and create documentation for data organization.

**Why This Matters**: Understanding data organization is the first step in any ML project. Proper data management ensures reproducibility and makes it easier to work with the dataset.

**Estimated Time**: 1-2 hours (mostly download time)

---

## ✅ Prerequisites

Before starting this task, ensure:
- ✅ Task 1.1 complete (project structure created)
- ✅ Task 1.2 complete (development environment setup - already done in 1.1)
- ✅ Virtual environment activated
- ✅ Git repository initialized

**Note**: We already completed Task 1.2 as part of Task 1.1! All dependencies are installed via `pyproject.toml` + `uv.lock`.

---

## 📋 Task Steps

### Step 1: Download Student Life Dataset

**Dataset URL**: https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2

**Download Size**: ~53 GB compressed

**Options**:

**Option A: Direct Download (Recommended)**
```powershell
# Create download directory
cd data\raw

# Using PowerShell
Invoke-WebRequest -Uri "https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2" -OutFile "dataset.tar.bz2"

# Or using wget (if installed)
wget https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2
```

**Option B: Manual Download**
1. Open browser: https://studentlife.cs.dartmouth.edu/dataset.html
2. Download: `dataset.tar.bz2`
3. Move to: `data/raw/dataset.tar.bz2`

---

### Step 2: Extract Dataset

```powershell
# Navigate to raw data directory
cd data\raw

# Extract using 7-Zip (Windows)
7z x dataset.tar.bz2
7z x dataset.tar

# Or using tar (Windows 10+)
tar -xjf dataset.tar.bz2

# The extraction will create a 'dataset' directory
```

**Expected Directory Structure After Extraction**:
```
data/raw/dataset/
├── sensing/
│   ├── activity/
│   ├── audio/
│   ├── bluetooth/
│   ├── conversation/
│   ├── gps/
│   ├── phonecharge/
│   ├── phonelock/
│   └── wifi/
├── EMA/
│   ├── response/
│   └── notification/
├── survey/
│   ├── pre/
│   └── post/
└── education/
```

---

### Step 3: Document Dataset Structure

Create a data inventory file:

```powershell
# Create a Python script to analyze dataset
python -m uv run python
```

Then run this analysis:

```python
import os
import json
from pathlib import Path

def analyze_dataset(root_path):
    """Analyze dataset structure and create manifest."""
    manifest = {
        "dataset_name": "StudentLife",
        "total_size_gb": 0,
        "participants": [],
        "data_types": {}
    }
    
    root = Path(root_path)
    
    # Count files and sizes
    for dirpath, dirnames, filenames in os.walk(root):
        rel_path = Path(dirpath).relative_to(root)
        data_type = str(rel_path).split(os.sep)[0] if len(rel_path.parts) > 0 else "root"
        
        if data_type not in manifest["data_types"]:
            manifest["data_types"][data_type] = {
                "file_count": 0,
                "total_size_bytes": 0
            }
        
        for filename in filenames:
            filepath = Path(dirpath) / filename
            size = filepath.stat().st_size
            manifest["data_types"][data_type]["file_count"] += 1
            manifest["data_types"][data_type]["total_size_bytes"] += size
            manifest["total_size_gb"] += size / (1024**3)
    
    # Save manifest
    manifest_path = root.parent / "dataset_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Manifest created: {manifest_path}")
    print(f"📊 Total size: {manifest['total_size_gb']:.2f} GB")
    print(f"📁 Data types: {len(manifest['data_types'])}")
    
    return manifest

# Run analysis
manifest = analyze_dataset("data/raw/dataset")
```

Save this as `src/data/analyze_dataset.py` and run:
```bash
python -m uv run python src/data/analyze_dataset.py
```

---

### Step 4: Create Data Documentation

Create `data/README.md`:

```markdown
# StudentLife Dataset

## Overview
- **Source**: Dartmouth College
- **Citation**: Wang, R., et al. (2014)
- **Duration**: 10 weeks (Spring 2013)
- **Participants**: 48 students
- **Total Size**: ~53 GB

## Directory Structure

### Sensing Data (`sensing/`)
Passive smartphone sensor data collected continuously.

- **activity/**: Physical activity (stationary, walking, running)
- **audio/**: Audio features (volume, not raw audio for privacy)
- **bluetooth/**: Bluetooth device encounters
- **conversation/**: Conversation detection (yes/no, duration)
- **gps/**: Location data
- **phonecharge/**: Phone charging events
- **phonelock/**: Phone lock/unlock events
- **wifi/**: WiFi access points detected

### EMA Data (`EMA/`)
Ecological Momentary Assessment - self-reported states.

- **response/**: Student responses to EMA prompts
- **notification/**: EMA notification logs

### Survey Data (`survey/`)
Pre/post psychological assessments.

- **pre/**: Beginning of term surveys (PHQ-9, Big Five, etc.)
- **post/**: End of term surveys

### Education Data (`education/`)
Academic performance and deadlines.

## Data Format
- Most files are CSV format
- Timestamped with Unix epoch time
- Participant IDs: u00 through u59

## Privacy
- All data is anonymized
- Audio is aggregated features only (no raw audio)
- Location is GPS coordinates (no addresses)

## Usage
See `notebooks/01_exploration/` for data exploration examples.
```

---

### Step 5: Verify Dataset Integrity

Run verification checks:

```python
# Create src/data/verify_dataset.py
import os
from pathlib import Path

def verify_dataset(dataset_path):
    """Verify dataset integrity."""
    checks = {
        "sensing_dirs": ["activity", "audio", "bluetooth", "conversation", "gps", 
                        "phonecharge", "phonelock", "wifi"],
        "ema_dirs": ["response", "notification"],
        "survey_dirs": ["pre", "post"],
        "education_dirs": []
    }
    
    issues = []
    dataset = Path(dataset_path)
    
    # Check sensing directories
    sensing = dataset / "sensing"
    for dir_name in checks["sensing_dirs"]:
        dir_path = sensing / dir_name
        if not dir_path.exists():
            issues.append(f"Missing: {dir_path}")
        elif not list(dir_path.glob("*.csv")):
            issues.append(f"Empty: {dir_path}")
    
    # Check EMA directories
    ema = dataset / "EMA"
    for dir_name in checks["ema_dirs"]:
        dir_path = ema / dir_name
        if not dir_path.exists():
            issues.append(f"Missing: {dir_path}")
    
    if issues:
        print("⚠️ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Dataset verification passed!")
        print(f"✅ All expected directories present")
        return True

# Run verification
verify_dataset("data/raw/dataset")
```

---

### Step 6: Update .gitignore

Ensure large data files are not tracked:

```bash
# Verify .gitignore excludes data
cat .gitignore | grep "data/"

# Should already have:
# data/
# *.csv
# *.tar.bz2
```

---

## 📤 Submission Checklist

When completed:
- [ ] Dataset downloaded (`data/raw/dataset.tar.bz2`)
- [ ] Dataset extracted (`data/raw/dataset/`)
- [ ] Data manifest created (`data/raw/dataset_manifest.json`)
- [ ] Data README.md created
- [ ] Verification script run successfully
- [ ] No data files in git (check with `git status`)

---

## 💡 Hints & Tips

1. **Download taking forever?**: The dataset is 53GB. Use a stable connection or download overnight
2. **Extraction failed?**: Make sure you have enough disk space (~100GB free recommended)
3. **Missing 7-zip?**: Download from https://www.7-zip.org/
4. **Verification failed?**: Check if extraction completed fully
5. **Git showing data files?**: Verify `.gitignore` is working

---

## ❓ Common Questions

**Q: Do I commit the dataset?**  
A: **NO!** The dataset is 53GB. It's in `.gitignore` and stays local only.

**Q: What if download fails?**  
A: Resume using `wget -c` or download manager. Or download in chunks if possible.

**Q: Do I need all the data?**  
A: Yes, for a complete analysis. But you can start EDA with a subset (few participants).

---

**Note**: Dataset download may take several hours. You can continue with other learning while it downloads!

---

_Last Updated: 2026-01-11_
