"""
StudentLife Behavioral Prediction API

FastAPI service for predicting student activity levels and detecting behavioral anomalies.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys

# Add project root
sys.path.append('.')

from src.api.schemas import (
    StudentBehavior, PredictionResponse, 
    AnomalyInput, AnomalyResponse,
    HealthResponse, FeatureInfo, ExamplePayloads,
    FEATURES_PER_HOUR, SEQUENCE_LENGTH, EXPECTED_FEATURES, FEATURE_NAMES,
    EXAMPLE_264_FEATURES, EXAMPLE_11_FEATURES
)

# Dynamic module loading for model classes (handles numbered filenames)
import importlib.util

def load_module(path, name):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Storage for loaded models and config
models = {}
thresholds = {}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_activity_interpretation(minutes: float) -> str:
    """Convert predicted activity minutes to human-readable interpretation."""
    if minutes < 20:
        return "Very low activity - potential concern"
    elif minutes < 35:
        return "Below average activity"
    elif minutes < 55:
        return "Normal activity level"
    elif minutes < 70:
        return "Above average activity"
    else:
        return "Very high activity"


def get_anomaly_interpretation(error: float, threshold: float) -> tuple[str, str]:
    """Convert anomaly score to interpretation and recommendation."""
    if error < 0.5:
        return "Normal behavior pattern", None
    elif error < threshold:
        return "Mild behavioral variation (within normal range)", None
    elif error < 1.5:
        return "Moderate behavioral deviation detected", "Monitor for 2-3 days"
    elif error < 2.5:
        return "Significant behavioral deviation detected", "Consider wellness check-in"
    else:
        return "Major behavioral anomaly detected", "Recommend immediate attention"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and configuration on startup."""
    print("🚀 Starting StudentLife Behavior API...")
    print(f"   Device: {DEVICE}")
    
    # Load anomaly thresholds from config
    threshold_path = Path('models/anomaly_thresholds.json')
    if threshold_path.exists():
        with open(threshold_path) as f:
            thresholds.update(json.load(f))
        print(f"   ✓ Loaded thresholds: weekday={thresholds.get('weekday_threshold', 'N/A'):.3f}, "
              f"weekend={thresholds.get('weekend_threshold', 'N/A'):.3f}")
    else:
        # Fallback defaults
        thresholds['weekday_threshold'] = 0.909
        thresholds['weekend_threshold'] = 0.859
        print("   ⚠ Using default thresholds (no config file found)")
    
    # Load Transformer model
    try:
        transformer_mod = load_module('src/analysis/modeling/09_transformer.py', 'transformer_mod')
        BehaviorTransformer = transformer_mod.BehaviorTransformer
        
        t_model = BehaviorTransformer(input_dim=FEATURES_PER_HOUR, d_model=32, nhead=4, num_layers=2)
        t_model.load_state_dict(torch.load('models/transformer_best.pth', map_location=DEVICE))
        t_model.to(DEVICE)
        t_model.eval()
        models['transformer'] = t_model
        print("   ✓ Transformer model loaded")
    except Exception as e:
        print(f"   ✗ Failed to load Transformer: {e}")

    # Load Autoencoder model
    try:
        autoencoder_mod = load_module('src/analysis/modeling/08_autoencoder.py', 'autoencoder_mod')
        Autoencoder = autoencoder_mod.Autoencoder
        
        ae_model = Autoencoder(input_dim=FEATURES_PER_HOUR, latent_dim=3)
        ae_model.load_state_dict(torch.load('models/autoencoder.pth', map_location=DEVICE))
        ae_model.to(DEVICE)
        ae_model.eval()
        models['autoencoder'] = ae_model
        print("   ✓ Autoencoder model loaded")
    except Exception as e:
        print(f"   ✗ Failed to load Autoencoder: {e}")
    
    print(f"\n📡 API Ready! Models loaded: {list(models.keys())}")
    print("   Docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health\n")
    
    yield
    
    print("Shutting down API...")


# Create FastAPI app with metadata for documentation
app = FastAPI(
    title="StudentLife Behavior Prediction API",
    description="""
## 🎯 Overview

Predict student physical activity levels and detect behavioral anomalies using deep learning models trained on smartphone sensor data.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Get Example Payloads
```bash
curl http://localhost:8000/examples
```
This returns ready-to-use JSON payloads you can copy into the Swagger UI!

### Step 2: Test Prediction
```bash
# Copy the predict_request from /examples, or use the "Try it out" button below
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @payload.json
```

### Step 3: Test Anomaly Detection
```bash
curl -X POST http://localhost:8000/anomaly -H "Content-Type: application/json" -d '{"features":[0.5,0.866,0.782,0.623,0.55,0.35,0.25,0.28,0.45,0.6,0.5]}'
```

---

## 📊 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API status and loaded models |
| `/features` | GET | Get feature names and schema |
| `/examples` | GET | **Get copy-paste ready payloads!** |
| `/predict` | POST | Predict next-day activity (264 features) |
| `/anomaly` | POST | Detect behavioral anomalies (11 features) |

---

## 📝 Feature Schema

Both endpoints expect normalized features (0-1 range):

| # | Feature | Description | Example |
|---|---------|-------------|---------|
| 1 | `hour_sin` | sin(2π × hour/24) | 0.0 at midnight |
| 2 | `hour_cos` | cos(2π × hour/24) | 1.0 at midnight |
| 3 | `day_of_week_sin` | sin(2π × day/7) | Weekly cycle |
| 4 | `day_of_week_cos` | cos(2π × day/7) | Weekly cycle |
| 5 | `activity_stationary_pct` | % time stationary | 0.8 = mostly sitting |
| 6 | `activity_active_minutes` | Physical activity (normalized) | 0.5 = 30 min |
| 7 | `audio_voice_minutes` | Conversation time | 0.3 = 18 min |
| 8 | `audio_noise_minutes` | Ambient noise level | 0.2 = quiet |
| 9 | `location_entropy` | Mobility diversity | 0.7 = varied locations |
| 10 | `sleep_duration_rolling` | Rolling avg sleep | 0.6 = 6 hours |
| 11 | `week_of_term` | Academic week (1-10) | 0.5 = mid-term |

---

## 💡 Tips for Swagger UI

1. Click on an endpoint to expand it
2. Click **"Try it out"** button
3. The example payload is pre-filled - just click **"Execute"**
4. Or visit `/examples` first to get different test scenarios

    """,
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Quick Start", "description": "Get example payloads and feature info"},
        {"name": "Prediction", "description": "Predict activity levels"},
        {"name": "Anomaly Detection", "description": "Detect behavioral anomalies"},
        {"name": "Status", "description": "Health checks and monitoring"},
    ]
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health_check():
    """Check API health status and list loaded models."""
    return HealthResponse(
        status="ok" if models else "degraded",
        models_loaded=list(models.keys()),
        version="1.0.0"
    )


@app.get("/features", response_model=FeatureInfo, tags=["Quick Start"])
async def get_feature_info():
    """Get information about expected input features.
    
    Use this endpoint to understand the required input format for /predict and /anomaly.
    """
    return FeatureInfo()


@app.get("/examples", response_model=ExamplePayloads, tags=["Quick Start"])
async def get_example_payloads():
    """Get ready-to-use example payloads for testing.
    
    **This is the easiest way to test the API!**
    
    Returns complete, valid JSON payloads that you can:
    1. Copy directly into the Swagger UI "Try it out" section
    2. Save to a file and use with curl: `curl -X POST /predict -d @payload.json`
    3. Use as templates for your own data
    
    **Scenarios included:**
    - `predict_request`: Normal weekday behavior pattern (24 hours)
    - `anomaly_request_normal`: Healthy daily summary
    - `anomaly_request_concerning`: Low activity pattern (may flag as anomaly)
    """
    return ExamplePayloads(
        predict_request={
            "participant_id": "demo_user",
            "features": EXAMPLE_264_FEATURES
        },
        anomaly_request_normal={
            "features": EXAMPLE_11_FEATURES,
            "is_weekend": False
        },
        anomaly_request_concerning={
            "features": [
                0.5, 0.866,           # hour encoding (noon)
                0.782, 0.623,         # day encoding (Tuesday)
                0.95,                 # 95% stationary (very sedentary!)
                0.05,                 # only 3 min activity
                0.02,                 # almost no conversation (isolated)
                0.1,                  # quiet environment
                0.1,                  # very low location variety (stayed in room)
                0.4,                  # poor sleep
                0.8                   # late in term (finals stress)
            ],
            "is_weekend": False
        },
        feature_guide={
            "hour_sin": "sin(2π × hour/24) - Circadian position (0 at 6am/6pm, 1 at midnight)",
            "hour_cos": "cos(2π × hour/24) - Circadian position (1 at midnight/noon, 0 at 6am/6pm)",
            "day_of_week_sin": "sin(2π × day/7) - Weekly cycle encoding",
            "day_of_week_cos": "cos(2π × day/7) - Weekly cycle encoding",
            "activity_stationary_pct": "Percentage of time stationary (0=always moving, 1=always still)",
            "activity_active_minutes": "Physical activity normalized to 0-1 (0.5 ≈ 30 minutes)",
            "audio_voice_minutes": "Conversation time normalized (0.5 ≈ 30 minutes)",
            "audio_noise_minutes": "Ambient noise level normalized",
            "location_entropy": "Mobility diversity (0=one location, 1=many varied locations)",
            "sleep_duration_rolling": "Rolling average sleep hours normalized (0.75 ≈ 7.5 hours)",
            "week_of_term": "Academic week 1-10 normalized (0.1=week 1, 1.0=week 10)"
        }
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_activity(data: StudentBehavior):
    """Predict next-day physical activity minutes.
    
    **Input:** 264 features representing 24 hours of behavioral data (24 × 11 features per hour).
    
    **Output:** Predicted activity minutes with human-readable interpretation.
    
    **Feature Order (per hour):**
    1. hour_sin, hour_cos (circadian encoding)
    2. day_of_week_sin, day_of_week_cos (weekly cycle)
    3. activity_stationary_pct (% time stationary)
    4. activity_active_minutes (normalized)
    5. audio_voice_minutes (conversation time)
    6. audio_noise_minutes (ambient noise)
    7. location_entropy (mobility diversity)
    8. sleep_duration_rolling (sleep hours)
    9. week_of_term (academic week)
    """
    if 'transformer' not in models:
        raise HTTPException(
            status_code=503, 
            detail="Transformer model not loaded. Check server logs."
        )
    
    try:
        # Reshape: [264] -> [24, 11] -> [24, 1, 11] for model input
        feats = np.array(data.features).reshape(SEQUENCE_LENGTH, FEATURES_PER_HOUR)
        tensor_in = torch.FloatTensor(feats).unsqueeze(1).to(DEVICE)  # [24, 1, 11]
        
        with torch.no_grad():
            pred = models['transformer'](tensor_in)
            # Handle both scalar and tensor outputs
            if pred.dim() > 0 and pred.numel() > 1:
                pred_value = pred.mean().item()
            else:
                pred_value = pred.item()
        
        interpretation = get_activity_interpretation(pred_value)
        
        return PredictionResponse(
            participant_id=data.participant_id,
            predicted_activity_minutes=round(pred_value, 2),
            interpretation=interpretation,
            confidence=None  # Could add uncertainty quantification here
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/anomaly", response_model=AnomalyResponse, tags=["Anomaly Detection"])
async def detect_anomaly(data: AnomalyInput):
    """Detect behavioral anomalies using autoencoder reconstruction error.
    
    **Input:** 11 features representing daily aggregated behavioral summary.
    
    **Output:** Anomaly flag, reconstruction error, and interpretation.
    
    **Thresholds:**
    - Weekday: 0.909 (95th percentile of training data)
    - Weekend: 0.859 (accounts for naturally different weekend behavior)
    
    Set `is_weekend: true` if checking weekend data for context-aware detection.
    """
    if 'autoencoder' not in models:
        raise HTTPException(
            status_code=503, 
            detail="Autoencoder model not loaded. Check server logs."
        )
    
    try:
        # Reshape for model: [11] -> [1, 11]
        feats = np.array(data.features).reshape(1, -1)
        tensor_in = torch.FloatTensor(feats).to(DEVICE)
        
        with torch.no_grad():
            recon, _ = models['autoencoder'](tensor_in)
            error = torch.mean((tensor_in - recon) ** 2).item()
        
        # Select threshold based on day type
        threshold = thresholds.get(
            'weekend_threshold' if data.is_weekend else 'weekday_threshold',
            0.909
        )
        
        is_anomaly = error > threshold
        interpretation, recommendation = get_anomaly_interpretation(error, threshold)
        
        return AnomalyResponse(
            is_anomaly=is_anomaly,
            reconstruction_error=round(error, 4),
            threshold=round(threshold, 4),
            interpretation=interpretation,
            recommendation=recommendation
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
