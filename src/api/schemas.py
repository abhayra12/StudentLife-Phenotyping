from pydantic import BaseModel, Field
from typing import List, Optional

class StudentBehavior(BaseModel):
    """Input schema for prediction."""
    # We need inputs corresponding to the features the model expects
    # For simplicity, we can take raw aggregated features
    # Or strict feature vector. Let's use feature vector for now as it's cleaner.
    
    # Example features (subset of what we use):
    # 'hour_sin', 'hour_cos', 'active_ratio', 'sleep_duration_24h', etc.
    # In production, we might want a raw list of floats if feature names are dynamic
    features: List[float] = Field(..., description="List of normalized feature values (order must match training)")
    
    # Optional metadata
    participant_id: Optional[str] = "unknown"

class PredictionResponse(BaseModel):
    participant_id: str
    predicted_activity_minutes: float
    confidence: Optional[float] = None

class AnomalyInput(BaseModel):
    features: List[float]

class AnomalyResponse(BaseModel):
    is_anomaly: bool
    reconstruction_error: float
    threshold: float
