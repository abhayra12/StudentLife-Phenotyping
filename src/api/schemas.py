"""
API Schemas for StudentLife Behavioral Prediction

This module defines Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

# Feature names in order (11 features per hour)
FEATURE_NAMES = [
    "hour_sin",              # sin(2π × hour/24) - Circadian encoding
    "hour_cos",              # cos(2π × hour/24) - Circadian encoding
    "day_of_week_sin",       # sin(2π × day/7) - Weekly cycle
    "day_of_week_cos",       # cos(2π × day/7) - Weekly cycle
    "activity_stationary_pct",  # % time stationary
    "activity_active_minutes",  # Minutes of physical activity (normalized)
    "audio_voice_minutes",      # Conversation time (normalized)
    "audio_noise_minutes",      # Ambient noise (normalized)
    "location_entropy",         # Mobility diversity index (normalized)
    "sleep_duration_rolling",   # Rolling average sleep hours (normalized)
    "week_of_term",             # Academic week 1-10 (normalized)
]

FEATURES_PER_HOUR = len(FEATURE_NAMES)  # 11
SEQUENCE_LENGTH = 24  # 24 hours
EXPECTED_FEATURES = FEATURES_PER_HOUR * SEQUENCE_LENGTH  # 264


class StudentBehavior(BaseModel):
    """Input schema for activity prediction.
    
    Requires 264 features: 24 hours × 11 features per hour.
    Features should be normalized to [0, 1] range for best results.
    """
    features: List[float] = Field(
        ...,
        description=f"List of {EXPECTED_FEATURES} normalized feature values (24h × 11 features). "
                    f"Features per hour: {', '.join(FEATURE_NAMES)}"
    )
    participant_id: Optional[str] = Field(
        default="unknown",
        description="Identifier for the student/participant"
    )
    
    @field_validator('features')
    @classmethod
    def validate_feature_length(cls, v):
        if len(v) != EXPECTED_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_FEATURES} features (24 hours × 11 features), "
                f"got {len(v)}. See API docs for feature schema."
            )
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "participant_id": "u42",
                    "features": [0.5] * EXPECTED_FEATURES,  # Placeholder
                    "description": "24 hours of behavioral data (264 values total)"
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for activity prediction."""
    participant_id: str = Field(description="Participant identifier")
    predicted_activity_minutes: float = Field(
        description="Predicted physical activity minutes for next 24 hours"
    )
    interpretation: str = Field(
        description="Human-readable interpretation of the prediction"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Prediction confidence (0-1), if available"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "participant_id": "u42",
                    "predicted_activity_minutes": 45.2,
                    "interpretation": "Normal activity level",
                    "confidence": None
                }
            ]
        }
    }


class AnomalyInput(BaseModel):
    """Input schema for anomaly detection.
    
    Requires 11 features: daily aggregated behavioral summary.
    """
    features: List[float] = Field(
        ...,
        description=f"List of {FEATURES_PER_HOUR} daily aggregated feature values. "
                    f"Features: {', '.join(FEATURE_NAMES)}"
    )
    is_weekend: Optional[bool] = Field(
        default=False,
        description="Whether this day is a weekend (affects threshold)"
    )
    
    @field_validator('features')
    @classmethod
    def validate_feature_length(cls, v):
        if len(v) != FEATURES_PER_HOUR:
            raise ValueError(
                f"Expected {FEATURES_PER_HOUR} features (daily aggregated), "
                f"got {len(v)}. See API docs for feature schema."
            )
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [0.5, 0.866, 0.433, 0.901, 0.75, 0.25, 0.15, 0.30, 0.45, 0.60, 0.30],
                    "is_weekend": False
                }
            ]
        }
    }


class AnomalyResponse(BaseModel):
    """Response schema for anomaly detection."""
    is_anomaly: bool = Field(
        description="Whether the behavior pattern is anomalous"
    )
    reconstruction_error: float = Field(
        description="Autoencoder reconstruction error (higher = more unusual)"
    )
    threshold: float = Field(
        description="Threshold used for anomaly detection"
    )
    interpretation: str = Field(
        description="Human-readable interpretation of the result"
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Suggested action based on the result"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "is_anomaly": True,
                    "reconstruction_error": 1.234,
                    "threshold": 0.909,
                    "interpretation": "Significant behavioral deviation detected",
                    "recommendation": "Consider wellness check-in"
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(description="Service status (ok/degraded)")
    models_loaded: List[str] = Field(description="List of loaded model names")
    version: str = Field(default="1.0.0", description="API version")


class FeatureInfo(BaseModel):
    """Information about expected features."""
    features_per_hour: int = FEATURES_PER_HOUR
    sequence_length: int = SEQUENCE_LENGTH
    total_features_for_prediction: int = EXPECTED_FEATURES
    feature_names: List[str] = FEATURE_NAMES
    description: str = (
        "For /predict: provide 264 features (24 hours × 11 features). "
        "For /anomaly: provide 11 features (daily aggregated). "
        "All features should be normalized to [0, 1] range."
    )
