"""
API Schemas for StudentLife Behavioral Prediction

This module defines Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import math

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


def generate_realistic_example() -> List[float]:
    """Generate a realistic 24-hour behavioral pattern for documentation."""
    features = []
    for hour in range(24):
        # Circadian encoding
        hour_sin = round(math.sin(2 * math.pi * hour / 24), 3)
        hour_cos = round(math.cos(2 * math.pi * hour / 24), 3)
        # Weekly encoding (Tuesday)
        day_sin = round(math.sin(2 * math.pi * 2 / 7), 3)
        day_cos = round(math.cos(2 * math.pi * 2 / 7), 3)
        
        # Activity pattern varies by hour
        if 0 <= hour < 6:  # Night - sleeping
            stationary, active, voice, noise, entropy, sleep = 0.95, 0.02, 0.0, 0.05, 0.1, 0.8
        elif 6 <= hour < 9:  # Morning - waking up
            stationary, active, voice, noise, entropy, sleep = 0.6, 0.25, 0.1, 0.2, 0.3, 0.6
        elif 9 <= hour < 12:  # Late morning - class/active
            stationary, active, voice, noise, entropy, sleep = 0.4, 0.45, 0.3, 0.35, 0.6, 0.5
        elif 12 <= hour < 14:  # Lunch
            stationary, active, voice, noise, entropy, sleep = 0.5, 0.35, 0.4, 0.4, 0.5, 0.5
        elif 14 <= hour < 18:  # Afternoon - class/study
            stationary, active, voice, noise, entropy, sleep = 0.45, 0.40, 0.25, 0.3, 0.55, 0.5
        elif 18 <= hour < 21:  # Evening - social/relaxing
            stationary, active, voice, noise, entropy, sleep = 0.55, 0.30, 0.35, 0.25, 0.45, 0.5
        else:  # Late night - winding down
            stationary, active, voice, noise, entropy, sleep = 0.75, 0.15, 0.1, 0.1, 0.2, 0.6
        
        week_of_term = 0.5  # Mid-term (week 5)
        
        features.extend([
            hour_sin, hour_cos, day_sin, day_cos,
            stationary, active, voice, noise, entropy, sleep, week_of_term
        ])
    
    return features


# Pre-generate example for documentation
EXAMPLE_264_FEATURES = generate_realistic_example()
EXAMPLE_11_FEATURES = [0.5, 0.866, 0.782, 0.623, 0.55, 0.35, 0.25, 0.28, 0.45, 0.6, 0.5]


class StudentBehavior(BaseModel):
    """Input schema for activity prediction.
    
    Requires 264 features: 24 hours × 11 features per hour.
    Features should be normalized to [0, 1] range for best results.
    
    **Tip:** Use the `/examples` endpoint to get copy-paste ready payloads!
    """
    participant_id: Optional[str] = Field(
        default="anonymous",
        description="Identifier for the student/participant (e.g., 'u42', 'demo_user')",
        examples=["u42", "demo_user", "test_participant"]
    )
    features: List[float] = Field(
        ...,
        description=(
            "264 normalized feature values representing 24 hours of behavioral data. "
            "Structure: [hour_0_feature_0, hour_0_feature_1, ..., hour_0_feature_10, "
            "hour_1_feature_0, ..., hour_23_feature_10]. "
            "Use /examples endpoint for a ready-to-use sample."
        ),
        min_length=264,
        max_length=264
    )
    
    @field_validator('features')
    @classmethod
    def validate_feature_length(cls, v):
        if len(v) != EXPECTED_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_FEATURES} features (24 hours × 11 features), "
                f"got {len(v)}. Use GET /examples to get a valid sample payload."
            )
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "participant_id": "demo_user",
                    "features": EXAMPLE_264_FEATURES
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
    
    **Use Case:** Detect unusual behavioral patterns that may indicate
    stress, depression, or other mental health concerns.
    
    **Tip:** Use the `/examples` endpoint to get copy-paste ready payloads!
    """
    features: List[float] = Field(
        ...,
        description=(
            "11 daily aggregated feature values. Order: hour_sin, hour_cos, "
            "day_of_week_sin, day_of_week_cos, activity_stationary_pct, "
            "activity_active_minutes, audio_voice_minutes, audio_noise_minutes, "
            "location_entropy, sleep_duration_rolling, week_of_term. "
            "All values should be normalized to [0, 1]."
        ),
        min_length=11,
        max_length=11
    )
    is_weekend: Optional[bool] = Field(
        default=False,
        description=(
            "Whether this is weekend data. Weekend behavior naturally differs "
            "(less structured, more varied), so a separate threshold is applied."
        ),
        examples=[False, True]
    )
    
    @field_validator('features')
    @classmethod
    def validate_feature_length(cls, v):
        if len(v) != FEATURES_PER_HOUR:
            raise ValueError(
                f"Expected {FEATURES_PER_HOUR} features (daily aggregated), "
                f"got {len(v)}. Use GET /examples to get a valid sample payload."
            )
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": EXAMPLE_11_FEATURES,
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


class ExamplePayloads(BaseModel):
    """Ready-to-use example payloads for testing the API."""
    predict_request: dict = Field(
        description="Copy this payload to test the /predict endpoint"
    )
    anomaly_request_normal: dict = Field(
        description="Copy this to test /anomaly with normal behavior"
    )
    anomaly_request_concerning: dict = Field(
        description="Copy this to test /anomaly with concerning behavior"
    )
    feature_guide: dict = Field(
        description="Explanation of each feature"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predict_request": {
                        "participant_id": "demo_user",
                        "features": "[264 values - see actual response]"
                    },
                    "anomaly_request_normal": {
                        "features": "[11 values]",
                        "is_weekend": False
                    },
                    "anomaly_request_concerning": {
                        "features": "[11 values - low activity pattern]",
                        "is_weekend": False
                    },
                    "feature_guide": {"hour_sin": "sin(2π × hour/24)"}
                }
            ]
        }
    }
