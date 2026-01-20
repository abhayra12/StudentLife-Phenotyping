"""
MLflow Configuration Module
Centralized configuration for experiment tracking and model registry.
"""
import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MODELS_DIR = PROJECT_ROOT / "models"

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:///{MLFLOW_DIR}")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "studentlife-phenotyping")

# Model registry names
TRANSFORMER_MODEL_NAME = "behavioral-transformer"
AUTOENCODER_MODEL_NAME = "behavioral-autoencoder"


def setup_mlflow():
    """
    Initialize MLflow with project-specific configuration.
    Should be called at the start of training scripts.
    """
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            mlflow.create_experiment(
                EXPERIMENT_NAME,
                artifact_location=str(MLFLOW_DIR / "artifacts")
            )
    except Exception as e:
        print(f"Warning: Could not create experiment: {e}")
    
    # Set the experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Enable autologging for PyTorch
    mlflow.pytorch.autolog(
        log_every_n_epoch=1,
        log_models=False,  # We'll log manually for more control
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False
    )
    
    print(f"✓ MLflow initialized:")
    print(f"  - Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"  - Experiment: {EXPERIMENT_NAME}")
    
    return mlflow.get_experiment_by_name(EXPERIMENT_NAME)


def get_mlflow_client():
    """Get MLflow client for advanced operations."""
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def log_model_to_registry(model, model_name, signature=None, input_example=None, pip_requirements=None):
    """
    Log a PyTorch model to MLflow registry with versioning.
    
    Args:
        model: PyTorch model to log
        model_name: Name for the model in registry
        signature: MLflow model signature (optional)
        input_example: Example input for model (optional)
        pip_requirements: List of pip requirements (optional)
    
    Returns:
        ModelVersion object
    """
    try:
        # Log the model
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            registered_model_name=model_name
        )
        
        print(f"✓ Model logged to registry: {model_name}")
        print(f"  - Run ID: {mlflow.active_run().info.run_id}")
        print(f"  - Model URI: {model_info.model_uri}")
        
        return model_info
        
    except Exception as e:
        print(f"Error logging model to registry: {e}")
        return None


def load_model_from_registry(model_name, version="latest", stage=None):
    """
    Load a model from MLflow registry.
    
    Args:
        model_name: Name of the model in registry
        version: Version number or "latest" (default: "latest")
        stage: Model stage filter ("Staging", "Production", None)
    
    Returns:
        Loaded PyTorch model
    """
    try:
        if stage:
            model_uri = f"models:/{model_name}/{stage}"
        elif version == "latest":
            client = get_mlflow_client()
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model: {model_name}")
            latest_version = max(versions, key=lambda x: int(x.version))
            model_uri = f"models:/{model_name}/{latest_version.version}"
        else:
            model_uri = f"models:/{model_name}/{version}"
        
        model = mlflow.pytorch.load_model(model_uri)
        print(f"✓ Model loaded from registry: {model_uri}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model from registry: {e}")
        return None


def create_model_signature(input_shape, output_shape=None):
    """
    Create MLflow model signature from input/output shapes.
    
    Args:
        input_shape: Tuple representing input tensor shape
        output_shape: Tuple representing output tensor shape (optional)
    
    Returns:
        MLflow ModelSignature
    """
    from mlflow.models.signature import infer_signature
    import numpy as np
    
    # Create dummy data with the specified shape
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    if output_shape:
        dummy_output = np.random.randn(*output_shape).astype(np.float32)
    else:
        dummy_output = None
    
    return infer_signature(dummy_input, dummy_output)


if __name__ == "__main__":
    # Test MLflow setup
    print("Testing MLflow configuration...")
    setup_mlflow()
    
    # List registered models
    client = get_mlflow_client()
    models = client.search_registered_models()
    
    if models:
        print(f"\nRegistered models ({len(models)}):")
        for model in models:
            print(f"  - {model.name}")
            versions = client.search_model_versions(f"name='{model.name}'")
            for v in versions:
                print(f"    Version {v.version}: {v.current_stage}")
    else:
        print("\nNo registered models found.")
