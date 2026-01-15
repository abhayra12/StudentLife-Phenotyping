"""
Training Script Entry Point
This script creates a reproducible training run by calling the core pipeline.
Actual logic is modularized in `src/analysis/modeling/09_transformer.py`.
"""
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.analysis.modeling import _09_transformer as transformer_trainer

if __name__ == "__main__":
    print("🚀 Starting SOTA Transformer Training...")
    # Execute the training logic
    transformer_trainer.main()
    print("\n✅ Training Complete. Model saved to 'models/transformer_best.pth'")
