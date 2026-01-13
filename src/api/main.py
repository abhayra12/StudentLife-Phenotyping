from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add project root
sys.path.append('.')

from src.api.schemas import StudentBehavior, PredictionResponse, AnomalyInput, AnomalyResponse

# Import Model Definitions
# We need to duplicate the class definitions or import them if the scripts allow
# Ideally, we should move model classes to src/models/ for reuse.
# For now, I will redefine them here to avoid import issues with the analysis scripts
# (or better, import them dynamically if they are structured well)

# Helper classes or dynamic imports below

# Problem: filenames 09_... are invalid identifiers for import without hacks.
# Solution: Redefine the small model classes here for stability.

class BehaviorTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(BehaviorTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        # self.pos_encoder = ... (Simplified for inference if we assume fixed size or handle differently)
        # Actually, let's copy the full class to be safe.
        
        # Simpler approach for this demo:
        # Just use the nn.TransformerEncoder structure directly or copy specific parts
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        # Quick & Dirty forward pass matching training logic
        # Ideally we load the EXACT class definition from training
        src = self.embedding(src)
        src = src.permute(1, 0, 2)
        # Skip PosEncoding for this quick re-def if we accept small drift, OR copy it.
        # Let's assume we copy logic fully.
        output = self.transformer_encoder(src)
        output = output[-1, :, :]
        output = self.decoder(output)
        return output.squeeze()

# Let's fix the Import Strategy.
# I will use importlib to import the modules dynamically because filenames have numbers.
import importlib.util

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load model definitions dynamically
transformer_mod = load_module('src/analysis/modeling/09_transformer.py', 'transformer_mod')
BehaviorTransformer = transformer_mod.BehaviorTransformer

autoencoder_mod = load_module('src/analysis/modeling/08_autoencoder.py', 'autoencoder_mod')
Autoencoder = autoencoder_mod.Autoencoder


models = {}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Models
    print("Loading models...")
    
    # 1. Transformer
    try:
        t_model = BehaviorTransformer(input_dim=11, d_model=32, nhead=4, num_layers=2) # Dim matched to training
        # Note: Input Dim is hardcoded here. In prod, save config.json with models.
        # We'll try to load state dict and if shape mismatch, it throws error.
        t_model.load_state_dict(torch.load('models/transformer_best.pth', map_location=DEVICE))
        t_model.to(DEVICE)
        t_model.eval()
        models['transformer'] = t_model
        print("Transformer loaded.")
    except Exception as e:
        print(f"Failed to load Transformer: {e}")

    # 2. Autoencoder
    try:
        ae_model = Autoencoder(input_dim=11, latent_dim=3)
        ae_model.load_state_dict(torch.load('models/autoencoder.pth', map_location=DEVICE))
        ae_model.to(DEVICE)
        ae_model.eval()
        models['autoencoder'] = ae_model
        print("Autoencoder loaded.")
    except Exception as e:
        print(f"Failed to load Autoencoder: {e}")
        
    yield
    print("Shutting down...")

app = FastAPI(title="StudentLife Behavior API", lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
async def predict_activity(data: StudentBehavior):
    if 'transformer' not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Convert input list to tensor
        # Expected shape: [seq_len, batch, features] -> [24, 1, 35]
        # BUT input schema 'features' is likely a flattened list or single timestep?
        # The model expects a sequence (24h).
        # For simplicity, let's assume the input IS the sequence (flattened or list of lists)
        # Adjusted schema to accept List[List[float]] would be better, but let's assume 
        # for this MVP we take a single time step and predict? No, Transformer needs history.
        # Let's Assume input is [24 * features] flattened, or handle reshaping.
        
        # Let's reshape assuming features=11, seq=24
        feats = np.array(data.features).reshape(24, -1) 
        tensor_in = torch.FloatTensor(feats).unsqueeze(1).to(DEVICE) # [24, 1, feat]
        
        with torch.no_grad():
            pred = models['transformer'](tensor_in)
            
        return PredictionResponse(
            participant_id=data.participant_id,
            predicted_activity_minutes=float(pred.item())
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/anomaly", response_model=AnomalyResponse)
async def detect_anomaly(data: AnomalyInput):
    if 'autoencoder' not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Input: Single vector (daily average) or sequence? 
        # AE was trained on standard scaled features (no sequence).
        feats = np.array(data.features).reshape(1, -1)
        tensor_in = torch.FloatTensor(feats).to(DEVICE)
        
        with torch.no_grad():
            recon, _ = models['autoencoder'](tensor_in)
            loss = torch.mean((tensor_in - recon)**2).item()
            
        # Hardcoded threshold from training log (approx 0.98)
        # Ideally load from config
        THRESHOLD = 0.98 
        
        return AnomalyResponse(
            is_anomaly=loss > THRESHOLD,
            reconstruction_error=loss,
            threshold=THRESHOLD
        )
    except Exception as e:
         raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
