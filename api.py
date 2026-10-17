"""Federated ScamShield - FastAPI Backend"""
import json, pathlib, sys
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.model import ScamMLP
from src import config

app = FastAPI(title="Federated ScamShield API", version="1.0.0")
RESULTS = ROOT / "results"
MODEL_PATH = ROOT / "models" / "fed_global.pt"
_model = None
_metrics = {}

def get_model():
    global _model
    if _model is None:
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        n_features = state["net.0.weight"].shape[1]
        _model = ScamMLP(n_features)
        _model.load_state_dict(state)
        _model.eval()
    return _model

def get_metrics():
    global _metrics
    if not _metrics:
        with open(RESULTS / "metrics.json") as f:
            _metrics = json.load(f)
    return _metrics

class Transaction(BaseModel):
    features: List[float] = Field(..., description="23 engineered features")

class BatchRequest(BaseModel):
    transactions: List[Transaction]

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float = 0.5

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int

@app.get("/")
def root():
    return {"service":"Federated ScamShield API","version":"1.0.0","platform":"IBM LinuxONE (s390x)"}

@app.get("/health")
def health():
    model = get_model()
    return {"status":"ok","model_loaded":True,"n_features":model.net[0].in_features}

@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction):
    model = get_model()
    expected = model.net[0].in_features
    if len(tx.features) != expected:
        raise HTTPException(status_code=422, detail=f"Expected {expected} features, got {len(tx.features)}")
    x = torch.tensor([tx.features], dtype=torch.float32)
    with torch.no_grad():
        prob = model(x).item()
    return PredictionResponse(fraud_probability=round(prob,6), is_fraud=prob>=0.5, threshold=0.5)

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    model = get_model()
    expected = model.net[0].in_features
    features = []
    for i,tx in enumerate(req.transactions):
        if len(tx.features) != expected:
            raise HTTPException(status_code=422, detail=f"Tx {i}: expected {expected} features, got {len(tx.features)}")
        features.append(tx.features)
    x = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        probs = model(x).squeeze(-1).tolist()
    if isinstance(probs, float):
        probs = [probs]
    preds = [PredictionResponse(fraud_probability=round(p,6), is_fraud=p>=0.5, threshold=0.5) for p in probs]
    return BatchResponse(predictions=preds, count=len(preds))

@app.get("/metrics")
def all_metrics():
    return get_metrics()

@app.get("/metrics/{setup}")
def setup_metrics(setup: str):
    m = get_metrics()
    key = f"setup_{setup.upper()}"
    if key not in m:
        raise HTTPException(status_code=404, detail="Use A, B, C, D, or E.")
    return {key: m[key]}

@app.get("/watchlist")
def watchlist_summary():
    m = get_metrics()
    d = m.get("setup_D",{})
    return {"rings_detected":d.get("rings_detected"),"total_rings":d.get("total_rings"),
            "ring_recall":d.get("ring_recall"),"ring_precision":d.get("ring_precision"),
            "cross_bank_full_recall":d.get("ring_full_recall"),
            "single_bank_full_recall":d.get("single_bank_baseline",{}).get("ring_full_recall")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
