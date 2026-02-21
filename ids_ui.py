from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import numpy as np
import os

# Directory for model files
MODEL_DIR = r"D:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\ids_output\models"

# Global models dictionary
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    model_files = {
        "rf": "rf_model.pkl",
        "xgb": "xgb_model.pkl",
        "lightgbm": "lightgbm_model.pkl",
        "decision_tree": "decision_tree_model.pkl",
    }
    
    for name, file in model_files.items():
        model_path = os.path.join(MODEL_DIR, file)
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
            print(f"Loaded model: {name} from {model_path}")
        else:
            print(f"Model file not found: {file}")
    
    yield  # yield control back to FastAPI
    
    # Cleanup (if needed) when shutting down
    models.clear()

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Request schema
class PredictionRequest(BaseModel):
    features: list[float]

# Predict endpoint
@app.post("/predict/{model_name}")
def predict(model_name: str, request: PredictionRequest):
    # Check if model exists
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    model = models[model_name]

    # Prepare input features
    try:
        features = np.array(request.features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid input features")

    # Make prediction
    try:
        prediction = model.predict(features)
        probability = model.predict_proba(features).tolist() if hasattr(model, "predict_proba") else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    # Response
    return {
        "model": model_name,
        "prediction": prediction.tolist(),
        "probability": probability,
    }

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Intrusion Detection System API is running"}