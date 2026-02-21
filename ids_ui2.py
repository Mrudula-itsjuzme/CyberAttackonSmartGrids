from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import logging
from typing import List, Optional

# Directory for model files
MODEL_DIR = r"D:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\ids_output\models"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global models dictionary
models = {}

# Define schema for prediction request
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_items=1)  # List of features with at least one element

# Define schema for the response
class PredictionResponse(BaseModel):
    model: str
    prediction: List[int]
    probability: Optional[List[List[float]]]  # Optional probability if available

# Load models
def load_model(model_name: str):
    model_file = f"{model_name}_model.pkl"
    model_path = os.path.join(MODEL_DIR, model_file)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model = joblib.load(model_path)
    logger.info(f"Loaded model: {model_name} from {model_path}")
    return model

# Initialize FastAPI app
app = FastAPI()

# Predict endpoint
@app.post("/predict/{model_name}", response_model=PredictionResponse)
def predict(model_name: str, request: PredictionRequest):
    # Check if model exists
    if model_name not in models:
        try:
            model = load_model(model_name)
            models[model_name] = model
        except HTTPException as e:
            raise e
    
    model = models[model_name]

    # Prepare input features
    try:
        features = np.array(request.features).reshape(1, -1)
    except Exception as e:
        logger.error(f"Error reshaping input features: {e}")
        raise HTTPException(status_code=400, detail="Invalid input features format")

    # Make prediction
    try:
        prediction = model.predict(features).tolist()
        probability = model.predict_proba(features).tolist() if hasattr(model, "predict_proba") else None
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed")

    # Return prediction response
    return {
        "model": model_name,
        "prediction": prediction,
        "probability": probability,
    }

# Root endpoint
@app.get("/", response_model=dict)
def read_root():
    return {"message": "Intrusion Detection System API is running"}
