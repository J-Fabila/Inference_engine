from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd

from pathlib import Path
import json
import joblib

from inference import load_model, InferenceEngine

MODEL_PATH = "sandbox/model"

model, metadata = load_model(MODEL_PATH)
engine = InferenceEngine(model, metadata)


app = FastAPI(title="Inference API")


class InputData(BaseModel):
    data: List[Dict[str, Any]]



@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/predict")
def predict(input_data: InputData):

    df = pd.DataFrame(input_data.data)

    preds = engine.predict(df)

    return {"predictions": preds}

@app.post("/reload")
def reload_model():
    global engine
    model, metadata = load_model("sandbox/model")
    engine = InferenceEngine(model, metadata)
    return {"status": "reloaded"}
