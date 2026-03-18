from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd

import requests
from datetime import datetime

from inference import load_model, InferenceEngine

MODEL_PATH = "sandbox/model"

model, metadata = load_model(MODEL_PATH)
engine = InferenceEngine(model, metadata)


FEAST_BASE_URL = "https://matrix.srdc.com.tr/ai4hf/feast/api"
FHIR_SERVER = "myFhirServer"
FEATURE_SET_ID = "maggic-mlp-fs"

app = FastAPI(title="Inference API")


class InputData(BaseModel):
    data: List[Dict[str, Any]]


class SRDCRequest(BaseModel):
    subject: str
    as_of: Optional[str] = None  # ISO time opcional

def retrieve_feature_values(subject: str, time_point: str):
    url = (
        f"{FEAST_BASE_URL}/DataSource/{FHIR_SERVER}/FeatureSet/{FEATURE_SET_ID}"
        f"/$retrieve-feature-values"
        f"?subject={subject}&asOf={time_point}&format=fhir&outcome=true"
    )

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Error fetching features: {response.text}")

    return response.json()


def extract_values(items):
    """
    Convierte el FHIR QuestionnaireResponse a dict plano:
    linkId -> value
    """
    values = {}

    for item in items:
        key = item.get("linkId")

        # Extraer valor
        if "answer" in item:
            for ans in item["answer"]:
                value = (
                    ans.get("valueBoolean")
                    or ans.get("valueString")
                    or ans.get("valueInteger")
                    or ans.get("valueDecimal")
                )
                values[key] = value

        # Recursivo (nested items)
        if "item" in item and isinstance(item["item"], list):
            values.update(extract_values(item["item"]))

    return values


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/predict")
def predict(input_data: InputData):

    df = pd.DataFrame(input_data.data)

    preds = engine.predict(df)

    return {"predictions": preds}

@app.post("/predict_from_srdc")
def predict_from_srdc(req: SRDCRequest):

    time_point = req.as_of or datetime.utcnow().isoformat() + "Z"

    fhir_response = retrieve_feature_values(req.subject, time_point)

    if "item" not in fhir_response or not fhir_response["item"]:
        return {"error": "No feature values found"}

    features_dict = extract_values(fhir_response["item"])

    df = pd.DataFrame([features_dict])

    try:
        if "feature_order" in metadata:
            df = df[metadata["feature_order"]]
    except Exception:
        pass

    df = df.fillna(0)

    try:
        df = df.astype(float)
    except Exception:
        pass

    preds = engine.predict(df)

    return {
        "input_features": features_dict,
        "predictions": preds
    }


@app.post("/reload")
def reload_model():
    global engine
    model, metadata = load_model("sandbox/model")
    engine = InferenceEngine(model, metadata)
    return {"status": "reloaded"}