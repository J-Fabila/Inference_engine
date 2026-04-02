from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
from uuid import uuid4

import requests
from datetime import datetime

from inference import load_model, InferenceEngine

MODEL_PATH = "./sandbox/experiment_1/models"

MODEL_CACHE = {}

def get_engine(model_name: str):
    if model_name not in MODEL_CACHE:
        model, metadata = load_model(MODEL_PATH, model=model_name, task=None)
        MODEL_CACHE[model_name] = InferenceEngine(model, metadata)
    return MODEL_CACHE[model_name]

model, metadata = load_model(MODEL_PATH,"cox", "survival")
engine = InferenceEngine(model, metadata)


FEAST_BASE_URL = "https://matrix.srdc.com.tr/ai4hf/feast/api"
FHIR_SERVER = "myFhirServer"
FEATURE_SET_ID = "maggic-mlp-fs"

app = FastAPI(title="Inference API")

class PredictRequest(BaseModel):
    patientId: str
    date: str
    model_name: str
    user_id: Optional[str] = None
    model_id: Optional[str] = None

class SRDCRequest(BaseModel):
    subject: str
    as_of: Optional[str] = None  # ISO time opcional
    user_id: Optional[str] = None
    model_name: Optional[str] = None
    model_id: Optional[str] = None

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

def format_output(preds, explanations, feature_names):
    if not explanations or not isinstance(explanations, list):
        return preds, explanations

    if isinstance(explanations[0], list):
        patient_exps = explanations[0]
    elif isinstance(explanations[0], dict):
        patient_exps = explanations
    else:
        return preds, explanations

    formatted = []

    for item in patient_exps:

        new_item = {}

        for key in ["horizon", "score"]:
            if key in item:
                new_item[key] = str(item[key])

        for key, value in item.items():

            if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):

                if len(value) == len(feature_names):
                    new_item[key] = [
                        {"key": feature_names[i], "value": v}
                        for i, v in enumerate(value)
                    ]
                else:
                    new_item[key] = [
                        {"key": f"feature_{i}", "value": v}
                        for i, v in enumerate(value)
                    ]

        formatted.append(new_item)

    return formatted, None


def _to_scalar_prediction(preds: Any) -> Any:
    if isinstance(preds, list) and preds:
        value = preds[0]
    else:
        value = preds

    if isinstance(value, bool):
        return float(value)

    return value


def _extract_confidence_score(explanations: Any) -> Optional[float]:
    if not explanations:
        return None

    patient_explanations = None

    if isinstance(explanations, list) and explanations:
        if isinstance(explanations[0], list) and explanations[0]:
            patient_explanations = explanations[0]
        elif isinstance(explanations[0], dict):
            patient_explanations = explanations

    if not patient_explanations:
        return None

    first_item = patient_explanations[0]
    score = first_item.get("score")

    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def build_prediction_payload(
    *,
    patient_id: str,
    model_name: str,
    model_id: Optional[str],
    user_id: Optional[str],
    input_predictors: Dict[str, Any],
    preds: Any,
    explanations: Any,
    timestamp: str,
) -> Dict[str, Any]:
    payload = {
        "event_type": "prediction",
        "prediction_id": str(uuid4()),
        "user_id": user_id,
        "patient_id": patient_id,
        "model_id": model_id or model_name,
        "model_name": model_name,
        "input_predictors": input_predictors,
        "ai_prediction": _to_scalar_prediction(preds),
        "confidence_score": _extract_confidence_score(explanations),
        "@timestamp": timestamp,
    }

    return payload

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(req: PredictRequest):

    # 1. obtener engine dinámico
    engine = get_engine(req.model_name)
    metadata = engine.metadata

    # 2. obtener datos desde SRDC (FHIR)
    time_point = req.date

    fhir_response = retrieve_feature_values(req.patientId, time_point)

    if "item" not in fhir_response or not fhir_response["item"]:
        return {"error": "No feature values found"}

    features_dict = extract_values(fhir_response["item"])

    # 3. convertir a DataFrame
    df = pd.DataFrame([features_dict])

    # 4. ordenar columnas si aplica
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

    # 5. predicción
    preds = engine.predict(df)

    explanations = None
    if hasattr(engine.model, "explain"):
        explanations = engine.explain(df)

    response = build_prediction_payload(
        patient_id=req.patientId,
        model_name=req.model_name,
        model_id=req.model_id or metadata.get("model_id"),
        user_id=req.user_id,
        input_predictors=features_dict,
        preds=preds,
        explanations=explanations,
        timestamp=req.date,
    )

    return response

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
    explanations = None
    if hasattr(engine.model, "explain"):
        explanations = engine.explain(df)

    default_model_name = metadata.get("model_type", "model")
    response = build_prediction_payload(
        patient_id=req.subject,
        model_name=req.model_name or default_model_name,
        model_id=req.model_id or metadata.get("model_id"),
        user_id=req.user_id,
        input_predictors=features_dict,
        preds=preds,
        explanations=explanations,
        timestamp=time_point,
    )

    return response


@app.post("/reload")
def reload_model():
    global engine
    model, metadata = load_model("./sandbox/experiment_1/models","cox", "survival")
    engine = InferenceEngine(model, metadata)
    return {"status": "reloaded"}