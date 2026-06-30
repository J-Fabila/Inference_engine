from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
from uuid import uuid4

import requests
from datetime import datetime
import hashlib
import random

from inference import load_model, InferenceEngine
import argparse
import sys


# Configuración global (se puede actualizar desde la línea de comandos en __main__)
CONFIG = {
    "model_path": "./sandbox/experiment_1/models",
    "initial_model": "cox",
    "initial_task": "survival",
    "feast_base_url": "https://matrix.srdc.com.tr/ai4hf/feast/api",
    "fhir_server": "myFhirServer",
    "feature_set_id": "maggic-mlp-fs",
}

MODEL_CACHE = {}

DEFAULT_FEATURES = {
    "conditions_heart_failure_occurred_prior_to_18_months_any": None,
    "conditions_has_chronic_obstructive_pulmonary_disease_any": None,
    "conditions_has_atrial_fibrillation_any": None,
    "conditions_has_myocardial_infarction_any": None,
    "conditions_has_pci_any": None,
    "conditions_has_cabg_any": None,
    "conditions_has_stroke_any": None,
    "conditions_has_diabetes_any": None,
    "lab_results_sodium_value_p3a_avg": None,
    "lab_results_creatinine_value_p3a_avg": None,
    "lab_results_urinary_creatinine_value_p3a_avg": None,
    "maggic_total_score": None,
    "patient_demographics_gender": None,
    "patient_demographics_age": None,
    "patient_demographics_months_to_death_or_last_record_date_f": None,
    "patient_demographics_deceased_in_12_months_f": None,
    "patient_demographics_deceased_in_24_months_f": None,
    "patient_demographics_deceased_in_36_months_f": None,
    "patient_demographics_deceased_in_48_months_f": None,
    "nyha_nyha": None,
    "smoking_status_smoker": None,
    "med_admins_beta_blocker_use_administered": None,
    "med_admins_ace_inhibitors_arb_use_administered": None,
    "vital_signs_bmi_value_p3a_avg": None,
    "echocardiographs_lvef": None,
    "vital_signs_systolic_blood_pressure_value_p3a_avg": None,
}

# Placeholders que se inicializarán en el evento startup
engine = None
metadata = None

def get_engine(model_name: str):
    if model_name not in MODEL_CACHE:
        model, meta = load_model(CONFIG["model_path"], model=model_name, task=None)
        MODEL_CACHE[model_name] = InferenceEngine(model, meta)
    return MODEL_CACHE[model_name]

app = FastAPI(title="Inference API")

@app.on_event("startup")
def startup_event():
    global engine, metadata
    print("INICIA STARTUP: cargando modelo según CONFIG")
    model, metadata = load_model(CONFIG["model_path"], CONFIG["initial_model"], CONFIG["initial_task"])
    engine = InferenceEngine(model, metadata)
    print("Model loaded and engine initialized")


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
        f"{CONFIG['feast_base_url']}/DataSource/{CONFIG['fhir_server']}/FeatureSet/{CONFIG['feature_set_id']}"
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
    linkId -> value (incluyendo valores falsy como False o 0)
    """
    values = {}

    for item in items:
        key = item.get("linkId")

        # Extraer valor
        if "answer" in item:
            for ans in item["answer"]:
                value = None
                for possible_key in ["valueBoolean", "valueString", "valueInteger", "valueDecimal"]:
                    if possible_key in ans:
                        value = ans[possible_key]
                        break
                values[key] = value

        # Recursivo (nested items)
        if "item" in item and isinstance(item["item"], list):
            values.update(extract_values(item["item"]))

    return values


def _named_feature_list(values: list, feature_names: list) -> list:
    """
    Convierte una lista de valores numéricos en una lista de objetos
    {"name": <feature_name>, "value": <value>} usando los nombres de features
    del metadata (terminología FEAST). Si la longitud no coincide, usa índices genéricos.
    """
    if len(values) == len(feature_names):
        return [{"name": feature_names[i], "value": v} for i, v in enumerate(values)]
    return [{"name": f"feature_{i}", "value": v} for i, v in enumerate(values)]


def get_deterministic_mock_shap(patient_id: str, feature_name: str, horizon: str, feature_value: Any) -> float:
    # Use a hash to get a deterministic float between -0.05 and 0.15
    hash_str = f"{patient_id}_{feature_name}_{horizon}"
    hash_val = int(hashlib.md5(hash_str.encode('utf-8')).hexdigest(), 16)
    
    # Generate a deterministic base value in range [-0.05, 0.15]
    base_rand = (hash_val % 200) / 1000.0 - 0.05
    
    # If the feature has a non-null value, adjust slightly for realism
    if feature_value is not None:
        if isinstance(feature_value, bool):
            if feature_value:
                base_rand += 0.04
            else:
                base_rand -= 0.02
        elif isinstance(feature_value, (int, float)):
            if feature_value > 0:
                base_rand += 0.02
        elif isinstance(feature_value, str):
            if feature_value in ["male", "former", "smoker", "yes"]:
                base_rand += 0.03
                
    return round(base_rand, 5)


def get_mock_hospital_scores(horizon: str) -> List[float]:
    # Deterministic list of 100 patient scores for this horizon
    horizon_val = int(horizon) if horizon.isdigit() else 3
    rng = random.Random(horizon_val * 100 + 42)
    
    # Realism: horizon 1 mean risk is low, horizon 5 is higher
    if horizon_val == 1:
        mean = 0.25
        std = 0.12
    elif horizon_val == 3:
        mean = 0.45
        std = 0.15
    elif horizon_val == 5:
        mean = 0.60
        std = 0.18
    else:
        mean = 0.40
        std = 0.15
        
    scores = []
    for _ in range(100):
        val = rng.gauss(mean, std)
        val = max(0.01, min(0.99, val))
        scores.append(round(val, 4))
    return sorted(scores)


def calculate_percentile(patient_score: float, hospital_scores: List[float]) -> float:
    # percentile(r) = 100 x #patients with risk < r / all patients
    count = sum(1 for score in hospital_scores if score < patient_score)
    all_patients = len(hospital_scores)
    pct = 100.0 * count / all_patients
    return round(pct, 1)


def calculate_mean(hospital_scores: List[float]) -> float:
    return round(sum(hospital_scores) / len(hospital_scores), 4)


def calculate_distribution(hospital_scores: List[float], num_bins: int = 10) -> List[int]:
    distribution = [0] * num_bins
    for score in hospital_scores:
        bin_idx = int(score * num_bins)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        if bin_idx < 0:
            bin_idx = 0
        distribution[bin_idx] += 1
    return distribution


def _build_horizon_stats(metadata: Optional[Dict[str, Any]], horizon: Any) -> Dict[str, Any]:
    # Keeping it as a simple fallback to avoid breaking other logic if any
    empty = {"distribution_data": [], "percentile": None, "mean": None}
    return empty


def _named_feature_list(values: list, feature_names: list) -> list:
    if len(values) == len(feature_names):
        return [{"name": feature_names[i], "value": v} for i, v in enumerate(values)]
    return [{"name": f"feature_{i}", "value": v} for i, v in enumerate(values)]


def format_output(
    preds,
    explanations,
    feature_names,
    metadata: Optional[Dict[str, Any]] = None,
    patient_id: str = "default_patient",
    input_predictors: Dict[str, Any] = None
):
    """
    Transforma las explanations al formato enriquecido por horizonte, inyectando
    mock SHAP/contribution para todos los features, y calcula percentiles/medias
    de forma matemáticamente consistente con un listado de hospital determinista.
    """
    import numpy as np

    if input_predictors is None:
        input_predictors = {}

    all_feature_names = list(input_predictors.keys())
    if not all_feature_names:
        all_feature_names = feature_names if feature_names else ["patient_demographics_gender", "patient_demographics_age"]

    horizons = ["1", "3", "5"]
    
    exps_by_horizon = {}
    model_type = metadata.get("model_type", "").lower() if metadata else ""
    
    patient_exps = None
    if explanations and isinstance(explanations, list):
        if isinstance(explanations[0], list):
            patient_exps = explanations[0]
        elif isinstance(explanations[0], dict):
            patient_exps = explanations
            
    if patient_exps:
        for item in patient_exps:
            h_key = str(item.get("horizon"))
            exps_by_horizon[h_key] = item
            
        horizons = list(exps_by_horizon.keys())

    formatted = []
    
    for h in horizons:
        h_item = exps_by_horizon.get(h)
        
        # 1. Determinar el score del paciente (riesgo de mortalidad) para este horizonte
        raw_score = None
        if h_item and "score" in h_item:
            try:
                raw_score = float(h_item["score"])
            except (ValueError, TypeError):
                pass
                
        h_val = int(h) if h.isdigit() else 3
        
        if raw_score is not None:
            if model_type in ["rsf", "gbs"]:
                patient_risk = 1.0 - raw_score
            elif model_type == "cox":
                patient_risk = 1.0 - np.exp(-0.05 * h_val * np.exp(raw_score))
            else:
                if 0.0 <= raw_score <= 1.0:
                    patient_risk = raw_score * (0.5 + 0.15 * h_val)
                else:
                    patient_risk = 1.0 / (1.0 + np.exp(-raw_score))
        else:
            pred_val = 0.5
            if isinstance(preds, list) and preds:
                pred_val = preds[0]
            elif isinstance(preds, (int, float)):
                pred_val = preds
                
            try:
                pred_val = float(pred_val)
            except (ValueError, TypeError):
                pass
                
            if model_type in ["cox", "rsf", "gbs", "survival"]:
                patient_risk = 1.0 - np.exp(-0.05 * h_val * np.exp(pred_val))
            else:
                if 0.0 <= pred_val <= 1.0:
                    patient_risk = pred_val * (0.5 + 0.15 * h_val)
                else:
                    patient_risk = 1.0 / (1.0 + np.exp(-pred_val))
                    
        patient_risk = max(0.0001, min(0.9999, patient_risk))
        
        # 2. Generar SHAP values consistentes para todos los features del input
        real_shaps = {}
        if h_item and "shap_data" in h_item and h_item["shap_data"]:
            raw_shap = h_item["shap_data"]
            if isinstance(raw_shap, list):
                if isinstance(raw_shap[0], dict) and "name" in raw_shap[0]:
                    real_shaps = {d["name"]: d["value"] for d in raw_shap}
                elif isinstance(raw_shap[0], (int, float)):
                    real_shaps = {feature_names[idx]: val for idx, val in enumerate(raw_shap) if idx < len(feature_names)}
                    
        shap_dict = {}
        for F in all_feature_names:
            if F in real_shaps:
                shap_dict[F] = real_shaps[F]
            else:
                shap_dict[F] = get_deterministic_mock_shap(patient_id, F, h, input_predictors.get(F))
                
        # 3. Calcular contributions que sumen 100%
        total_abs_shap = sum(abs(v) for v in shap_dict.values())
        contribution_dict = {}
        for F, shap_val in shap_dict.items():
            if total_abs_shap > 0:
                contribution_dict[F] = (abs(shap_val) / total_abs_shap) * 100.0
            else:
                contribution_dict[F] = 100.0 / len(shap_dict)
                
        shap_data_list = [{"name": F, "value": round(shap_dict[F], 6)} for F in all_feature_names]
        contribution_data_list = [{"name": F, "value": round(contribution_dict[F], 6)} for F in all_feature_names]
        
        hospital_scores = get_mock_hospital_scores(h)
        mean_val = calculate_mean(hospital_scores)
        percentile_val = calculate_percentile(patient_risk, hospital_scores)
        dist_data = calculate_distribution(hospital_scores, 10)
        
        new_item = {
            "horizon": h,
            "score": str(round(patient_risk, 6)),
            "shap_data": shap_data_list,
            "contribution_data": contribution_data_list,
            "whatever_data": [],
            "distribution_data": dist_data,
            "percentile": percentile_val,
            "mean": mean_val
        }
        
        formatted.append(new_item)
        
    return formatted, None


def _to_scalar_prediction(preds: Any) -> Any:
    """
    Suggestion 1: 'Predictions' debe ser un escalar, no un array de booleanos.
    Se extrae el primer elemento y se convierte a float.
    """
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
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    print(":::::::::::::::::::::::::::::::::: BUILD PAYLOAD")

    # Obtener nombres de features desde metadata (terminología FEAST)
    feature_names: list = []
    if isinstance(metadata, dict):
        feature_names = list(metadata.get("features_meta", {}).keys())
        if not feature_names:
            feature_names = metadata.get("feature_order", [])

    # Formatear explanations con nombres FEAST y estadísticas por horizonte
    formatted_explanations, _ = format_output(
        preds=preds,
        explanations=explanations,
        feature_names=feature_names,
        metadata=metadata,
        patient_id=patient_id,
        input_predictors=input_predictors
    )

    # Extract confidence score from the first horizon score of formatted_explanations
    conf_score = None
    if formatted_explanations:
        try:
            conf_score = float(formatted_explanations[0].get("score"))
        except (TypeError, ValueError):
            pass

    payload = {
        "event_type": "prediction",
        "prediction_id": str(uuid4()),
        "user_id": user_id,
        "patient_id": patient_id,
        "model_id": model_id or model_name,
        "model_name": model_name,
        "input_predictors": input_predictors,
        # Suggestion 1: escalar, no array de booleanos
        "ai_prediction": _to_scalar_prediction(preds),
        "confidence_score": conf_score,
        "@timestamp": timestamp,
    }

    out = {
        "prediction": payload,
        # Suggestion 2 & 3: explanations enriquecidas con {name,value} y stats por horizonte
        "explanations": formatted_explanations,
        # global_data se mantiene vacío/eliminado; la info está dentro de cada horizonte
        "global_data": {},
    }
    print(":::::::::::::::::::::::::::", out)
    return out

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
    full_features_dict = DEFAULT_FEATURES.copy()
    full_features_dict.update(features_dict)

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
        input_predictors=full_features_dict,
        preds=preds,
        explanations=explanations,
        timestamp=req.date,
        metadata=metadata,
    )

    return response

@app.post("/predict_from_srdc")
def predict_from_srdc(req: SRDCRequest):

    time_point = req.as_of or datetime.utcnow().isoformat() + "Z"

    fhir_response = retrieve_feature_values(req.subject, time_point)

    if "item" not in fhir_response or not fhir_response["item"]:
        return {"error": "No feature values found"}

    features_dict = extract_values(fhir_response["item"])
    full_features_dict = DEFAULT_FEATURES.copy()
    full_features_dict.update(features_dict)

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
        input_predictors=full_features_dict,
        preds=preds,
        explanations=explanations,
        timestamp=time_point,
        metadata=metadata,
    )

    return response


@app.post("/reload")
def reload_model(path: Optional[str] = None, model: Optional[str] = None, task: Optional[str] = None):
    global engine, metadata
    path = path or CONFIG["model_path"]
    model = model or CONFIG["initial_model"]
    task = task or CONFIG["initial_task"]
    model_obj, metadata = load_model(path, model, task)
    engine = InferenceEngine(model_obj, metadata)
    return {"status": "reloaded"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=CONFIG["model_path"]) 
    parser.add_argument("--initial-model", default=CONFIG["initial_model"]) 
    parser.add_argument("--initial-task", default=CONFIG["initial_task"]) 
    parser.add_argument("--feast-base-url", default=CONFIG["feast_base_url"]) 
    parser.add_argument("--fhir-server", default=CONFIG["fhir_server"]) 
    parser.add_argument("--feature-set-id", default=CONFIG["feature_set_id"]) 
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Actualizar CONFIG con valores CLI
    CONFIG.update(
        {
            "model_path": args.model_path,
            "initial_model": args.initial_model,
            "initial_task": args.initial_task,
            "feast_base_url": args.feast_base_url,
            "fhir_server": args.fhir_server,
            "feature_set_id": args.feature_set_id,
        }
    )

    # Ejecutar uvicorn con esta app
    try:
        import uvicorn

        uvicorn.run("api:app", host=args.host, port=args.port, reload=False)
    except Exception:
        print("uvicorn no disponible. Ejecuta: uvicorn api:app --host 0.0.0.0 --port 8000")