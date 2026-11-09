
"""
AI4HF LASSO Inference API

Provides:
- Patient feature retrieval from SRDC / FEAST
- LASSO 12-month mortality risk prediction
- Patient-specific SHAP explanations
- Response formatted according to Panos's expected structure
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ---------------------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------------------

ENGINE_DIR = Path(__file__).resolve().parent

if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from lasso.lasso_inference import LassoInferenceEngine


CONFIG = {
    "model_path": os.getenv(
        "LASSO_MODEL_PATH",
        str(ENGINE_DIR / "lasso" / "lasso_model.json")
    ),
    "feast_base_url": os.getenv(
        "FEAST_BASE_URL",
        "http://host.docker.internal/dt4h/feast/api"
    ),
    "fhir_server": os.getenv(
        "FHIR_SERVER",
        "myFhirServer"
    ),
    "feature_set_id": os.getenv(
        "FEATURE_SET_ID",
        "care-heart-inpatient-fs"
    ),
    "model_id": os.getenv(
        "LASSO_MODEL_ID",
        "care-heart-inpatient"
    ),
    "model_name": os.getenv(
        "LASSO_MODEL_NAME",
        "care-heart-inpatient"
    ),
}


# ---------------------------------------------------------------------
# INFERENCE ENGINE
# ---------------------------------------------------------------------

lasso_engine: Optional[LassoInferenceEngine] = None


app = FastAPI(
    title="AI4HF LASSO Inference API",
    description=(
        "Inference service for the AI4HF LASSO mortality prediction model. "
        "Provides 12-month mortality risk and SHAP explanations."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    """Load the LASSO model when the API starts."""

    global lasso_engine

    print("[*] Initializing LASSO Inference Engine...")

    try:
        lasso_engine = LassoInferenceEngine(CONFIG["model_path"])

        print("[+] LASSO Inference Engine initialized successfully!")
        print(f"[+] Model path: {CONFIG['model_path']}")

    except Exception as e:
        lasso_engine = None
        print(f"[!] Could not load LASSO model: {e}")


# ---------------------------------------------------------------------
# REQUEST SCHEMAS
# ---------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request containing patient information for SRDC retrieval."""

    patientId: str
    vignetteDate: str

    modelName: Optional[str] = "care-heart-inpatient-fs"

    encounterId: Optional[str] = None
    eligibilityEventTime: Optional[str] = None
    eligibilityExitTime: Optional[str] = None

    userId: Optional[str] = None


class PredictRawRequest(BaseModel):
    """Request containing already retrieved patient features."""

    patientId: str
    features: Dict[str, Any]

    userId: Optional[str] = None


# ---------------------------------------------------------------------
# SRDC / FEAST DATA RETRIEVAL
# ---------------------------------------------------------------------

def retrieve_feature_values(
    subject: str,
    time_point: str,
    eligibility_event_time: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieves patient features from SRDC FEAST."""

    url = (
        f"{CONFIG['feast_base_url']}/DataSource/"
        f"{CONFIG['fhir_server']}/FeatureSet/"
        f"{CONFIG['feature_set_id']}"
        f"/$retrieve-feature-values"
    )

    params = {
        "subject": subject,
        "asOf": time_point,
        "eligibilityEventTime": eligibility_event_time,
        "format": "raw",
        "outcome": "true",
    }

    try:
        response = requests.get(url, params=params, timeout=60)

        if response.status_code == 200:
            data = response.json()
            print("FEAST raw response type:", type(data).__name__)
            return data

        raise HTTPException(
            status_code=502,
            detail={
                "message": "SRDC FEAST request failed",
                "http_status": response.status_code,
                "response_body": response.text,
            }
        )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to SRDC FEAST: {str(e)}"
        )
# ---------------------------------------------------------------------
# FHIR FEATURE EXTRACTION
# ---------------------------------------------------------------------

def get_fhir_answer_value(answer: Dict[str, Any]) -> Any:
    """Extract a value from a FHIR answer."""

    possible_keys = [
        "valueBoolean",
        "valueString",
        "valueInteger",
        "valueDecimal",
    ]

    for key in possible_keys:
        if key in answer:
            return answer[key]

    if "valueQuantity" in answer:
        quantity = answer["valueQuantity"]

        if isinstance(quantity, dict):
            return quantity.get("value")

    if "value" in answer:
        return answer["value"]

    return None

def inspect_missing_feature_candidates(data):
    terms = ["egfr", "age"]

    def recursive_inspect(node, path="root"):
        if isinstance(node, dict):

            link_id = node.get("linkId")

            if isinstance(link_id, str) and any(
                term in link_id.lower() for term in terms
            ):
                answer = node.get("answer", [])

                print(
                    "[FEAST DEBUG]",
                    "path:", path,
                    "linkId:", link_id,
                    "answer_present:", bool(answer),
                    "answer_keys:", [
                        list(a.keys())
                        for a in answer
                        if isinstance(a, dict)
                    ]
                )

            for key, value in node.items():
                if any(term in key.lower() for term in terms):
                    print(
                        "[FEAST DEBUG] Matching key:",
                        key,
                        "type:",
                        type(value).__name__
                    )

                if isinstance(value, (dict, list)):
                    recursive_inspect(value, f"{path}/{key}")

        elif isinstance(node, list):
            for i, item in enumerate(node):
                recursive_inspect(item, f"{path}[{i}]")

    recursive_inspect(data)
    
    
    
def inspect_egfr_fields(data, path="root"):
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}/{key}"

            if "egfr" in key.lower():
                print(
                    "[EGFR DEBUG]",
                    "path:", current_path,
                    "type:", type(value).__name__,
                    "is_null:", value is None,
                    "nested_keys:",
                    list(value.keys()) if isinstance(value, dict) else None
                )

            if isinstance(value, (dict, list)):
                inspect_egfr_fields(value, current_path)

    elif isinstance(data, list):
        for i, item in enumerate(data):
            inspect_egfr_fields(item, f"{path}[{i}]")

def extract_lasso_features(
    data: Dict[str, Any],
    required_features
) -> Dict[str, Any]:
    """
    Extract LASSO features from the FEAST response.

    Supports:
    - FHIR QuestionnaireResponse items using linkId and answer
    - Nested dictionaries containing feature names directly
    """

    extracted = {}
    required_features = set(required_features)

    def recursive_extract(node):

        if isinstance(node, dict):

            # FHIR QuestionnaireResponse item.
            link_id = node.get("linkId")

            if (
                link_id in required_features
                and isinstance(node.get("answer"), list)
                and len(node["answer"]) > 0
            ):
                value = get_fhir_answer_value(node["answer"][0])

                if value is not None:
                    extracted[link_id] = value

            # Feature names directly present in the response.
            for key, value in node.items():

                if key in required_features:

                    if not isinstance(value, (dict, list)):
                        if value is not None:
                            extracted[key] = value

                    elif isinstance(value, dict) and "value" in value:
                        extracted[key] = value["value"]

                if isinstance(value, (dict, list)):
                    recursive_extract(value)

        elif isinstance(node, list):

            for item in node:
                recursive_extract(item)

    recursive_extract(data)

    return extracted


# ---------------------------------------------------------------------
# RESPONSE GENERATION
# ---------------------------------------------------------------------

def build_lasso_response(
    *,
    patient_id: str,
    features: Dict[str, Any],
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate the LASSO prediction and format it according to
    Panos's expected response structure.
    """

    if lasso_engine is None:
        raise HTTPException(
            status_code=503,
            detail="LASSO inference engine is not loaded."
        )

    try:
        # Generate prediction.
        prediction_result = lasso_engine.predict(features)

        # Generate patient-specific SHAP explanations.
        shap_data = lasso_engine.explain(features)

    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not process patient features: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LASSO inference failed: {str(e)}"
        )

    prediction_score = float(prediction_result["prediction"])

    formatted_shap_data = [
        {
            "name": item["name"],
            "value": float(item["value"])
        }
        for item in shap_data
    ]

    timestamp = (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    # Synthetic example values for demonstration purposes.
    # These are placeholders, not clinically validated statistics.

    explanation_template = {
        "score": f"{prediction_score:.6f}",

        # Synthetic confidence interval.
        "scoreCI": [0.35, 0.42],

        # Same data as the patient-specific SHAP values.
        "shap_data": formatted_shap_data,
        "contribution_data": formatted_shap_data,

        # Synthetic distribution represented as integer bin counts.
        "distribution_data": [10, 5, 11, 12, 13],

        # Synthetic population statistics.
        "percentile": 65,
        "mean": 0.31,
        "median": 0.28,

        "whatever_data": [],
    }

    horizons = [
        "all_cause_mortality_6mo",
        "all_cause_mortality_1y",
        "rehospitalization_6mo",
        "rehospitalization_1y",
    ]

    explanations = [
        {
            **explanation_template,
            "horizon": horizon,
        }
        for horizon in horizons
    ]

    response = {
        "care-heart-inpatient-fs": {
            "prediction_id": str(uuid4()),
            "user_id": user_id,
            "patient_id": patient_id,
            "model_id": CONFIG["model_id"],
            "model_name": CONFIG["model_name"],
            "@timestamp": timestamp,
            "explanations": explanations,
        }
    }

    return response


# ---------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------

@app.get("/")
def root():
    """Basic service status."""

    return {
        "message": "AI4HF LASSO Inference API is online",
        "model": CONFIG["model_name"],
        "status": (
            "ready"
            if lasso_engine is not None
            else "model_not_loaded"
        ),
        "feature_set": CONFIG["feature_set_id"],
    }


@app.get("/health")
def health():
    """Check whether the LASSO model is loaded."""

    if lasso_engine is None:
        raise HTTPException(
            status_code=503,
            detail="LASSO model is not loaded."
        )

    return {
        "status": "healthy",
        "model_type": "LASSO Logistic Regression",
        "outcome": lasso_engine.outcome_name,
        "features": lasso_engine.feature_names,
        "prediction_horizon_months": 12,
    }


@app.post("/predict")
@app.post("/predict_from_srdc")
def predict_from_srdc(request: PredictRequest):
    """
    Retrieve patient data from SRDC FEAST,
    extract the required features and generate a prediction.
    """

    if lasso_engine is None:
        raise HTTPException(
            status_code=503,
            detail="LASSO inference engine is not loaded."
        )

    # Retrieve patient data from FEAST.
    fhir_data = retrieve_feature_values(
        subject=f"{request.patientId}|{request.encounterId}",
        time_point=request.vignetteDate,
        eligibility_event_time=request.eligibilityEventTime
    )

    # Extract the features required by LASSO.
    features = extract_lasso_features(
        fhir_data,
        lasso_engine.feature_names
    )
    
    # TEMPORARY IMPUTATION FOR TECHNICAL TESTING
    imputed_features = []

    if features.get("lab_results_eGFR_value_avg") is None:
        features["lab_results_eGFR_value_avg"] = (
            lasso_engine.standardization[
                "lab_results_eGFR_value_avg"
            ]["mean"]
        )

        imputed_features.append("lab_results_eGFR_value_avg")

        print(
            "[IMPUTATION] lab_results_eGFR_value_avg "
            "imputed with training mean"
        )
    
    inspect_egfr_fields(fhir_data)
    inspect_missing_feature_candidates(fhir_data)

    # Check whether all required features are available.
    missing_features = [
        feature
        for feature in lasso_engine.feature_names
        if feature not in features or features[feature] is None
    ]

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Some required LASSO features are missing.",
                "missing_features": missing_features,
                "extracted_features": list(features.keys()),
            }
        )

    response = build_lasso_response(
        patient_id=request.patientId,
        features=features,
        user_id=request.userId,
    )

    response["care-heart-inpatient-fs"]["imputed_features"] = imputed_features

    return response

@app.post("/predict_raw")
def predict_raw(request: PredictRawRequest):
    """
    Generate a prediction using features supplied directly
    in the request, without contacting SRDC FEAST.
    """

    if lasso_engine is None:
        raise HTTPException(
            status_code=503,
            detail="LASSO inference engine is not loaded."
        )

    missing_features = [
        feature
        for feature in lasso_engine.feature_names
        if feature not in request.features
        or request.features[feature] is None
    ]

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Some required LASSO features are missing.",
                "missing_features": missing_features,
            }
        )

    return build_lasso_response(
        patient_id=request.patientId,
        features=request.features,
        user_id=request.userId,
    )



# ---------------------------------------------------------------------
# LOCAL EXECUTION
# ---------------------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api_lasso:app",
        host="0.0.0.0",
        port=8001,
        reload=False
    )
