import json
import math
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timezone
import uuid

import numpy as np
import shap


class LassoInferenceEngine:

    def __init__(self, model_path=None):

        if model_path is None:
            model_path = Path(__file__).parent / "lasso_model.json"

        with open(model_path, "r", encoding="utf-8") as f:
            self.model = json.load(f)

        self.intercept = self.model["intercept"]
        self.coefficients = self.model["coefficients"]
        self.standardization = self.model["standardization_parameters"]
        self.outcome_name = self.model["outcome_name"]

        self.feature_names = list(self.coefficients.keys())

        # Temporary SHAP background: 10 example observations
        samples = self.model["sample_predictions"]

        background = []
        for sample in samples.values():
            background.append([sample[f] for f in self.feature_names])

        self.background = np.array(background, dtype=float)
        self.linear_model = LogisticRegression()

        self.linear_model.coef_ = np.array([
            [self.coefficients[f] for f in self.feature_names]
        ])

        self.linear_model.intercept_ = np.array([self.intercept])
        self.linear_model.classes_ = np.array([0, 1])
        self.linear_model.n_features_in_ = len(self.feature_names)

        self.explainer = shap.LinearExplainer(
            self.linear_model,
            self.background
        )

        print("LASSO inference engine initialized successfully!")
        print(f"Outcome: {self.outcome_name}")
        print(f"Number of features: {len(self.coefficients)}")

    def standardize_features(self, features):

        standardized = {}

        for feature in self.feature_names:

            if feature not in features or features[feature] is None:
                raise ValueError(f"Missing feature: {feature}")

            value = float(features[feature])

            if not math.isfinite(value):
                raise ValueError(f"Invalid value: {feature}")

            mean = self.standardization[feature]["mean"]
            sd = self.standardization[feature]["sd"]

            if sd == 0:
                raise ValueError(f"Standard deviation is zero: {feature}")

            standardized[feature] = (value - mean) / sd

        return standardized

    def predict(self, features):

        standardized = self.standardize_features(features)

        z = self.intercept

        for feature, coefficient in self.coefficients.items():
            z += coefficient * standardized[feature]

        if z >= 0:
            probability = 1 / (1 + math.exp(-z))
        else:
            exp_z = math.exp(z)
            probability = exp_z / (1 + exp_z)

        return {
            "outcome": self.outcome_name,
            "prediction": probability,
            "prediction_percent": probability * 100,
            "prediction_horizon_months": 12
        }

    def explain(self, features):

        standardized = self.standardize_features(features)

        values = np.array(
            [[standardized[f] for f in self.feature_names]],
            dtype=float
        )

        shap_values = self.explainer(values)

        return [
            {
                "name": feature,
                "value": float(value)
            }
            for feature, value in zip(
                self.feature_names,
                shap_values.values[0]
            )
        ]

    def build_panos_response(
        self,
        features,
        patient_id,
        user_id=None,
        model_id="care-heart-inpatient",
        model_name="care-heart-inpatient"
    ):

        prediction = self.predict(features)
        shap_data = self.explain(features)

        response = {
            "care-heart-inpatient-fs": {
                "prediction_id": str(uuid.uuid4()),
                "user_id": user_id,
                "patient_id": patient_id,
                "model_id": model_id,
                "model_name": model_name,
                "@timestamp": datetime.now(timezone.utc).isoformat(),

                "explanations": [
                    {
                        "horizon": "all_cause_mortality_1y",
                        "score": f"{prediction['prediction']:.6f}",

                        "scoreCI": None,

                        "shap_data": shap_data,

                        "contribution_data": None,

                        "whatever_data": [],

                        "distribution_data": None,

                        "percentile": None,
                        "mean": None,
                        "median": None
                    }
                ]
            }
        }

        return response
if __name__ == "__main__":

    engine = LassoInferenceEngine()

    samples = engine.model["sample_predictions"]

    print("\n--- TEST ALL 10 SAMPLE PREDICTIONS AND SHAP ---\n")

    passed = 0

    for sample_name, sample in samples.items():

        raw_features = {}

        for feature in engine.feature_names:
            mean = engine.standardization[feature]["mean"]
            sd = engine.standardization[feature]["sd"]
            raw_features[feature] = sample[feature] * sd + mean

        result = engine.predict(raw_features)
        explanations = engine.explain(raw_features)

        # Verify SHAP additivity
        standardized = engine.standardize_features(raw_features)

        input_values = np.array([[
            standardized[f] for f in engine.feature_names
        ]])

        shap_result = engine.explainer(input_values)

        shap_sum = float(np.sum(shap_result.values[0]))
        base_value = float(np.array(shap_result.base_values).flatten()[0])

        reconstructed_logit = base_value + shap_sum

        expected_logit = engine.intercept

        for feature in engine.feature_names:
            expected_logit += (
                engine.coefficients[feature] * standardized[feature]
            )

        shap_difference = abs(reconstructed_logit - expected_logit)
        shap_passed = shap_difference < 1e-8

        difference = abs(sample["pred_prob"] - result["prediction"])
        test_passed = (
            difference < 1e-9
            and len(explanations) == len(engine.feature_names)
            and shap_passed
        )

        if test_passed:
            passed += 1

        print(f"{sample_name}: {'PASS' if test_passed else 'FAIL'}")
        print(f"  Prediction: {result['prediction']:.8f}")
        print(f"  SHAP features: {len(explanations)}")

    # Test Panos JSON response
    response = engine.build_panos_response(
        features=raw_features,
        patient_id="test-patient-001"
    )

    print("\n--- PANOS JSON RESPONSE TEST ---\n")
    print(json.dumps(response, indent=4))

    assert "care-heart-inpatient-fs" in response
    explanation = response["care-heart-inpatient-fs"]["explanations"][0]

    assert explanation["horizon"] == "all_cause_mortality_1y"
    assert len(explanation["shap_data"]) == 14

    print("\nPANOS JSON STRUCTURE: PASS")
    
    print(f"\nRESULT: {passed}/10 TESTS PASSED")