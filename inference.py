import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append("/home/jorge/workdir/flcore-suite")

import re

def load_model(model_dir,model,task):

    model_dir = Path(model_dir)

    def get_round(filepath):
        match = re.search(r'_round_(\d+)', filepath.name)
        return int(match.group(1)) if match else -1

    if model and task:
        model_name = f"{model}_{task}"
        metadata_files = list(model_dir.glob(f"{model_name}*_model_metadata.json"))
        if not metadata_files:
            metadata_path = model_dir / f"{model_name}_model_metadata.json"
            if metadata_path.exists():
                metadata_files = [metadata_path]
    else:
        metadata_files = list(model_dir.glob("*_model_metadata.json"))

    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found in {model_dir}")

    metadata_files.sort(key=get_round, reverse=True)
    metadata_file = metadata_files[0]

    with open(metadata_file) as f:
        metadata = json.load(f)

    model_type = metadata.get("model_type", "").lower()

    if model and task:
        model_files = (
            list(model_dir.glob(f"{model_name}*_model.joblib")) +
            list(model_dir.glob(f"{model_name}*_model.pkl")) +
            list(model_dir.glob(f"{model_name}*_model.npz")) +
            list(model_dir.glob(f"{model_name}*_model.json"))
        )
        if not model_files:
            for ext in [".joblib", ".pkl", ".npz", ".json"]:
                exact_path = model_dir / f"{model_name}_model{ext}"
                if exact_path.exists():
                    model_files.append(exact_path)
    else:
        model_files = (
            list(model_dir.glob("*_model.joblib")) + 
            list(model_dir.glob("*_model.pkl")) + 
            list(model_dir.glob("*_model.npz")) +
            list(model_dir.glob("*_model.json"))
        )
        
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    model_files.sort(key=get_round, reverse=True)
    model_file = model_files[0]

    if model_type == "cox":
        from flcore.models.cox.model import CoxPHModel
        model = CoxPHModel()
        model.load_model(model_file)
    elif model_type == "rsf":
        from flcore.models.rsf.model import RSFModel
        model = RSFModel()
        model.load_model(model_file)
    elif model_type == "gbs":
        from flcore.models.gbs.model import GBSModel
        model = GBSModel()
        model.load_model(model_file)
    elif model_type == "nn":
        from flcore.models.nn.mc_dropout_mlp import MCDropoutMLP
        n_feats = metadata.get("n_feats")
        n_out = metadata.get("n_out")
        task = metadata.get("task", "classification")
        base_model = MCDropoutMLP(n_feats=n_feats, n_out=n_out, task=task)
        data = np.load(model_file)
        weights = [data[k] for k in sorted(data.files)]
        base_model.set_weights(weights)

        class NnWrapper:
            def __init__(self, m): self.m = m
            def predict(self, X):
                logits = self.m(X.values if hasattr(X, "values") else X)
                if self.m.task == "regression": return logits
                if self.m.n_out == 1:
                    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
                    return (probs[:, 0] > 0.5).astype(int)
                return logits.argmax(axis=1)
        model = NnWrapper(base_model)
    elif model_type in ["linear_models", "logistic_regression", "random_forest", "weighted_random_forest", "xgb"]:
        model = joblib.load(model_file)
    else:
        model = joblib.load(model_file)

    return model, metadata

class InferenceEngine:
    def __init__(self, model, metadata, normalization_method="IQR"):
        self.model = model
        self.metadata = metadata
        self.normalization_method = normalization_method

        self.features_meta = metadata["features_meta"]
        self.outcomes_meta = metadata["outcomes_meta"]

        self.feature_names = metadata["feature_names"]
        self.target_names = metadata["target_names"]

        self.boolean_map = {False: 0, True: 1, "False": 0, "True": 1}

    def preprocess(self, df):

        dat = df.copy()

        for name, feat in self.features_meta.items():

            dtype = feat["dataType"]
            stats = feat.get("stats", {})

            if name not in dat.columns:
                continue

            if dtype == "NUMERIC":

                if self.normalization_method == "IQR":

                    q1 = stats.get("q1")
                    q2 = stats.get("q2")
                    q3 = stats.get("q3")

                    if None not in (q1, q2, q3):
                        dat[name] = (dat[name] - q2) / (q3 - q1)

                elif self.normalization_method == "MIN_MAX":

                    mini = stats.get("min")
                    maxi = stats.get("max")

                    if None not in (mini, maxi):
                        dat[name] = (dat[name] - mini) / (maxi - mini)

            elif dtype == "NOMINAL":

                value_set = stats.get("valueSet", [])

                if len(value_set) > 0:
                    cat_map = {cat: i for i, cat in enumerate(value_set)}
                    dat[name] = dat[name].map(cat_map)

            elif dtype == "BOOLEAN":

                dat[name] = dat[name].map(self.boolean_map)

        return dat[self.feature_names]

    def predict(self, df):

        X = self.preprocess(df)

        if hasattr(self.model, "predict_risk"):
            preds = self.model.predict_risk(X)
        else:
            preds = self.model.predict(X)

        target = self.target_names[0] if len(self.target_names) > 0 else None
        target_meta = self.outcomes_meta.get(target, {}) if target else {}

        dtype = target_meta.get("dataType", None)

        if dtype == "NOMINAL":

            value_set = target_meta.get("stats", {}).get("valueSet", [])

            if len(value_set) > 0:
                inv_map = {i: cat for i, cat in enumerate(value_set)}
                preds = [inv_map.get(p, p) for p in preds]

        elif dtype == "BOOLEAN":

            preds = [bool(p) for p in preds]

        elif dtype == "NUMERIC":

            stats = target_meta.get("stats", {})

            if self.normalization_method == "IQR":
                q1 = stats.get("q1")
                q2 = stats.get("q2")
                q3 = stats.get("q3")

                if None not in (q1, q2, q3):
                    preds = [p * (q3 - q1) + q2 for p in preds]

            elif self.normalization_method == "MIN_MAX":
                mini = stats.get("min")
                maxi = stats.get("max")

                if None not in (mini, maxi):
                    preds = [p * (maxi - mini) + mini for p in preds]

        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        elif not isinstance(preds, list):
            preds = list(preds)

        return preds

    def explain(self, df):
        """
        Runs explainability using SHAP via the model's `explain` method.
        Only runs if the underlying model implements this method.
        """
        if hasattr(self.model, "explain"):
            X = self.preprocess(df)
            return self.model.explain(X)
        return None

#**** * * * * * *  *  *   *   *     *  *  * * * * *******  INPUT
model_path = "/home/jorge/workdir/Inference_engine/sandbox/experiment_1/models"
# test model and task names explicitly so there is no NameError 
test_model = "cox"
test_task = "survival"
print(test_model,test_task)

model, metadata = load_model(model_path, model=test_model, task=test_task)
new_data_path = "/home/jorge/workdir/flcore-suite/dataset/bucarest_sintetico/synthetic_dt4h_dataset.csv"
#new_data_path = "/home/jorge/workdir/flcore-suite/dataset/bucarest_sintetico/synthetic_survival.csv"
#**** * * * * * *  *  *   *   *     *  *  * * * * *******  INPUT
engine = InferenceEngine(model, metadata)

df_new = pd.read_csv(new_data_path)

predictions = engine.predict(df_new)
print("Predictions:", predictions)

explanations = engine.explain(df_new)
if explanations is not None:
    print("Explanations:", explanations)
