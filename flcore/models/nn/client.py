# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Uncertainty-Aware Neural Network
# Author: Jorge Fabila Fabian
# Fecha: September 2025
# Project: DT4H
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
import json
import numpy as np
import flwr as fl

from pathlib import Path
from typing  import List, Dict, Tuple, Optional

from flcore.metrics import calculate_metrics
from flcore.models.nn.mc_dropout_mlp import MCDropoutMLP, uncertainty_metrics

def _to_numpy(x) -> np.ndarray:
    """Convierte tensores PyTorch, DataFrames o arrays a numpy float64."""
    if hasattr(x, "detach"):          # tensor PyTorch
        return x.detach().cpu().numpy().astype(np.float64)
    if hasattr(x, "values"):          # pandas DataFrame / Series
        return x.values.astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    """
    Genera una lista de batches (X_batch, y_batch) en numpy.
    Reemplaza TensorDataset + DataLoader.
    """
    N   = X.shape[0]
    idx = np.random.permutation(N) if shuffle else np.arange(N)
    batches = []
    for start in range(0, N, batch_size):
        bi = idx[start:start + batch_size]
        batches.append((X[bi], y[bi]))
    return batches


def _numpy_loss(logits: np.ndarray, y: np.ndarray, task: str, n_out: int, penalty: str = "l2") -> float:
    """
    Calcula la perdida en numpy. Misma logica que _compute_loss_and_delta
    pero solo devuelve el escalar (para el bucle de evaluate).
    """
    eps = 1e-8
    N   = y.shape[0]

    if task == "regression":
        if penalty == "l1":
            return float(np.abs(logits - y).mean())
        elif penalty in ("smooth", "smooth_l1", "smoothl1"):
            diff = np.abs(logits - y)
            return float(np.where(diff < 1, 0.5 * diff**2, diff - 0.5).mean())
        else:  # l2 / mse (default)
            return float(((logits - y) ** 2).mean())

    else:  # classification
        if n_out == 1:
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
            return float(
                -(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)).mean()
            )
        else:
            e     = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            y_idx = y.astype(int).ravel()
            return float(-np.log(probs[np.arange(N), y_idx] + eps).mean())


def _numpy_preds(logits: np.ndarray, task: str, n_out: int) -> np.ndarray:
    """Predicciones discretas a partir de logits."""
    if task == "regression":
        return logits
    if n_out == 1:
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        return (probs[:, 0] > 0.5).astype(int)
    return logits.argmax(axis=1)


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, config: Dict, data: Tuple):
        self.config     = config
        self.batch_size = config["batch_size"]
        self.lr         = config["lr"]
        self.epochs     = config["local_epochs"]
        self.device     = "cpu"   # numpy no usa device; se conserva por retrocompatibilidad

        (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = data

        self.X_train = _to_numpy(X_train_raw)   # (N, F)
        self.X_test  = _to_numpy(X_test_raw)

        self.y_train = _to_numpy(y_train_raw)   # (N,) o (N, 1)
        self.y_test  = _to_numpy(y_test_raw)

        # Asegurar shape (N,) para etiquetas
        if self.y_train.ndim > 1 and self.y_train.shape[1] == 1:
            self.y_train = self.y_train.ravel()
        if self.y_test.ndim > 1 and self.y_test.shape[1] == 1:
            self.y_test = self.y_test.ravel()

        # ── Loaders numpy ──────────────────────────────────────────────────
        # Equivalen a TensorDataset + DataLoader
        self._rebuild_loaders()

        # ── Modelo ────────────────────────────────────────────────────────
        self.model = MCDropoutMLP(
            n_feats      = config["n_feats"],
            n_out        = config["n_out"],
            dropout_p    = config["dropout_p"],
            hidden_sizes = config.get("hidden_sizes", [128, 64]),
            activation   = config.get("activation", "relu"),
            task         = config["task"],
        )

        self.round = 0

    def _rebuild_loaders(self):
        """Reconstruye los loaders numpy (llamar tras cambios en datos)."""
        self.train_loader = _make_loader(self.X_train, self.y_train, self.batch_size, shuffle=True)
        self.test_loader  = _make_loader(self.X_test,  self.y_test,  self.batch_size, shuffle=False)
        self.val_loader   = _make_loader(self.X_test,  self.y_test,  self.batch_size, shuffle=False)

    # ── Flower: intercambio de parametros ─────────────────────────────────────

    def get_parameters(self, config) -> List[np.ndarray]:
        # Antes: [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return self.model.get_weights()

    def set_parameters(self, parameters: List[np.ndarray]):
        self.model.train()
        # Antes: load_state_dict con OrderedDict de tensores
        self.model.set_weights(parameters)

    # ── Flower: entrenamiento local ───────────────────────────────────────────

    def fit(self, parameters, params):
        self.set_parameters(parameters)

        # Reconstruimos loaders con shuffle fresco cada ronda
        self._rebuild_loaders()

        # ── Loop de entrenamiento — numpy puro ────────────────────────────
        # Antes: loop manual con optimizer.zero_grad / loss.backward / step
        # Ahora: model.fit() lo gestiona todo internamente
        penalty = self.config.get("penalty", "l2")

        history = self.model.fit(
            X          = self.X_train,
            y          = self.y_train,
            epochs     = self.epochs,
            lr         = self.lr,
            batch_size = self.batch_size,
            verbose    = True,
        )

        # Imprime resumen por epoca en el mismo formato que el original
        for i, loss_val in enumerate(history["train_loss"]):
            print(f"Epoch {i+1:02d} | Train Loss: {loss_val:.4f}")

        dataset_len = self.y_train.shape[0]

        if self.round % self.config["save_every_n_rounds"] == 0:
            self.save_model()

        self.round += 1
        return self.get_parameters(config={}), dataset_len, {}

    # ── Flower: evaluacion ────────────────────────────────────────────────────

    def evaluate(self, parameters, params):
        self.set_parameters(parameters)

        self.model.eval()

        # ── Metricas de incertidumbre MC ──────────────────────────────────
        if self.config["dropout_p"] > 0.0:
            metrics = uncertainty_metrics(
                model      = self.model,
                val_loader = self.val_loader,   # lista de batches numpy
                device     = self.device,        # ignorado internamente, por compatibilidad
                T          = int(self.config["T"]),
            )
        else:
            # Sin dropout: prediccion determinista
            # Antes: pred = self.model(self.X_test); y_pred = pred[:,0]
            logits = self.model(self.X_test)     # (N, n_out) numpy
            y_pred = logits[:, 0]
            metrics = calculate_metrics(self.y_test, y_pred, self.config)

        # ── Loop de perdida en test ───────────────────────────────────────
        # Antes: loop PyTorch con F.binary_cross_entropy_with_logits / F.cross_entropy
        total_loss = 0.0
        correct    = 0
        total      = 0
        penalty    = self.config.get("penalty", "l2")

        for X_batch, y_batch in self.test_loader:
            # X_batch, y_batch ya son numpy (no hace falta .to(device))
            logits = self.model(X_batch)         # (B, n_out)

            loss = _numpy_loss(logits, y_batch.reshape(-1, 1) if self.config["n_out"] == 1
                               else y_batch, self.config["task"], self.config["n_out"], penalty)

            preds = _numpy_preds(logits, self.config["task"], self.config["n_out"])

            if self.config["task"] == "classification":
                y_int   = y_batch.astype(int).ravel()
                correct += int((preds == y_int).sum())

            total_loss += loss * X_batch.shape[0]
            total      += X_batch.shape[0]

        test_loss   = total_loss / total
        dataset_len = self.y_test.shape[0]

        return float(test_loss), dataset_len, metrics

    # ── Guardado del modelo ───────────────────────────────────────────────────

    def save_model(self):
        save_path = Path(self.config["experiment_dir"]) / "models"
        save_path.mkdir(parents=True, exist_ok=True)

        model_name = (
            f"{self.config['model']}_{self.config['task']}"
            f"_round_{getattr(self, 'round', 0)}"
        )

        # Antes: torch.save(self.model.state_dict(), model_path) → archivo .pt
        # Ahora: numpy .npz (portable, sin dependencias)
        model_path = save_path / f"{model_name}_model.npz"
        weights    = self.model.get_weights()
        np.savez(model_path, *weights)

        # ── Metadata (sin cambios respecto al original) ───────────────────
        with open(self.config["metadata_file"], "r") as f:
            data_metadata = json.load(f)

        entity = data_metadata.get("entries", [])[0]

        features_list = entity.get("features", [])
        outcomes_list = entity.get("outcomes", [])
        dataset_stats = entity.get("datasetStats", {})
        feature_stats = dataset_stats.get("featureStats", {})
        outcome_stats = dataset_stats.get("outcomeStats", {})

        all_features_meta = {feat["name"]: feat for feat in features_list}
        all_outcomes_meta = {out["name"]:  out  for out  in outcomes_list}

        for f_name, f_meta in all_features_meta.items():
            f_meta["stats"] = feature_stats.get(f_name, {})
        for o_name, o_meta in all_outcomes_meta.items():
            o_meta["stats"] = outcome_stats.get(o_name, {})

        features_meta = {}
        for label in self.config["train_labels"]:
            if label in all_features_meta:
                features_meta[label] = all_features_meta[label]
            elif label in all_outcomes_meta:
                features_meta[label] = all_outcomes_meta[label]

        outcomes_meta = {}
        for label in self.config["target_labels"]:
            if label in all_outcomes_meta:
                outcomes_meta[label] = all_outcomes_meta[label]
            elif label in all_features_meta:
                outcomes_meta[label] = all_features_meta[label]

        metadata = {
            "node_name":     self.config["node_name"],
            "task":          self.config["task"],
            "n_out":         self.config["n_out"],
            "n_feats":       self.config["n_feats"],
            "model_type":    self.config["model"],
            "feature_names": self.config["train_labels"],
            "target_names":  self.config["target_labels"],
            "metrics":       getattr(self, "last_metrics", None),
            "features_meta": features_meta,
            "outcomes_meta": outcomes_meta,
        }

        metadata_path = save_path / f"{model_name}_model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[Client] NN model saved at {model_path}")

    # ── Carga del modelo (bonus: simetrico con save_model) ────────────────────

    def load_model(self, round_n: int):
        """Carga pesos desde un .npz guardado previamente."""
        save_path  = Path(self.config["sandbox_path"]) / "model"
        model_name = f"{self.config['model']}_{self.config['task']}_round_{round_n}"
        model_path = save_path / f"{model_name}_model.npz"

        data    = np.load(model_path)
        weights = [data[k] for k in sorted(data.files)]   # arr_0, arr_1, ...
        self.model.set_weights(weights)
        print(f"[Client] NN model loaded from {model_path}")

        
def get_client(config,data) -> fl.client.Client:
#    client = FlowerClient(params).to_client()
    return FlowerClient(config,data)
#_______________________________________________________________________________________
