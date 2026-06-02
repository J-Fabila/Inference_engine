import numpy as np
from typing import List, Tuple, Optional, Dict

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def _tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def _sigmoid_grad(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s * (1.0 - s)

def _softmax(x: np.ndarray) -> np.ndarray:
    """Softmax numericamente estable, fila a fila."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

_ACTIVATIONS: Dict[str, Tuple] = {
    "relu":    (_relu,    _relu_grad),
    "tanh":    (_tanh,    _tanh_grad),
    "sigmoid": (_sigmoid, _sigmoid_grad),
}


def _to_numpy(x) -> np.ndarray:
    """Convierte tensores PyTorch o arrays numpy a numpy float64."""
    if hasattr(x, "detach"):          # es un tensor PyTorch
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)

class MCDropoutMLP:
    def __init__(
        self,
        n_feats: int,
        n_out: int,
        dropout_p: float = 0.3,
        hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        task: str = "classification",
        seed: int = 42,
    ):
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        assert 0.0 <= dropout_p < 1.0, "dropout_p debe estar en [0, 1)"
        assert activation in _ACTIVATIONS, f"activation debe ser uno de {list(_ACTIVATIONS)}"

        self.n_feats    = n_feats
        self.n_out      = n_out
        self.dropout_p  = dropout_p
        self.task       = task
        self.seed       = seed
        self._training  = True   # analogo a model.train() / model.eval()

        self._act, self._act_grad = _ACTIVATIONS[activation]

        # Arquitectura completa: entrada -> ocultas -> salida
        layer_sizes = [n_feats] + hidden_sizes + [n_out]
        self._layer_sizes = layer_sizes
        self._n_layers    = len(layer_sizes) - 1

        np.random.seed(seed)
        self._init_weights()

    def _init_weights(self):
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(self._n_layers):
            fan_in  = self._layer_sizes[i]
            fan_out = self._layer_sizes[i + 1]
            # He para ReLU, Xavier generalizado para el resto
            if self._act is _relu:
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.W.append(np.random.randn(fan_in, fan_out) * scale)
            self.b.append(np.zeros(fan_out))

    def train(self):
        """Activa el modo entrenamiento (dropout estocastico)."""
        self._training = True

    def eval(self):
        """
        Activa modo evaluacion.
        NOTA: en MC Dropout el dropout permanece ACTIVO durante eval cuando
        se llama a predict_proba_mc / predictive_entropy (mc=True).
        El flag _training solo desactiva el dropout en __call__ normal.
        """
        self._training = False

    def _forward(self, X: np.ndarray, mc: bool = False) -> np.ndarray:
        """
        Pasa X por la red y devuelve LOGITS crudos (sin softmax/sigmoid final).

        Parametros
        ----------
        X  : (N, n_feats)
        mc : si True -> aplica dropout aunque _training sea False.
             Usado en las T pasadas Monte Carlo durante eval.

        Retorna
        -------
        logits : (N, n_out)
        """
        use_dropout = self._training or mc

        h = X
        self._cache = {"h": [h], "z": [], "masks": []}

        for i in range(self._n_layers):
            z = h @ self.W[i] + self.b[i]
            self._cache["z"].append(z)

            is_last = (i == self._n_layers - 1)

            if is_last:
                h = z   # logits crudos; softmax/sigmoid se aplica fuera
            else:
                h = self._act(z)
                if use_dropout and self.dropout_p > 0.0:
                    mask = (
                        np.random.rand(*h.shape) > self.dropout_p
                    ).astype(np.float64)
                    mask /= (1.0 - self.dropout_p)   # inverted dropout
                    h = h * mask
                    self._cache["masks"].append(mask)
                else:
                    self._cache["masks"].append(None)

            self._cache["h"].append(h)

        return h   # (N, n_out) logits

    def __call__(self, x) -> np.ndarray:
        """
        Interfaz publica: model(x) -> logits (N, n_out).
        Acepta numpy arrays o tensores PyTorch.
        El dropout depende de self._training (train/eval mode).
        """
        X = _to_numpy(x)
        return self._forward(X, mc=False)

    def predict_proba_mc(
        self,
        x,
        T: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        T pasadas con dropout activo -> distribucion de probabilidades.

        Replica exactamente el loop de BasicNN:
            for _ in range(T):
                logits = self(x)
                probs.append(F.softmax(logits, dim=-1))
            probs = torch.stack(probs, dim=0)   # [T, B, C]

        Parametros
        ----------
        x : array-like (N, n_feats) o tensor PyTorch
        T : numero de muestras Monte Carlo

        Retorna
        -------
        mean : np.ndarray (N, n_out)   — media de las T probabilidades
        var  : np.ndarray (N, n_out)   — varianza de las T probabilidades
        """
        X = _to_numpy(x)
        all_probs: List[np.ndarray] = []

        for _ in range(T):
            logits = self._forward(X, mc=True)   # dropout activo siempre
            if self.n_out > 1:
                probs = _softmax(logits)          # (N, C)
            else:
                probs = _sigmoid(logits)          # (N, 1)
            all_probs.append(probs)

        stack = np.stack(all_probs, axis=0)   # (T, N, n_out)
        mean  = stack.mean(axis=0)            # (N, n_out)
        var   = stack.var(axis=0)             # (N, n_out)
        return mean, var

    def predictive_entropy(self, x, T: int = 20) -> np.ndarray:
        """
        Entropia predictiva H[y|x] = -sum( p_bar * log(p_bar) )
        donde p_bar = E_w[p(y|x,w)]  (media sobre las T muestras MC).

        Parametros
        ----------
        x : array-like (N, n_feats) o tensor PyTorch
        T : numero de muestras Monte Carlo

        Retorna
        -------
        entropy : np.ndarray (N,)
        """
        mean, _ = self.predict_proba_mc(x, T=T)   # (N, n_out)
        eps     = 1e-8
        entropy = -(mean * np.log(mean + eps)).sum(axis=-1)   # (N,)
        return entropy

    def get_weights(self) -> List[np.ndarray]:
        """
        Devuelve los pesos como lista plana [W0, b0, W1, b1, ...].
        Reemplaza a get_parameters() del cliente Flower original.
        """
        weights = []
        for W, b in zip(self.W, self.b):
            weights.append(W.copy())
            weights.append(b.copy())
        return weights

    def set_weights(self, weights: List[np.ndarray]):
        """
        Carga pesos desde lista plana [W0, b0, W1, b1, ...].
        Reemplaza a set_parameters() del cliente Flower original.
        """
        assert len(weights) == 2 * self._n_layers, (
            f"Se esperan {2 * self._n_layers} arrays, llegaron {len(weights)}"
        )
        for i in range(self._n_layers):
            self.W[i] = np.asarray(weights[2 * i],     dtype=np.float64)
            self.b[i] = np.asarray(weights[2 * i + 1], dtype=np.float64)


    def _compute_loss_and_delta(
        self, logits: np.ndarray, y: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Calcula perdida y delta inicial para backprop.

        Tarea            Loss                  Delta
        ─────────────────────────────────────────────────────────────
        regression       MSE                   2*(logits-y)/N
        classification   BCE binaria (n_out=1) sigmoid(logits)-y / N
        classification   CE multiclase         softmax(logits)-y_oh / N
        """
        N   = y.shape[0]
        eps = 1e-8

        if self.task == "regression":
            loss  = float(((logits - y) ** 2).mean())
            delta = 2.0 * (logits - y) / N

        else:  # classification
            if self.n_out == 1:
                # BCE con logits (numericamente estable)
                probs = _sigmoid(logits)                       # (N, 1)
                loss  = float(
                    -(y * np.log(probs + eps) +
                      (1 - y) * np.log(1 - probs + eps)).mean()
                )
                delta = (probs - y) / N                        # dBCE/dlogit

            else:
                # Cross-entropy multiclase con softmax
                probs = _softmax(logits)                       # (N, C)
                # y puede ser enteros (N,) o one-hot (N, C)
                if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
                    y_idx = y.astype(int).ravel()
                    y_oh  = np.zeros_like(probs)
                    y_oh[np.arange(N), y_idx] = 1.0
                else:
                    y_oh = y
                loss  = float(-(y_oh * np.log(probs + eps)).sum(axis=1).mean())
                delta = (probs - y_oh) / N                     # dCE/dlogit

        return loss, delta

    def _backprop(self, delta: np.ndarray, lr: float):
        for i in reversed(range(self._n_layers)):
            h_prev = self._cache["h"][i]
            dW = h_prev.T @ delta
            db = delta.sum(axis=0)

            self.W[i] -= lr * dW
            self.b[i] -= lr * db

            if i > 0:
                delta = delta @ self.W[i].T
                delta = delta * self._act_grad(self._cache["z"][i - 1])
                mask  = self._cache["masks"][i - 1]
                if mask is not None:
                    delta = delta * mask

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        verbose: bool = True,
        val_data: Optional[Tuple] = None,
    ) -> Dict[str, List[float]]:
        """
        Entrenamiento con mini-batches y dropout activo.

        Parametros
        ----------
        X, y       : datos de entrenamiento (numpy arrays o tensores)
        epochs     : epocas de entrenamiento
        lr         : learning rate
        batch_size : tamano del mini-lote
        verbose    : imprime loss cada 10 epocas
        val_data   : (X_val, y_val) opcional

        Retorna
        -------
        history : dict con listas 'train_loss' y 'val_loss'
        """
        X = _to_numpy(X)
        y = _to_numpy(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        N = X.shape[0]
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        self.train()

        for epoch in range(epochs):
            idx        = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N, batch_size):
                bi  = idx[start:start + batch_size]
                Xb  = X[bi]
                yb  = y[bi]

                logits      = self._forward(Xb, mc=False)   # _training=True -> dropout activo
                loss, delta = self._compute_loss_and_delta(logits, yb)
                self._backprop(delta, lr)

                epoch_loss += loss
                n_batches  += 1

            avg_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_loss)

            if val_data is not None:
                Xv = _to_numpy(val_data[0])
                yv = _to_numpy(val_data[1])
                if yv.ndim == 1:
                    yv = yv.reshape(-1, 1)
                self.eval()
                logits_v    = self._forward(Xv, mc=False)
                val_loss, _ = self._compute_loss_and_delta(logits_v, yv)
                history["val_loss"].append(val_loss)
                self.train()

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                msg = f"Epoca {epoch+1:>4}/{epochs}  train_loss: {avg_loss:.6f}"
                if val_data is not None:
                    msg += f"  val_loss: {history['val_loss'][-1]:.6f}"
                print(msg)

        return history

def uncertainty_metrics(
    model: MCDropoutMLP,
    val_loader,           # iterable de (x, y); x/y pueden ser tensores o arrays
    device: str = "cpu",  # ignorado (numpy siempre CPU); aceptado por compatibilidad
    T: int = 20,
) -> Dict[str, float]:
    """
    Calcula metricas de incertidumbre sobre val_loader usando MC Dropout.

    Replica uncertainty_metrics() del cliente PyTorch original:
        ent = model.predictive_entropy(x, T=T)
        mean, _ = model.predict_proba_mc(x, T=T)
        pred = mean.argmax(dim=-1)

    Parametros
    ----------
    model      : MCDropoutMLP entrenado
    val_loader : iterable de batches (x, y)
    device     : ignorado, solo por compatibilidad de firma
    T          : muestras Monte Carlo

    Retorna
    -------
    dict con:
        accuracy      : fraccion de predicciones correctas
        mean_entropy  : entropia predictiva media (mayor -> mas incierto)
        mean_variance : varianza MC media
        epistemic_unc : incertidumbre epistemica media
        aleatoric_unc : incertidumbre aleatoria media (proxy via varianza)
    """
    model.eval()   # __call__ sin dropout; predict_proba_mc lo reactiva internamente

    all_ents : List[np.ndarray] = []
    all_var  : List[np.ndarray] = []
    all_pred : List[np.ndarray] = []
    all_y    : List[np.ndarray] = []

    for x_batch, y_batch in val_loader:
        x_np = _to_numpy(x_batch)
        y_np = _to_numpy(y_batch).ravel().astype(int)

        # entropia predictiva: H[y|x] = -sum(p_bar * log p_bar)
        ent = model.predictive_entropy(x_np, T=T)        # (N,)

        # probabilidades MC: (mean, var)
        mean, var = model.predict_proba_mc(x_np, T=T)    # (N, C)

        # prediccion: argmax de la media
        if model.n_out > 1:
            pred = mean.argmax(axis=-1)
        else:
            pred = (mean[:, 0] > 0.5).astype(int)

        all_ents.append(ent)
        all_var.append(var)
        all_pred.append(pred)
        all_y.append(y_np)

    ents  = np.concatenate(all_ents,  axis=0)   # (N_total,)
    vars_ = np.concatenate(all_var,   axis=0)   # (N_total, C)
    preds = np.concatenate(all_pred,  axis=0)   # (N_total,)
    ys    = np.concatenate(all_y,     axis=0)   # (N_total,)

    # Descomposicion epistemica / aleatoria (Gal & Ghahramani 2016)
    # Epistemica ~ H[E_w[p]] - E_w[H[p]]
    # Aleatoria  ~ E_w[H[p]]  (aproximada como varianza MC media como proxy ligero)
    aleatoric = float(vars_.mean())
    epistemic = float(max(ents.mean() - aleatoric, 0.0))

    model.train()

    return {
        "accuracy":      float((preds == ys).mean()),
        "mean_entropy":  float(ents.mean()),
        "mean_variance": float(vars_.mean()),
        "epistemic_unc": epistemic,
        "aleatoric_unc": aleatoric,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)

    N, F, C = 500, 10, 4
    X_all = np.random.randn(N, F)
    y_all = (X_all[:, 0] + X_all[:, 1] > 0).astype(int) * 2 + (X_all[:, 2] > 0).astype(int)
    y_all = y_all % C

    split        = int(0.8 * N)
    X_tr, X_val  = X_all[:split], X_all[split:]
    y_tr, y_val  = y_all[:split], y_all[split:]

    model = MCDropoutMLP(
        n_feats      = F,
        n_out        = C,
        dropout_p    = 0.3,
        hidden_sizes = [128, 64],
        activation   = "relu",
        task         = "classification",
    )

    print("── Entrenamiento ──────────────────────────────────────")
    model.fit(X_tr, y_tr, epochs=60, lr=3e-3, batch_size=32, verbose=True)

    print("\n── predict_proba_mc ────────────────────────────────────")
    mean, var = model.predict_proba_mc(X_val[:5], T=50)
    print("mean shape:", mean.shape, "  var shape:", var.shape)
    print("mean (5 muestras):\n", mean.round(3))

    print("\n── predictive_entropy ──────────────────────────────────")
    ent = model.predictive_entropy(X_val[:5], T=50)
    print("entropy (5 muestras):", ent.round(4))

    print("\n── uncertainty_metrics con loader sintetico ────────────")
    bs     = 32
    loader = [
        (X_val[i:i+bs], y_val[i:i+bs])
        for i in range(0, len(X_val), bs)
    ]
    metrics = uncertainty_metrics(model, loader, device="cpu", T=30)
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")

    print("\n── get_weights / set_weights ───────────────────────────")
    w = model.get_weights()
    print(f"  {len(w)} arrays: " + ", ".join(str(wi.shape) for wi in w))
    model.set_weights(w)
    print("  set_weights OK")
