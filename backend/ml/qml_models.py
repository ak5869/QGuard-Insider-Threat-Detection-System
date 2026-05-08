"""
QuantumWatch — QML Detection Engine
=====================================
Two quantum models:

A) Variational Quantum Classifier (VQC)
   - Trainable quantum circuit with RY/RZ rotation gates and CNOT entanglement
   - Uses angle encoding to embed behavioral features as qubit rotations
   - Trained via gradient descent (Adam optimizer) on labeled data
   - Output: probability of insider threat (0 to 1)

B) Quantum Kernel SVM (QSVM)  
   - Uses a quantum feature map to compute kernel matrix
   - Classical SVM trained on the quantum kernel
   - Detects threats in quantum feature space
   - Output: decision score

Both run on classical hardware via PennyLane's default.qubit simulator.
Training uses 5,000 representative samples (quantum simulation is slow).
Inference applies to all employees using the trained models.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

import pennylane as qml
from pennylane import numpy as pnp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from backend.core.config import settings
from backend.ml.preprocessing import (
    clean_dataframe, engineer_features,
    build_classical_preprocessor, build_quantum_preprocessor,
    rescale_to_angles, save_preprocessor, load_preprocessor,
    NUMERIC_FEATURES, ENGINEERED_FEATURES, CATEGORICAL_FEATURES
)


# ── Quantum Circuit Definitions ───────────────────────────────────────────────

def get_device(n_qubits: int):
    """Create a PennyLane quantum device (simulator)."""
    return qml.device("default.qubit", wires=n_qubits)


def angle_encoding(x: np.ndarray, n_qubits: int):
    """
    Angle encoding: embed each feature as a rotation angle on a qubit.
    RY rotation encodes the feature value as a rotation around Y axis.
    """
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)


def entanglement_layer(n_qubits: int):
    """
    CNOT entanglement: connect adjacent qubits.
    This creates quantum correlations between features — the key
    advantage of quantum computing over classical methods.
    """
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    # Ring connection
    qml.CNOT(wires=[n_qubits - 1, 0])


def variational_layer(params: np.ndarray, n_qubits: int, layer_idx: int):
    """
    One variational layer: RY and RZ rotations with trainable parameters.
    params shape: (n_layers, n_qubits, 2) — 2 angles per qubit per layer
    """
    for i in range(n_qubits):
        qml.RY(params[layer_idx, i, 0], wires=i)
        qml.RZ(params[layer_idx, i, 1], wires=i)


def build_vqc_circuit(n_qubits: int, n_layers: int):
    """
    Build the Variational Quantum Classifier circuit.
    Architecture:
      1. Angle encode input features
      2. Repeat n_layers times: [variational rotations → entanglement]
      3. Measure expectation value of Pauli-Z on qubit 0
    """
    dev = get_device(n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, params):
        # Step 1: Encode input features
        angle_encoding(x, n_qubits)

        # Step 2: Variational layers
        for layer in range(n_layers):
            variational_layer(params, n_qubits, layer)
            entanglement_layer(n_qubits)

        # Step 3: Measure — returns expectation value in [-1, 1]
        return qml.expval(qml.PauliZ(0))

    return circuit


def build_quantum_kernel_circuit(n_qubits: int):
    """
    Quantum kernel: compute <x1|U†(x2)U(x1)|x1> = |<φ(x1)|φ(x2)>|²
    This is the inner product in quantum feature space.
    Used by QSVM to measure similarity between two data points.
    """
    dev = get_device(n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        # Encode x1
        angle_encoding(x1, n_qubits)
        # Apply inverse encoding of x2
        qml.adjoint(angle_encoding)(x2, n_qubits)
        # Measure probability of returning to |0⟩ state
        return qml.probs(wires=range(n_qubits))

    def kernel(x1, x2):
        """Quantum kernel value: similarity in quantum feature space."""
        probs = kernel_circuit(x1, x2)
        return float(probs[0])  # probability of |00...0⟩ state

    return kernel


# ── VQC Training ──────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_vqc(X_train: np.ndarray, y_train: np.ndarray,
              n_qubits: int, n_layers: int, epochs: int, lr: float) -> Tuple[np.ndarray, list]:
    """
    Train the Variational Quantum Classifier.
    Uses gradient descent with Adam optimizer.
    Loss: Binary cross-entropy on sigmoid of circuit output.
    """
    circuit = build_vqc_circuit(n_qubits, n_layers)

    # Initialize random parameters
    np.random.seed(42)
    params = pnp.array(
        np.random.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 2)),
        requires_grad=True
    )

    opt = qml.AdamOptimizer(stepsize=lr)
    loss_history = []

    # Convert labels to {-1, +1} for quantum circuit
    y_qml = np.where(y_train == 1, 1.0, -1.0)

    print(f"[VQC] Training on {len(X_train)} samples, {n_qubits} qubits, {n_layers} layers, {epochs} epochs")

    for epoch in range(epochs):
        # Mini-batch gradient descent
        batch_size = min(32, len(X_train))
        idx = np.random.choice(len(X_train), batch_size, replace=False)
        X_batch = X_train[idx]
        y_batch = y_qml[idx]

        def cost(params):
            predictions = pnp.array([circuit(x, params) for x in X_batch])
            # Binary cross-entropy
            probs = (predictions + 1) / 2  # convert [-1,1] to [0,1]
            probs = pnp.clip(probs, 1e-7, 1 - 1e-7)
            y_bin = (pnp.array(y_batch) + 1) / 2
            loss = -pnp.mean(y_bin * pnp.log(probs) + (1 - y_bin) * pnp.log(1 - probs))
            return loss

        params, loss_val = opt.step_and_cost(cost, params)
        loss_history.append(float(loss_val))

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {loss_val:.4f}")

    print(f"[VQC] Training complete. Final loss: {loss_history[-1]:.4f}")
    return np.array(params), loss_history


def vqc_predict_proba(X: np.ndarray, params: np.ndarray, n_qubits: int, n_layers: int) -> np.ndarray:
    """Get VQC probabilities for all samples."""
    circuit = build_vqc_circuit(n_qubits, n_layers)
    raw_outputs = np.array([float(circuit(x, params)) for x in X])
    # Convert from [-1, 1] to [0, 1]
    return (raw_outputs + 1) / 2


# ── QSVM Training ─────────────────────────────────────────────────────────────

def train_qsvm(X_train: np.ndarray, y_train: np.ndarray, n_qubits: int) -> SVC:
    """
    Train Quantum Kernel SVM.
    1. Compute quantum kernel matrix K[i,j] = |<φ(xi)|φ(xj)>|²
    2. Train classical SVM with precomputed kernel
    """
    print(f"[QSVM] Computing quantum kernel matrix for {len(X_train)} samples...")
    kernel_fn = build_quantum_kernel_circuit(n_qubits)

    n = len(X_train)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            k_val = kernel_fn(X_train[i], X_train[j])
            K[i, j] = k_val
            K[j, i] = k_val
        if (i + 1) % 100 == 0:
            print(f"  Kernel matrix: {i+1}/{n} rows computed")

    print(f"[QSVM] Kernel matrix computed. Training SVM...")
    svm = SVC(kernel="precomputed", probability=True, class_weight={0: 1, 1: 3}, random_state=42)
    svm.fit(K, y_train)

    # Save training data for inference kernel computation
    return svm, X_train


def qsvm_predict(X_test: np.ndarray, svm: SVC, X_train: np.ndarray, n_qubits: int) -> np.ndarray:
    """Compute quantum kernel between test and training data, then predict."""
    kernel_fn = build_quantum_kernel_circuit(n_qubits)

    print(f"[QSVM] Computing test kernel matrix ({len(X_test)} × {len(X_train)})...")
    K_test = np.zeros((len(X_test), len(X_train)))
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            K_test[i, j] = kernel_fn(X_test[i], X_train[j])
        if (i + 1) % 500 == 0:
            print(f"  Test kernel: {i+1}/{len(X_test)} rows computed")

    probs = svm.predict_proba(K_test)[:, 1]
    return probs


# ── Full Training Pipeline ────────────────────────────────────────────────────

def train_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Complete QML training pipeline:
    1. Preprocess data
    2. Subsample for quantum training (QML is slow)
    3. Train VQC on angle-encoded features
    4. Train QSVM on quantum kernel
    5. Evaluate both models
    6. Save all artifacts
    """
    n_qubits  = settings.N_QUBITS
    n_layers  = settings.N_LAYERS
    epochs    = settings.QML_EPOCHS
    lr        = settings.QML_LEARNING_RATE
    n_samples = settings.QML_TRAIN_SAMPLES

    # Step 1: Preprocess
    X_raw, y, _ = clean_dataframe(df)
    if y is None:
        raise ValueError("Dataset must contain 'is_malicious' column to train models.")

    X_eng = engineer_features(X_raw)
    y_arr = y.values
    print(f"[Train] {len(df)} rows | Malicious: {y_arr.sum()} ({y_arr.mean()*100:.1f}%)")

    # Step 2: Classical preprocessing (full dataset for scaler fitting)
    classical_pre = build_classical_preprocessor()
    X_classical = classical_pre.fit_transform(X_eng)
    save_preprocessor(classical_pre, "classical_preprocessor.joblib")

    # Step 3: Quantum preprocessing — PCA to n_qubits dimensions
    # Fit PCA on full dataset for best component extraction
    numeric_cols = NUMERIC_FEATURES + ENGINEERED_FEATURES
    X_numeric = X_eng[numeric_cols].values

    scaler_q = StandardScaler()
    X_numeric_scaled = scaler_q.fit_transform(X_numeric)

    pca = PCA(n_components=n_qubits, random_state=42)
    X_pca = pca.fit_transform(X_numeric_scaled)
    X_quantum = rescale_to_angles(X_pca)

    joblib.dump({"scaler": scaler_q, "pca": pca}, Path(settings.MODEL_DIR) / "quantum_preprocessor.joblib")
    print(f"[Train] PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Step 4: Subsample for quantum training
    # Stratified subsample preserving class ratio
    n_mal  = min(int(n_samples * y_arr.mean()), y_arr.sum())
    n_norm = min(n_samples - n_mal, (y_arr == 0).sum())

    mal_idx  = np.where(y_arr == 1)[0]
    norm_idx = np.where(y_arr == 0)[0]
    np.random.seed(42)
    sel_mal  = np.random.choice(mal_idx,  n_mal,  replace=False)
    sel_norm = np.random.choice(norm_idx, n_norm, replace=False)
    sel_idx  = np.concatenate([sel_mal, sel_norm])
    np.random.shuffle(sel_idx)

    X_q_train = X_quantum[sel_idx]
    y_q_train = y_arr[sel_idx]
    print(f"[Train] Quantum training subsample: {len(sel_idx)} records "
          f"(malicious: {y_q_train.sum()}, normal: {(y_q_train==0).sum()})")

    # SMOTE on subsample
    smote = SMOTE(random_state=42, k_neighbors=min(5, y_q_train.sum()-1))
    X_q_sm, y_q_sm = smote.fit_resample(X_q_train, y_q_train)
    print(f"[Train] After SMOTE: {len(X_q_sm)} samples")

    # Train/test split on subsample
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_q_sm, y_q_sm, test_size=0.2, random_state=42, stratify=y_q_sm
    )

    # ── A) Train VQC ──────────────────────────────────────────────────────────
    vqc_params, loss_history = train_vqc(X_tr, y_tr, n_qubits, n_layers, epochs, lr)
    joblib.dump({
        "params":   vqc_params,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "loss_history": loss_history
    }, Path(settings.MODEL_DIR) / "vqc.joblib")

    # Evaluate VQC
    vqc_probs_te = vqc_predict_proba(X_te, vqc_params, n_qubits, n_layers)
    vqc_thresh = 0.5
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.3, 0.7, 0.05):
        f = f1_score(y_te, (vqc_probs_te >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    vqc_thresh = best_t
    vqc_preds = (vqc_probs_te >= vqc_thresh).astype(int)

    vqc_metrics = {
        "accuracy": round(float(accuracy_score(y_te, vqc_preds)), 4),
        "f1":       round(float(f1_score(y_te, vqc_preds, zero_division=0)), 4),
        "roc_auc":  round(float(roc_auc_score(y_te, vqc_probs_te)), 4),
        "threshold": round(vqc_thresh, 2),
    }
    joblib.dump({"threshold": vqc_thresh}, Path(settings.MODEL_DIR) / "vqc_threshold.joblib")
    print(f"[VQC] Accuracy: {vqc_metrics['accuracy']:.4f} | F1: {vqc_metrics['f1']:.4f} | AUC: {vqc_metrics['roc_auc']:.4f}")

    # ── B) Train QSVM ────────────────────────────────────────────────────────
    # Use smaller subset for QSVM (kernel matrix is O(n²))
    qsvm_n = min(500, len(X_tr))
    idx_qsvm = np.random.choice(len(X_tr), qsvm_n, replace=False)
    X_qsvm_tr = X_tr[idx_qsvm]
    y_qsvm_tr = y_tr[idx_qsvm]

    svm_model, X_support = train_qsvm(X_qsvm_tr, y_qsvm_tr, n_qubits)
    joblib.dump({
        "svm": svm_model,
        "X_train": X_support,
        "n_qubits": n_qubits
    }, Path(settings.MODEL_DIR) / "qsvm.joblib")

    # Evaluate QSVM on test subset
    qsvm_n_te = min(100, len(X_te))
    idx_te = np.random.choice(len(X_te), qsvm_n_te, replace=False)
    qsvm_probs_te = qsvm_predict(X_te[idx_te], svm_model, X_support, n_qubits)
    qsvm_preds_te = (qsvm_probs_te >= 0.5).astype(int)

    qsvm_metrics = {
        "accuracy": round(float(accuracy_score(y_te[idx_te], qsvm_preds_te)), 4),
        "f1":       round(float(f1_score(y_te[idx_te], qsvm_preds_te, zero_division=0)), 4),
        "roc_auc":  round(float(roc_auc_score(y_te[idx_te], qsvm_probs_te)), 4),
    }
    print(f"[QSVM] Accuracy: {qsvm_metrics['accuracy']:.4f} | F1: {qsvm_metrics['f1']:.4f} | AUC: {qsvm_metrics['roc_auc']:.4f}")

    # Save normalization bounds for QSVM scores
    # Get QSVM scores on subsample for normalization
    qsvm_all_probs = qsvm_predict(X_q_train[:min(200, len(X_q_train))],
                                   svm_model, X_support, n_qubits)
    qsvm_p2  = float(np.percentile(qsvm_all_probs, 2))
    qsvm_p98 = float(np.percentile(qsvm_all_probs, 98))
    joblib.dump({"p2": qsvm_p2, "p98": qsvm_p98}, Path(settings.MODEL_DIR) / "qsvm_bounds.joblib")

    return {
        "vqc_accuracy":    vqc_metrics["accuracy"],
        "vqc_f1":          vqc_metrics["f1"],
        "vqc_roc_auc":     vqc_metrics["roc_auc"],
        "vqc_threshold":   vqc_metrics["threshold"],
        "qsvm_accuracy":   qsvm_metrics["accuracy"],
        "qsvm_f1":         qsvm_metrics["f1"],
        "qsvm_roc_auc":    qsvm_metrics["roc_auc"],
        "n_qubits":        n_qubits,
        "n_layers":        n_layers,
        "train_samples":   len(sel_idx),
        "epochs":          epochs,
        "pca_variance":    round(float(pca.explained_variance_ratio_.sum()), 4),
        "loss_final":      round(loss_history[-1], 4),
        "message":         f"VQC AUC: {vqc_metrics['roc_auc']:.4f} | QSVM AUC: {qsvm_metrics['roc_auc']:.4f}"
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def models_exist() -> bool:
    for f in ["vqc.joblib", "qsvm.joblib", "quantum_preprocessor.joblib",
              "classical_preprocessor.joblib", "vqc_threshold.joblib"]:
        if not (Path(settings.MODEL_DIR) / f).exists():
            return False
    return True


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Run QML inference on the full dataset."""
    # Load artifacts
    classical_pre = load_preprocessor("classical_preprocessor.joblib")
    q_artifacts   = joblib.load(Path(settings.MODEL_DIR) / "quantum_preprocessor.joblib")
    vqc_data      = joblib.load(Path(settings.MODEL_DIR) / "vqc.joblib")
    qsvm_data     = joblib.load(Path(settings.MODEL_DIR) / "qsvm.joblib")
    vqc_thresh    = joblib.load(Path(settings.MODEL_DIR) / "vqc_threshold.joblib")["threshold"]
    qsvm_bounds   = joblib.load(Path(settings.MODEL_DIR) / "qsvm_bounds.joblib")

    scaler_q  = q_artifacts["scaler"]
    pca       = q_artifacts["pca"]
    vqc_params = vqc_data["params"]
    n_qubits   = vqc_data["n_qubits"]
    n_layers   = vqc_data["n_layers"]
    svm_model  = qsvm_data["svm"]
    X_support  = qsvm_data["X_train"]

    # Preprocess
    X_raw, y_actual, employee_ids = clean_dataframe(df)
    X_eng = engineer_features(X_raw)

    # Quantum features
    numeric_cols = NUMERIC_FEATURES + ENGINEERED_FEATURES
    X_numeric    = X_eng[numeric_cols].values
    X_scaled     = scaler_q.transform(X_numeric)
    X_pca        = pca.transform(X_scaled)
    X_quantum    = rescale_to_angles(X_pca)

    # VQC predictions (batch processing for large datasets)
    print(f"[Predict] Running VQC on {len(X_quantum)} employees...")
    vqc_probs = vqc_predict_proba(X_quantum, vqc_params, n_qubits, n_layers)
    vqc_preds = (vqc_probs >= vqc_thresh).astype(int)

    # QSVM predictions
    print(f"[Predict] Running QSVM on {len(X_quantum)} employees...")
    qsvm_probs = qsvm_predict(X_quantum, svm_model, X_support, n_qubits)

    # Normalize QSVM scores
    p2, p98 = qsvm_bounds["p2"], qsvm_bounds["p98"]
    if p98 > p2:
        qsvm_norm = np.clip((qsvm_probs - p2) / (p98 - p2), 0, 1)
    else:
        qsvm_norm = qsvm_probs

    # Risk score fusion
    w_vqc  = settings.VQC_WEIGHT
    w_qsvm = settings.QSVM_WEIGHT
    risk_scores = np.clip(w_vqc * vqc_probs + w_qsvm * qsvm_norm, 0, 1) * 100

    def to_level(s):
        if s >= 67: return "high"
        if s >= 34: return "medium"
        return "low"

    result = X_raw.copy().reset_index(drop=True)
    result["employee_id"]      = employee_ids.values
    result["vqc_prob"]         = np.round(vqc_probs, 4)
    result["qsvm_score"]       = np.round(qsvm_probs, 6)
    result["qsvm_normalized"]  = np.round(qsvm_norm, 4)
    result["risk_score"]       = np.round(risk_scores, 2)
    result["risk_level"]       = [to_level(s) for s in risk_scores]
    result["is_malicious_pred"] = vqc_preds.astype(bool)
    if y_actual is not None:
        result["is_malicious_actual"] = y_actual.values

    print(f"[Predict] Distribution:")
    print(f"  Low:    {(risk_scores<34).sum()} ({(risk_scores<34).mean()*100:.1f}%)")
    print(f"  Medium: {((risk_scores>=34)&(risk_scores<67)).sum()} ({((risk_scores>=34)&(risk_scores<67)).mean()*100:.1f}%)")
    print(f"  High:   {(risk_scores>=67).sum()} ({(risk_scores>=67).mean()*100:.1f}%)")

    return result
