import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, Dict, Any

from backend.core.config import (
    settings, TARGET_COLUMN, COLUMN_ALIASES,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    ENGINEERED_FEATURES, ALL_FEATURES
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    return df


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    df = normalize_columns(df.copy())
    present  = set(df.columns)
    required = set(ALL_FEATURES)
    missing  = sorted(required - present)
    extra    = sorted(present - required - {TARGET_COLUMN, "employee_id"})
    errors, warnings = [], []
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
    if TARGET_COLUMN not in df.columns:
        warnings.append("No 'is_malicious' column — unsupervised mode only.")
    return {
        "valid": len(errors) == 0,
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_columns": missing,
        "extra_columns": extra,
        "errors": errors,
        "warnings": warnings,
        "sample_rows": min(5, len(df)),
    }


def clean_dataframe(df: pd.DataFrame):
    df = normalize_columns(df.copy())

    y = None
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].astype(int)
        df = df.drop(columns=[TARGET_COLUMN])

    employee_ids = None
    for id_col in ["employee_id", "id", "emp_id"]:
        if id_col in df.columns:
            employee_ids = df[id_col].astype(str)
            df = df.drop(columns=[id_col])
            break
    if employee_ids is None:
        employee_ids = pd.Series([f"EMP-{i:05d}" for i in range(len(df))])

    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df[ALL_FEATURES].copy()

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        med = df[col].median()
        df[col] = df[col].fillna(med if not pd.isna(med) else 0.0)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).fillna("Unknown").replace("nan", "Unknown")

    return df, y, employee_ids


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total = df["total_printed_pages"].clip(lower=1)
    df["off_hours_print_ratio"] = df["num_printed_pages_off_hours"] / total
    df["exfiltration_score"] = (
        0.424 * df["total_files_burned"] +
        0.298 * df["burned_from_other"] +
        0.197 * df["num_printed_pages_off_hours"] +
        0.157 * df["total_printed_pages"]
    )
    df["physical_risk_score"]    = df["late_exit"] + df["weekend_entry"] + df["building_entries"] * 0.1
    df["travel_risk_score"]      = df["trip_duration"] * df["country_risk_level"]
    df["background_risk"]        = df["foreign_citizenship"] + df["criminal_record"] * 2 + df["medical_history"]
    df["access_seniority_ratio"] = df["building_entries"] / df["seniority"].clip(lower=1)
    return df


def build_classical_preprocessor() -> ColumnTransformer:
    """Full preprocessor for classical feature space (27 dims)."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("cat", Pipeline([("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )


def build_quantum_preprocessor(n_qubits: int = 8) -> Pipeline:
    """
    Quantum preprocessor: StandardScaler → PCA → reduce to n_qubits dims → rescale to [-π, π].
    Quantum circuits encode features as rotation angles, so we need:
    1. Standardize features
    2. PCA to reduce to exactly n_qubits dimensions
    3. Rescale to [-π, π] for angle encoding
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_qubits)),
        ("angle_scaler", StandardScaler()),  # will be rescaled after
    ])


def rescale_to_angles(X: np.ndarray) -> np.ndarray:
    """Rescale PCA-reduced features to [-π, π] for quantum angle encoding."""
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    rng = X_max - X_min
    rng[rng == 0] = 1
    X_norm = (X - X_min) / rng          # [0, 1]
    return (X_norm * 2 - 1) * np.pi     # [-π, π]


def save_preprocessor(preprocessor, name: str):
    joblib.dump(preprocessor, Path(settings.MODEL_DIR) / name)

def load_preprocessor(name: str):
    path = Path(settings.MODEL_DIR) / name
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor '{name}' not found. Train models first.")
    return joblib.load(path)
