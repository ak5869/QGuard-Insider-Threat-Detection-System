from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    SECRET_KEY: str = "quantumwatch-secret-key-2025"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    DATABASE_URL: str = "sqlite:///./quantumwatch.db"
    UPLOAD_DIR: str = "uploads"
    MODEL_DIR: str = "models"
    REPORTS_DIR: str = "reports"

    # QML hyperparameters
    N_QUBITS: int = 8           # number of qubits in the quantum circuit
    N_LAYERS: int = 3           # number of variational layers
    QML_TRAIN_SAMPLES: int = 5000   # subsample for quantum training
    QML_EPOCHS: int = 30        # training epochs
    QML_LEARNING_RATE: float = 0.01

    # Risk score weights
    VQC_WEIGHT: float = 0.55    # Variational Quantum Classifier weight
    QSVM_WEIGHT: float = 0.45   # Quantum Kernel SVM weight

    class Config:
        env_file = ".env"


settings = Settings()

for d in [settings.UPLOAD_DIR, settings.MODEL_DIR, settings.REPORTS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "is_malicious"

# Column aliases — same as InsiderWatch, works with your existing CSV
COLUMN_ALIASES = {
    "employee_department":      "department",
    "employee_campus":          "campus",
    "employee_position":        "position",
    "employee_origin_country":  "country",
    "employee_seniority_years": "seniority",
    "is_contractor":            "contractor_status",
    "employee_classification":  "clearance_level",
    "has_foreign_citizenship":  "foreign_citizenship",
    "has_criminal_record":      "criminal_record",
    "has_medical_history":      "medical_history",
    "is_abroad":                "abroad_status",
    "trip_day_number":          "trip_duration",
    "hostility_country_level":  "country_risk_level",
    "num_entries":              "building_entries",
    "num_unique_campus":        "campus_visits",
    "late_exit_flag":           "late_exit",
    "entry_during_weekend":     "weekend_entry",
    "malicious":                "is_malicious",
    "label":                    "is_malicious",
}

CATEGORICAL_FEATURES = ["department", "campus", "position", "country"]

NUMERIC_FEATURES = [
    "seniority", "contractor_status", "clearance_level",
    "total_printed_pages", "num_printed_pages_off_hours",
    "total_files_burned", "burned_from_other",
    "abroad_status", "trip_duration", "country_risk_level",
    "building_entries", "campus_visits", "late_exit", "weekend_entry",
    "foreign_citizenship", "criminal_record", "medical_history",
]

ENGINEERED_FEATURES = [
    "off_hours_print_ratio", "exfiltration_score",
    "physical_risk_score", "travel_risk_score",
    "background_risk", "access_seniority_ratio",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
