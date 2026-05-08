from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from backend.db.models import UserRole, RiskLevel, UploadStatus


class UserCreate(BaseModel):
    username:  str = Field(..., min_length=3, max_length=64)
    email:     EmailStr
    password:  str = Field(..., min_length=6)
    full_name: Optional[str] = None
    role:      UserRole = UserRole.viewer

class UserOut(BaseModel):
    id: int; username: str; email: str; full_name: Optional[str]
    role: UserRole; is_active: bool; created_at: datetime; last_login: Optional[datetime]
    class Config: from_attributes = True

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)

class Token(BaseModel):
    access_token: str; token_type: str = "bearer"; user: UserOut

class LoginRequest(BaseModel):
    username: str; password: str

class UploadOut(BaseModel):
    id: int; filename: str; original_name: str
    file_size: Optional[int]; row_count: Optional[int]
    status: UploadStatus; error_message: Optional[str]
    uploaded_at: datetime; processed_at: Optional[datetime]
    class Config: from_attributes = True

class ValidationResult(BaseModel):
    valid: bool; row_count: int; column_count: int
    missing_columns: List[str]; extra_columns: List[str]
    errors: List[str]; warnings: List[str]; sample_rows: int

class EmployeeRiskOut(BaseModel):
    id: int; employee_id: str
    department: Optional[str]; campus: Optional[str]
    position: Optional[str]; country: Optional[str]
    vqc_prob: Optional[float]; qsvm_score: Optional[float]
    qsvm_normalized: Optional[float]; risk_score: Optional[float]
    risk_level: Optional[RiskLevel]; is_malicious_pred: Optional[bool]
    is_malicious_actual: Optional[bool]
    total_files_burned: Optional[float]; num_printed_pages_off_hours: Optional[float]
    total_printed_pages: Optional[float]
    class Config: from_attributes = True

class RiskSummary(BaseModel):
    upload_id: int; total_employees: int
    high_risk: int; medium_risk: int; low_risk: int
    avg_risk_score: float; max_risk_score: float; flagged_count: int

class DashboardStats(BaseModel):
    total_employees: int; high_risk: int; medium_risk: int; low_risk: int
    total_uploads: int; last_upload_at: Optional[datetime]
    vqc_accuracy: Optional[float]; vqc_roc_auc: Optional[float]
    qsvm_accuracy: Optional[float]; qsvm_roc_auc: Optional[float]

class TrainRequest(BaseModel):
    upload_id: int

class TrainResult(BaseModel):
    vqc_accuracy: float; vqc_f1: float; vqc_roc_auc: float; vqc_threshold: float
    qsvm_accuracy: float; qsvm_f1: float; qsvm_roc_auc: float
    n_qubits: int; n_layers: int; train_samples: int; epochs: int
    pca_variance: float; loss_final: float; message: str
