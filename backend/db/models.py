from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import enum

from backend.core.config import settings

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class UserRole(str, enum.Enum):
    admin   = "admin"
    analyst = "analyst"
    viewer  = "viewer"


class RiskLevel(str, enum.Enum):
    low    = "low"
    medium = "medium"
    high   = "high"


class UploadStatus(str, enum.Enum):
    pending    = "pending"
    processing = "processing"
    completed  = "completed"
    failed     = "failed"


class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(64), unique=True, index=True, nullable=False)
    email           = Column(String(128), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    full_name       = Column(String(128))
    role            = Column(Enum(UserRole), default=UserRole.viewer, nullable=False)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
    last_login      = Column(DateTime, nullable=True)
    uploads         = relationship("LogUpload", back_populates="uploaded_by_user")


class LogUpload(Base):
    __tablename__ = "log_uploads"
    id            = Column(Integer, primary_key=True, index=True)
    filename      = Column(String(256), nullable=False)
    original_name = Column(String(256), nullable=False)
    file_path     = Column(String(512), nullable=False)
    file_size     = Column(Integer)
    row_count     = Column(Integer)
    status        = Column(Enum(UploadStatus), default=UploadStatus.pending)
    error_message = Column(Text, nullable=True)
    uploaded_by   = Column(Integer, ForeignKey("users.id"))
    uploaded_at   = Column(DateTime, default=datetime.utcnow)
    processed_at  = Column(DateTime, nullable=True)
    uploaded_by_user = relationship("User", back_populates="uploads")
    results          = relationship("EmployeeRisk", back_populates="upload")


class EmployeeRisk(Base):
    __tablename__ = "employee_risks"
    id                          = Column(Integer, primary_key=True, index=True)
    upload_id                   = Column(Integer, ForeignKey("log_uploads.id"), index=True)
    employee_id                 = Column(String(64), index=True)
    department                  = Column(String(128))
    campus                      = Column(String(64))
    position                    = Column(String(128))
    country                     = Column(String(64))
    seniority                   = Column(Float)
    contractor_status           = Column(Integer)
    clearance_level             = Column(Integer)
    total_printed_pages         = Column(Float)
    num_printed_pages_off_hours = Column(Float)
    total_files_burned          = Column(Float)
    burned_from_other           = Column(Float)
    abroad_status               = Column(Integer)
    trip_duration               = Column(Float)
    country_risk_level          = Column(Float)
    building_entries            = Column(Float)
    campus_visits               = Column(Float)
    late_exit                   = Column(Float)
    weekend_entry               = Column(Float)
    foreign_citizenship         = Column(Integer)
    criminal_record             = Column(Integer)
    medical_history             = Column(Integer)
    # QML-specific outputs
    vqc_prob                    = Column(Float)   # VQC classification probability
    qsvm_score                  = Column(Float)   # QSVM decision score
    qsvm_normalized             = Column(Float)   # normalized 0-1
    risk_score                  = Column(Float)   # final 0-100
    risk_level                  = Column(Enum(RiskLevel))
    is_malicious_pred           = Column(Boolean)
    is_malicious_actual         = Column(Boolean, nullable=True)
    created_at                  = Column(DateTime, default=datetime.utcnow)
    upload = relationship("LogUpload", back_populates="results")


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id               = Column(Integer, primary_key=True, index=True)
    model_type       = Column(String(64))
    vqc_accuracy     = Column(Float, nullable=True)
    vqc_roc_auc      = Column(Float, nullable=True)
    vqc_f1           = Column(Float, nullable=True)
    qsvm_accuracy    = Column(Float, nullable=True)
    qsvm_roc_auc     = Column(Float, nullable=True)
    n_qubits         = Column(Integer)
    n_layers         = Column(Integer)
    train_samples    = Column(Integer)
    epochs           = Column(Integer)
    is_active        = Column(Boolean, default=True)
    trained_at       = Column(DateTime, default=datetime.utcnow)
    trained_by       = Column(Integer, ForeignKey("users.id"), nullable=True)


def create_tables():
    Base.metadata.create_all(bind=engine)
