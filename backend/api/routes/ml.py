from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import pandas as pd

from backend.db.models import get_db, User, LogUpload, EmployeeRisk, ModelRegistry, UploadStatus, RiskLevel
from backend.core.security import require_admin, require_any
from backend.api.schemas.schemas import RiskSummary, EmployeeRiskOut, DashboardStats, TrainResult
from backend.ml.qml_models import train_models, models_exist
from backend.core.config import settings

router = APIRouter(prefix="/api/ml", tags=["QML"])


class TrainRequest(BaseModel):
    upload_id: int


@router.post("/train", response_model=TrainResult)
def trigger_training(payload: TrainRequest, db: Session = Depends(get_db),
                     current_user: User = Depends(require_admin)):
    upload = db.query(LogUpload).filter(LogUpload.id == payload.upload_id).first()
    if not upload: raise HTTPException(404, "Upload not found")
    try:
        df = pd.read_csv(upload.file_path)
    except Exception as e:
        raise HTTPException(422, f"Cannot read file: {e}")
    try:
        metrics = train_models(df)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")

    db.query(ModelRegistry).update({"is_active": False})
    db.add(ModelRegistry(
        model_type="quantum_hybrid",
        vqc_accuracy=metrics["vqc_accuracy"],
        vqc_roc_auc=metrics["vqc_roc_auc"],
        vqc_f1=metrics["vqc_f1"],
        qsvm_accuracy=metrics["qsvm_accuracy"],
        qsvm_roc_auc=metrics["qsvm_roc_auc"],
        n_qubits=metrics["n_qubits"],
        n_layers=metrics["n_layers"],
        train_samples=metrics["train_samples"],
        epochs=metrics["epochs"],
        is_active=True, trained_by=current_user.id,
    ))
    db.commit()
    return TrainResult(**{k: metrics[k] for k in TrainResult.model_fields if k in metrics})


@router.get("/model-status")
def model_status(db: Session = Depends(get_db), _: User = Depends(require_any)):
    active = db.query(ModelRegistry).filter(ModelRegistry.is_active == True).first()
    return {
        "models_trained": models_exist(),
        "active_model": {
            "trained_at":     active.trained_at.isoformat() if active else None,
            "vqc_accuracy":   active.vqc_accuracy if active else None,
            "vqc_roc_auc":    active.vqc_roc_auc if active else None,
            "qsvm_accuracy":  active.qsvm_accuracy if active else None,
            "qsvm_roc_auc":   active.qsvm_roc_auc if active else None,
            "n_qubits":       active.n_qubits if active else None,
            "n_layers":       active.n_layers if active else None,
            "train_samples":  active.train_samples if active else None,
        } if active else None
    }


@router.get("/results/{upload_id}", response_model=list[EmployeeRiskOut])
def get_results(upload_id: int, risk_level: str = None, limit: int = 50000, offset: int = 0,
                db: Session = Depends(get_db), _: User = Depends(require_any)):
    q = db.query(EmployeeRisk).filter(EmployeeRisk.upload_id == upload_id)
    if risk_level:
        try: q = q.filter(EmployeeRisk.risk_level == RiskLevel(risk_level))
        except ValueError: raise HTTPException(400, "Invalid risk_level")
    return q.order_by(EmployeeRisk.risk_score.desc()).offset(offset).limit(limit).all()


@router.get("/results/{upload_id}/summary", response_model=RiskSummary)
def get_summary(upload_id: int, db: Session = Depends(get_db), _: User = Depends(require_any)):
    rows = db.query(EmployeeRisk).filter(EmployeeRisk.upload_id == upload_id).all()
    if not rows: raise HTTPException(404, "No results")
    scores = [r.risk_score for r in rows if r.risk_score is not None]
    return RiskSummary(
        upload_id=upload_id, total_employees=len(rows),
        high_risk=sum(1 for r in rows if r.risk_level == RiskLevel.high),
        medium_risk=sum(1 for r in rows if r.risk_level == RiskLevel.medium),
        low_risk=sum(1 for r in rows if r.risk_level == RiskLevel.low),
        avg_risk_score=round(sum(scores)/len(scores), 2) if scores else 0,
        max_risk_score=round(max(scores), 2) if scores else 0,
        flagged_count=sum(1 for r in rows if r.is_malicious_pred),
    )


@router.get("/dashboard", response_model=DashboardStats)
def dashboard(db: Session = Depends(get_db), _: User = Depends(require_any)):
    latest = (db.query(LogUpload).filter(LogUpload.status == UploadStatus.completed)
              .order_by(LogUpload.processed_at.desc()).first())
    active = db.query(ModelRegistry).filter(ModelRegistry.is_active == True).first()
    if not latest:
        return DashboardStats(total_employees=0, high_risk=0, medium_risk=0, low_risk=0,
                              total_uploads=db.query(LogUpload).count(),
                              last_upload_at=None, vqc_accuracy=None, vqc_roc_auc=None,
                              qsvm_accuracy=None, qsvm_roc_auc=None)
    rows = db.query(EmployeeRisk).filter(EmployeeRisk.upload_id == latest.id).all()
    return DashboardStats(
        total_employees=len(rows),
        high_risk=sum(1 for r in rows if r.risk_level == RiskLevel.high),
        medium_risk=sum(1 for r in rows if r.risk_level == RiskLevel.medium),
        low_risk=sum(1 for r in rows if r.risk_level == RiskLevel.low),
        total_uploads=db.query(LogUpload).count(),
        last_upload_at=latest.processed_at,
        vqc_accuracy=active.vqc_accuracy if active else None,
        vqc_roc_auc=active.vqc_roc_auc if active else None,
        qsvm_accuracy=active.qsvm_accuracy if active else None,
        qsvm_roc_auc=active.qsvm_roc_auc if active else None,
    )


@router.get("/top-threats/{upload_id}")
def top_threats(upload_id: int, n: int = 10, db: Session = Depends(get_db),
                _: User = Depends(require_any)):
    rows = (db.query(EmployeeRisk).filter(EmployeeRisk.upload_id == upload_id)
            .order_by(EmployeeRisk.risk_score.desc()).limit(n).all())
    return [{"employee_id": r.employee_id, "risk_score": r.risk_score,
             "risk_level": r.risk_level, "vqc_prob": r.vqc_prob,
             "qsvm_normalized": r.qsvm_normalized, "department": r.department,
             "position": r.position, "total_files_burned": r.total_files_burned,
             "num_printed_pages_off_hours": r.num_printed_pages_off_hours} for r in rows]
