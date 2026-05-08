import uuid
from pathlib import Path
from datetime import datetime
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from backend.db.models import get_db, User, LogUpload, EmployeeRisk, UploadStatus, RiskLevel
from backend.core.security import require_analyst, require_any
from backend.core.config import settings
from backend.api.schemas.schemas import UploadOut, ValidationResult
from backend.ml.preprocessing import validate_dataframe
from backend.ml.qml_models import predict, models_exist

router = APIRouter(prefix="/api/uploads", tags=["Uploads"])
MAX_FILE_SIZE = 50 * 1024 * 1024


@router.post("/", response_model=UploadOut, status_code=201)
async def upload_log(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                     db: Session = Depends(get_db), current_user: User = Depends(require_analyst)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files accepted.")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large (max 50MB).")

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path   = upload_dir / unique_name
    file_path.write_bytes(content)

    try:
        df = pd.read_csv(file_path)
        row_count = len(df)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(422, f"Cannot parse CSV: {e}")

    record = LogUpload(filename=unique_name, original_name=file.filename,
                       file_path=str(file_path), file_size=len(content),
                       row_count=row_count, status=UploadStatus.pending,
                       uploaded_by=current_user.id)
    db.add(record); db.commit(); db.refresh(record)

    if models_exist():
        background_tasks.add_task(_process_upload, record.id, str(file_path))

    return record


@router.post("/validate", response_model=ValidationResult)
async def validate_file(file: UploadFile = File(...), _: User = Depends(require_analyst)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV accepted.")
    content = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as e:
        raise HTTPException(422, str(e))
    return validate_dataframe(df)


@router.get("/", response_model=list[UploadOut])
def list_uploads(db: Session = Depends(get_db), _: User = Depends(require_any)):
    return db.query(LogUpload).order_by(LogUpload.uploaded_at.desc()).all()


@router.get("/{upload_id}", response_model=UploadOut)
def get_upload(upload_id: int, db: Session = Depends(get_db), _: User = Depends(require_any)):
    r = db.query(LogUpload).filter(LogUpload.id == upload_id).first()
    if not r: raise HTTPException(404, "Upload not found")
    return r


@router.post("/{upload_id}/process")
def process_upload(upload_id: int, background_tasks: BackgroundTasks,
                   db: Session = Depends(get_db), _: User = Depends(require_analyst)):
    r = db.query(LogUpload).filter(LogUpload.id == upload_id).first()
    if not r: raise HTTPException(404, "Upload not found")
    if r.status == UploadStatus.processing:
        raise HTTPException(409, "Already processing")
    if not models_exist():
        raise HTTPException(400, "No trained models. Train models first.")
    background_tasks.add_task(_process_upload, upload_id, r.file_path)
    return {"message": "Processing started", "upload_id": upload_id}


def _process_upload(upload_id: int, file_path: str):
    from backend.db.models import SessionLocal
    db = SessionLocal()
    try:
        record = db.query(LogUpload).filter(LogUpload.id == upload_id).first()
        if not record: return
        record.status = UploadStatus.processing
        db.commit()

        df = pd.read_csv(file_path)
        result_df = predict(df)

        db.query(EmployeeRisk).filter(EmployeeRisk.upload_id == upload_id).delete()

        feature_cols = [
            "department","campus","position","country","seniority","contractor_status",
            "clearance_level","total_printed_pages","num_printed_pages_off_hours",
            "total_files_burned","burned_from_other","abroad_status","trip_duration",
            "country_risk_level","building_entries","campus_visits","late_exit",
            "weekend_entry","foreign_citizenship","criminal_record","medical_history",
        ]

        for _, row in result_df.iterrows():
            obj = EmployeeRisk(
                upload_id=upload_id,
                employee_id=str(row.get("employee_id", "")),
                vqc_prob=float(row["vqc_prob"]),
                qsvm_score=float(row["qsvm_score"]),
                qsvm_normalized=float(row["qsvm_normalized"]),
                risk_score=float(row["risk_score"]),
                risk_level=RiskLevel(row["risk_level"]),
                is_malicious_pred=bool(row["is_malicious_pred"]),
                is_malicious_actual=bool(row["is_malicious_actual"]) if "is_malicious_actual" in row.index else None,
            )
            for col in feature_cols:
                if col in row.index:
                    setattr(obj, col, row[col])
            db.add(obj)

        record.status = UploadStatus.completed
        record.processed_at = datetime.utcnow()
        db.commit()
        print(f"[Upload {upload_id}] Done — {len(result_df)} records saved.")
    except Exception as e:
        print(f"[Upload {upload_id}] ERROR: {e}")
        import traceback; traceback.print_exc()
        try:
            record = db.query(LogUpload).filter(LogUpload.id == upload_id).first()
            if record:
                record.status = UploadStatus.failed
                record.error_message = str(e)[:500]
                db.commit()
        except: pass
    finally:
        db.close()
