import csv, io
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable

from backend.db.models import get_db, User, EmployeeRisk, LogUpload, RiskLevel
from backend.core.security import require_any

router = APIRouter(prefix="/api/reports", tags=["Reports"])
RISK_COLORS = {"high": colors.HexColor("#ff2a2a"), "medium": colors.HexColor("#ffaa00"), "low": colors.HexColor("#00cc66")}


@router.get("/{upload_id}/csv")
def export_csv(upload_id: int, risk_level: str = None, db: Session = Depends(get_db),
               _: User = Depends(require_any)):
    q = db.query(EmployeeRisk).filter(EmployeeRisk.upload_id == upload_id)
    if risk_level:
        try: q = q.filter(EmployeeRisk.risk_level == RiskLevel(risk_level))
        except ValueError: raise HTTPException(400, "Invalid risk_level")
    rows = q.order_by(EmployeeRisk.risk_score.desc()).all()
    if not rows: raise HTTPException(404, "No results")

    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["employee_id","risk_score","risk_level","vqc_prob","qsvm_normalized",
                "department","position","total_files_burned","num_printed_pages_off_hours",
                "is_malicious_pred","is_malicious_actual"])
    for r in rows:
        w.writerow([r.employee_id, r.risk_score, r.risk_level, r.vqc_prob, r.qsvm_normalized,
                    r.department, r.position, r.total_files_burned, r.num_printed_pages_off_hours,
                    r.is_malicious_pred, r.is_malicious_actual])
    output.seek(0)
    fname = f"quantumwatch_{upload_id}_{datetime.now().strftime('%Y%m%d')}.csv"
    return StreamingResponse(io.BytesIO(output.getvalue().encode()), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={fname}"})


@router.get("/{upload_id}/pdf")
def export_pdf(upload_id: int, risk_level: str = None, db: Session = Depends(get_db),
               _: User = Depends(require_any)):
    upload = db.query(LogUpload).filter(LogUpload.id == upload_id).first()
    if not upload: raise HTTPException(404, "Upload not found")
    q = db.query(EmployeeRisk).filter(EmployeeRisk.upload_id == upload_id)
    if risk_level:
        try: q = q.filter(EmployeeRisk.risk_level == RiskLevel(risk_level))
        except ValueError: raise HTTPException(400, "Invalid risk_level")
    rows = q.order_by(EmployeeRisk.risk_score.desc()).all()
    if not rows: raise HTTPException(404, "No results")

    high   = sum(1 for r in rows if r.risk_level == RiskLevel.high)
    med    = sum(1 for r in rows if r.risk_level == RiskLevel.medium)
    low    = sum(1 for r in rows if r.risk_level == RiskLevel.low)
    scores = [r.risk_score for r in rows if r.risk_score]
    avg    = round(sum(scores)/len(scores), 1) if scores else 0

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
                            leftMargin=15*mm, rightMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    story  = []
    story.append(Paragraph("QuantumWatch — Quantum ML Threat Intelligence Report",
                            ParagraphStyle("T", parent=styles["Title"], fontSize=16, fontName="Helvetica-Bold")))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(f"Upload: {upload.original_name}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Total: {len(rows)}",
                            ParagraphStyle("B", parent=styles["Normal"], fontSize=9)))
    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#7b2fbe")))
    story.append(Spacer(1, 4*mm))

    sum_data = [["Metric","Value"],["Total Employees",str(len(rows))],
                ["High Risk",str(high)],["Medium Risk",str(med)],["Low Risk",str(low)],
                ["Avg Risk Score",str(avg)]]
    st = Table(sum_data, colWidths=[55*mm, 35*mm])
    st.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a0a2e")),
                             ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                             ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                             ("FONTSIZE",(0,0),(-1,-1),9),
                             ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#cccccc")),
                             ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f8f8f8"),colors.white])]))
    story.append(st)
    story.append(Spacer(1, 6*mm))

    header = ["Employee ID","Level","Score","VQC Prob","QSVM","Department","Position","Files","Off-Hr Prints"]
    tdata  = [header]
    for r in rows[:200]:
        tdata.append([r.employee_id or "—", str(r.risk_level or "").upper(),
                      f"{r.risk_score:.1f}" if r.risk_score else "—",
                      f"{r.vqc_prob:.3f}" if r.vqc_prob else "—",
                      f"{r.qsvm_normalized:.3f}" if r.qsvm_normalized else "—",
                      r.department or "—", r.position or "—",
                      f"{r.total_files_burned:.0f}" if r.total_files_burned is not None else "—",
                      f"{r.num_printed_pages_off_hours:.0f}" if r.num_printed_pages_off_hours is not None else "—"])

    et = Table(tdata, colWidths=[30*mm,18*mm,18*mm,20*mm,18*mm,38*mm,38*mm,20*mm,24*mm], repeatRows=1)
    row_styles = [("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a0a2e")),
                  ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                  ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                  ("FONTSIZE",(0,0),(-1,-1),7),
                  ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#dddddd")),
                  ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#fafafa"),colors.white])]
    for i, r in enumerate(rows[:200], 1):
        c = RISK_COLORS.get(str(r.risk_level or "low"), colors.gray)
        row_styles += [("TEXTCOLOR",(1,i),(1,i),c),("FONTNAME",(1,i),(1,i),"Helvetica-Bold")]
    et.setStyle(TableStyle(row_styles))
    story.append(et)
    doc.build(story)
    buf.seek(0)
    fname = f"quantumwatch_{upload_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
    return StreamingResponse(buf, media_type="application/pdf",
                             headers={"Content-Disposition": f"attachment; filename={fname}"})
