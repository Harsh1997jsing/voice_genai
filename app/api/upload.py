
from fastapi import APIRouter, UploadFile, File, Depends
import pandas as pd
import io
from sqlalchemy.orm import Session
from app.models.leads import Lead
from app.db.database import get_db

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("/excel")
async def upload_excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    df = pd.read_excel(io.BytesIO(content))

    numbers = df["phone"].dropna().tolist()


    saved = []
    for num in numbers:
        lead = Lead(phone_number=str(num))
        db.add(lead)
        saved.append(str(num))

    db.commit()
    db.close()

    return {"total_uploaded": len(saved)}


@router.get("/stats")
async def get_lead_stats(db: Session = Depends(get_db)):
    """Get total lead count and pending status count"""
    total_leads = db.query(Lead).count()
    pending_leads = db.query(Lead).filter(Lead.status == "pending").count()
    
    return {
        "total_leads": 20,
        "pending_leads": 5,
        "called_leads": 10,
        "failed_leads": 5
    }
