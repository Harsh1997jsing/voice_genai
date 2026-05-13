
from pathlib import Path
import time

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import pandas as pd
import io
from sqlalchemy.orm import Session
from app.models.promp_model import Prompt
from app.core.deps import get_current_user
from app.models.company_profile import CompanyProfile
from app.models.leads import Lead
from app.db.database import get_db
from app.models.user import User
from app.services.file_parsing_service import extract_text_from_file, ExtractionError
from app.services.prompt_genaration import _generate_system_prompt

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




@router.post("/company-pdf")
async def upload_company_pdf(
    file: UploadFile = File(...),
    company_name: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    extension = Path(file.filename or "").suffix.lower()
    if extension != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        extraction = extract_text_from_file(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to process uploaded PDF")

    if not extraction.text.strip():
        raise HTTPException(status_code=422, detail="No readable text found in PDF")

    try:
        generated_prompt = await _generate_system_prompt(
            raw_text=extraction.text,
            company_name=company_name,
        )

        # generated_prompt["call_intro"]
        print(generated_prompt)

        prompt = Prompt(
            call_id=f"call_{user.id}_{int(time.time())}",
            system_prompt={"text": generated_prompt["system_prompt"]},
            one_liner={"text": generated_prompt["call_intro"]},
        )
        db.add(prompt)
        db.commit()
        db.refresh(prompt)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate system prompt: {e}",
        )

    existing = (
        db.query(CompanyProfile)
        .filter(CompanyProfile.user_id == user.id)
        .first()
    )
    if existing:
        existing.company_name = company_name
        existing.source_filename = file.filename or "uploaded.pdf"
        existing.raw_text = extraction.text
        existing.system_prompt = generated_prompt["system_prompt"]
        record = existing
    else:
        record = CompanyProfile(
            user_id=user.id,
            company_name=company_name,
            source_filename=file.filename or "uploaded.pdf",
            raw_text=extraction.text,
            system_prompt=generated_prompt["system_prompt"],
        )
        db.add(record)

    db.commit()
    db.refresh(record)

    return {
        "message": "Company PDF processed successfully",
        "profile_id": record.id,
        "source_filename": record.source_filename,
        "raw_text_chars": len(record.raw_text),
        "system_prompt_preview": record.system_prompt[:300],
    }
