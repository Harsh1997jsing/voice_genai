from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.services.twilio_service import search_numbers, buy_number, get_countries_code
from app.core.deps import get_current_user
from app.models.user import User
from app.models.phone_number import PhoneNumber
from app.models.leads import Lead
from app.db.database import get_db
from app.services.twilio_service import buy_number, selected_number
router = APIRouter(prefix="/number", tags=["Numbers"])


@router.get("/search")
def get_numbers(country: str = "US", current_user: User = Depends(get_current_user)):
    numbers = search_numbers(country)
    return {"available_numbers": numbers}

@router.get("/get_countries")
async def get_supported_countries():
    try:
        countries =get_countries_code()
        result = []

        for country in countries:
            result.append({
                "country_code": country.country_code,
                "country": country.country
            })

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/buy")
def buying_number(
    phone_number: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = buy_number(
        db=db,
        user_id=current_user.id,
        phone_number=phone_number
    )

    return {
        "message": "Number purchased",
        "number": result.number
    }

# @router.get("/my-numbers")
# def get_my_numbers(
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     numbers = selected_number(current_user=current_user, db=db)
#     return numbers


@router.get("/bought")
def get_bought_numbers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all bought/owned numbers for current user"""

    bought_numbers = db.query(PhoneNumber).filter(
        PhoneNumber.user_id == current_user.id
    ).all()

    return {
        "total_bought": len(bought_numbers),
        "numbers": [
            {
                "id": num.id,
                "number": num.number,
                "created_at": num.created_at.isoformat() if num.created_at else None
            }
            for num in bought_numbers
        ]
    }


@router.get("/target_numbers")
async def get_target_numbers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get all target numbers from leads table for current user"""

    target_numbers = db.query(Lead).filter(
        Lead.user_id == current_user.id
    ).all()

    return {
        "total_targets": len(target_numbers),
        "numbers": [
            {
                "id": num.id,
                "phone_number": num.phone_number,
                "name": num.name,
                "status": num.status,
            }
            for num in target_numbers
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Manually add / list / delete leads
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/lead")
async def add_lead(
    phone_number: str,
    name: str | None = None,
    customer_type: str | None = "new",
    campaign_name: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Manually add a single phone number as a lead."""

    phone_number = phone_number.strip()
    if not phone_number:
        raise HTTPException(status_code=400, detail="Phone number is required")

    existing = db.query(Lead).filter(Lead.phone_number == phone_number).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Phone number {phone_number} already exists (lead id={existing.id})",
        )

    lead = Lead(
        user_id=current_user.id,
        phone_number=phone_number,
        name=name,
        customer_type=customer_type,
        campaign_name=campaign_name,
    )
    db.add(lead)
    db.commit()
    db.refresh(lead)

    return {
        "message": "Lead added successfully",
        "lead": {
            "id": lead.id,
            "phone_number": lead.phone_number,
            "name": lead.name,
            "customer_type": lead.customer_type,
            "campaign_name": lead.campaign_name,
            "status": lead.status,
        },
    }


@router.post("/leads")
async def add_leads_bulk(
    numbers: list[dict],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Manually add multiple phone numbers.

    Request body example:
    ```json
    [
        {"phone_number": "+911234567890", "name": "John"},
        {"phone_number": "+919876543210"}
    ]
    ```
    """
    added = []
    skipped = []

    for entry in numbers:
        phone = str(entry.get("phone_number", "")).strip()
        if not phone:
            continue

        existing = db.query(Lead).filter(Lead.phone_number == phone).first()
        if existing:
            skipped.append(phone)
            continue

        lead = Lead(
            id=current_user.id,
            phone_number=phone,
            name=entry.get("name"),
            customer_type=entry.get("customer_type", "new"),
            campaign_name=entry.get("campaign_name"),
        )
        db.add(lead)
        added.append(phone)

    db.commit()

    return {
        "message": f"Added {len(added)} leads, skipped {len(skipped)} duplicates",
        "added": added,
        "skipped": skipped,
    }


@router.get("/leads")
async def list_leads(
    status: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all leads for the current user. Optionally filter by status."""

    query = db.query(Lead).filter(Lead.user_id == current_user.id)
    if status:
        query = query.filter(Lead.status == status)
    leads = query.order_by(Lead.created_at.desc()).all()

    return {
        "total": len(leads),
        "leads": [
            {
                "id": l.id,
                "phone_number": l.phone_number,
                "name": l.name,
                "customer_type": l.customer_type,
                "campaign_name": l.campaign_name,
                "status": l.status,
                "call_outcome": l.call_outcome,
                "created_at": l.created_at.isoformat() if l.created_at else None,
            }
            for l in leads
        ],
    }


@router.delete("/lead/{lead_id}")
async def delete_lead(
    lead_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a lead by ID (only if it belongs to the current user)."""

    lead = (
        db.query(Lead)
        .filter(Lead.id == lead_id, Lead.user_id == current_user.id)
        .first()
    )
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    db.delete(lead)
    db.commit()

    return {"message": f"Lead {lead_id} deleted"}