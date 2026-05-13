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
    current_user: User = Depends(get_current_user)
):
    """Get all target numbers from leads table for current user"""

   

    target_numbers = db.query(Lead).filter(
        Lead.id == current_user.id
    ).all()

    return {
        "total_targets": len(target_numbers),
        "numbers": [
            {
                "phone_number": num.phone_number,
                "status": num.status
            }
            for num in target_numbers
        ]
    }