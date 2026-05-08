from app.core.config import settings
from app.core.twilio_clinet import client
from app.models.phone_number import PhoneNumber
from app.models.user import User

def get_countries_code():
    countries = client.available_phone_numbers.list()
    print(f"Retrieved {len(countries)} countries from Twilio API")

    return countries

def make_call(to_number: str):
    # FIX #4: Use PUBLIC_BASE_URL — Twilio cannot reach 127.0.0.1
    # Set PUBLIC_BASE_URL=https://yourdomain.com in your .env
    base_url = settings.PUBLIC_BASE_URL.rstrip("/")

    call = client.calls.create(
        to=to_number,
        from_=settings.TWILIO_NUMBER,
        url=f"{base_url}/twilio/voice"
    )
    return call.sid

def search_numbers(country_code: str = "US"):
    numbers = client.available_phone_numbers(country_code).local.list(
        sms_enabled=True,
        voice_enabled=True,
        limit=10
    )
    return [num.phone_number for num in numbers]



def buy_number(phone_number: str, user_id: int, db):
    purchased = client.incoming_phone_numbers.create(
        phone_number=phone_number
    )

    db_number = PhoneNumber(
            number=purchased.phone_number,
            twilio_sid=purchased.sid,
            user_id=user_id,
            status="active"
        )

    db.add(db_number)
    db.commit()
    db.refresh(db_number)

    return db_number

def selected_number(current_user: User, db):
    numbers = db.query(PhoneNumber).filter(
        PhoneNumber.user_id == current_user.id
    ).all()

    return numbers

def release_number(db, number_id: int):
    number = db.query(PhoneNumber).filter(
        PhoneNumber.id == number_id
    ).first()

    if not number:
        return None

    client.incoming_phone_numbers(number.twilio_sid).delete()

    number.status = "released"
    db.commit()

    return number