from app.core.config import settings
from app.core.twilio_clinet import client
from app.models.phone_number import PhoneNumber
from app.models.user import User
from urllib.parse import urlencode

def get_countries_code():
    countries = client.available_phone_numbers.list()
    print(f"Retrieved {len(countries)} countries from Twilio API")

    return countries

def make_call(
    to_number: str,
    user_id: int | None = None,
    one_liner: str | None = None,
    system_prompt: str | None = None,
    from_number: str | None = None,
) -> str:
    """
    Synchronous Twilio call creator.
    Always call this via asyncio.to_thread() from async routes
    to avoid blocking the event loop.
    """
    base_url = settings.PUBLIC_BASE_URL.rstrip("/")
 
    # Build query params for the /twilio/voice webhook
    query: dict[str, str | int] = {}
    if user_id is not None:
        query["user_id"] = user_id
    # Keep webhook URL short and stable; prompt content is fetched server-side.
 
    voice_url = f"{base_url}/twilio/voice"
    if query:
        voice_url = f"{voice_url}?{urlencode(query)}"
 
    call = client.calls.create(
        to=to_number,
        from_=from_number,
        url=voice_url,
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
    number = db.query(PhoneNumber).filter(PhoneNumber.user_id == user_id).first()
    if number:
        raise Exception("User already has a number")

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
