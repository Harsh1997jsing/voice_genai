from app.db.database import SessionLocal
from app.models.transcript_model import Transcript


def save_call_transcript_to_db_sync(
    call_id,
    transcript_json,
):

    db = SessionLocal()

    try:

        row = Transcript(
            call_id=call_id,
            transcript=transcript_json,
        )

        db.add(row)

        db.commit()

    finally:
        db.close()