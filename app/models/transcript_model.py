from sqlalchemy import (
    JSON,
    Column,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
)

from sqlalchemy.sql import func

from app.db.database import Base


class Transcript(Base):

    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True)
    call_id = Column(String(100), index=True)
    speaker = Column(String(20))
    transcript = Column(JSON)
    is_final = Column(Boolean, default=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )