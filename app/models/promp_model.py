from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    JSON,
)

from datetime import datetime
from app.db.database import Base


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True)
    call_id = Column(String(100), index=True)
    system_prompt = Column(JSON)
    one_liner = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)