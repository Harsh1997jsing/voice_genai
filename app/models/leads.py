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


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True)

    name = Column(String)
    phone_number = Column(String, index=True, unique=True)
    customer_type = Column(String, default="new")
    campaign_name = Column(String)
    status = Column(String, default="pending") # pending / calling / completed / failed / retry
    interest_level = Column(String)# high / medium / low
    call_outcome = Column(String)# interested / followup / rejected / existing_customer
    call_summary = Column(Text)
    retry_count = Column(Integer, default=0)


    current_plan = Column(String)
    last_service_date = Column(String)

  
    callback_time = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
