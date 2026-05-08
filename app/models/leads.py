from sqlalchemy import Column, Integer, String
from app.db.database import Base

class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True)
    phone_number = Column(String, index=True)
    status = Column(String, default="pending")  # pending, called, failed