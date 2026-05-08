from pydantic import BaseModel
from typing import List, Optional


class KBAddRequest(BaseModel):
    text: str
    metadata: dict = {}


class KBSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class KBSearchResponse(BaseModel):
    results: List[str]