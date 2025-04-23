from pydantic import BaseModel
from typing import Union, List

class ContentBase(BaseModel):
    content_type: str
    description: str
    page: int
    positions: List
    original: Union[str, bytes, dict]

class VectorRecord(ContentBase):
    vector: List[float]  