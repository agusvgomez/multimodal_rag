from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum

class RoleEnum(str, Enum):
    human = "user"
    ai = "assistant"
    system = "system"

class HistoryEntry(BaseModel):
    role: RoleEnum
    content: str

class PredictRequest(BaseModel):
    question: str
    chat_history: List[HistoryEntry]
    summary: Optional[str] = ""

class Document(BaseModel):
    page_content: str
    metadata: Dict[str, str]
    type: str

class Response(BaseModel):
    answer: str
    documents: str # List[Document]
    chat_history: List[HistoryEntry]
    summary: Optional[str] = ""