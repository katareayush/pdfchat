from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class PageData(BaseModel):
    page_number: int
    text_chunk: str
    char_count: int

class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    total_pages: int
    total_characters: int
    pages_with_text: int
    upload_time: datetime

class DocumentDetail(BaseModel):
    doc_id: str
    filename: str
    total_pages: int
    pages: List[PageData]
    total_chars: int
    upload_time: datetime

class DocumentList(BaseModel):
    documents: List[DocumentResponse]
    total_count: int

class UploadResponse(BaseModel):
    message: str
    doc_id: str
    filename: str
    total_pages: int
    total_characters: int
    pages_with_text: int

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None