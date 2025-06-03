"""
Pydantic models for the PDF RAG API
"""

from .pdf_models import (
    PageData,
    DocumentResponse,
    DocumentDetail,
    DocumentList,
    UploadResponse,
    ErrorResponse
)

from .search_models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    EmbeddingStats
)

__all__ = [
    "PageData",
    "DocumentResponse", 
    "DocumentDetail",
    "DocumentList",
    "UploadResponse",
    "ErrorResponse",
    "SearchRequest",
    "SearchResponse", 
    "SearchResult",
    "EmbeddingStats"
]