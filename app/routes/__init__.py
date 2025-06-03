"""
API routes for the PDF RAG API
"""

from .pdf_routes import router as pdf_router
from .search_routes import router as search_router

__all__ = [
    "pdf_router",
    "search_router"
]