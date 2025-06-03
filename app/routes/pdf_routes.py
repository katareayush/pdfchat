from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
import logging
import asyncio
import time
import os  # ← MISSING IMPORT ADDED
from concurrent.futures import ThreadPoolExecutor
from app.services.pdf_service import pdf_service
from app.models.pdf_models import (
    DocumentDetail, 
    DocumentList, 
    DocumentResponse, 
    PageData, 
    UploadResponse,
    ErrorResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and extract text from PDF file with async embedding generation
    """
    upload_start_time = time.time()
    try:
        logger.info(f"📤 Starting upload: {file.filename}")
        # Validate file
        try:
            pdf_service.validate_pdf_file(file.size, file.filename)
        except Exception as ve:
            logger.error(f"❌ Validation error: {str(ve)}")
            raise HTTPException(status_code=400, detail=f"Invalid PDF: {str(ve)}")
        # Read file contents
        logger.info("📖 Reading file contents...")
        pdf_bytes = await file.read()
        if not pdf_bytes or len(pdf_bytes) == 0:
            logger.error("❌ Empty file uploaded")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        logger.info(f"✅ File read successfully: {len(pdf_bytes)} bytes")
        # Process PDF (extract text)
        logger.info("📄 Extracting text from PDF...")
        text_start_time = time.time()
        try:
            document = await asyncio.get_event_loop().run_in_executor(
                executor,
                pdf_service.process_pdf_without_embeddings,
                pdf_bytes,
                file.filename
            )
        except Exception as pe:
            logger.error(f"❌ PDF processing error: {str(pe)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error extracting text: {str(pe)}")
        text_time = time.time() - text_start_time
        logger.info(f"✅ Text extraction completed in {text_time:.2f}s")
        # Prepare response
        upload_time = time.time() - upload_start_time
        logger.info(f"🚀 Upload response sent in {upload_time:.2f}s")
        response = UploadResponse(
            message="PDF uploaded and text extracted successfully",
            doc_id=document.doc_id,
            filename=document.filename,
            total_pages=document.total_pages,
            total_characters=document.total_chars,
            pages_with_text=len([p for p in document.pages if p.char_count > 0])
        )
        # Generate embeddings in background (don't wait for this)
        try:
            asyncio.create_task(generate_embeddings_background(document.doc_id, document.pages))
        except Exception as e:
            logger.error(f"❌ Failed to schedule background embedding: {str(e)}", exc_info=True)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

async def generate_embeddings_background(doc_id: str, pages: List[PageData]):
    """
    Generate embeddings in the background after upload response is sent
    """
    try:
        from app.services.pdf_service import pdf_service
        # Defensive: ensure pages is a list of dicts
        page_dicts = [p.dict() if hasattr(p, 'dict') else p for p in pages]
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            pdf_service.generate_embeddings_for_document,
            doc_id,
            page_dicts
        )
        logger.info(f"✅ Background embeddings generated for doc {doc_id}: {result}")
    except Exception as e:
        logger.error(f"❌ Error in background embedding generation for doc {doc_id}: {str(e)}", exc_info=True)

@router.get("/document/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: str):
    """
    Retrieve extracted document data by ID
    """
    document = pdf_service.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document

@router.get("/document/{doc_id}/page/{page_number}", response_model=PageData)
async def get_page(doc_id: str, page_number: int):
    """
    Get specific page content from a document
    """
    page_data = pdf_service.get_page(doc_id, page_number)
    if not page_data:
        raise HTTPException(status_code=404, detail=f"Page {page_number} not found in document {doc_id}")
    
    return page_data

@router.get("/documents", response_model=DocumentList)
async def list_documents():
    """
    List all uploaded documents
    """
    documents = pdf_service.list_documents()
    return DocumentList(
        documents=documents,
        total_count=len(documents)
    )

@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document and its associated file
    """
    success = pdf_service.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": f"Document deleted successfully"}

@router.get("/document/{doc_id}/summary")
async def get_document_summary(doc_id: str):
    """
    Get a summary of document statistics
    """
    document = pdf_service.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    pages_with_text = [p for p in document.pages if p.char_count > 0]
    empty_pages = [p for p in document.pages if p.char_count == 0]
    
    return {
        "doc_id": document.doc_id,
        "filename": document.filename,
        "total_pages": document.total_pages,
        "pages_with_text": len(pages_with_text),
        "empty_pages": len(empty_pages),
        "total_characters": document.total_chars,
        "average_chars_per_page": document.total_chars / len(pages_with_text) if pages_with_text else 0,
        "upload_time": document.upload_time
    }

@router.get("/document/{doc_id}/embedding-status")
async def get_embedding_status(doc_id: str):
    """
    Check if embeddings are ready for a document
    """
    try:
        from app.services.embedding_service import embedding_service
        
        # Check if embedding service is available
        if embedding_service is None:
            return {
                "doc_id": doc_id,
                "embeddings_ready": False,
                "embedding_count": 0,
                "can_search": False,
                "error": "Embedding service not available"
            }
        
        # Check if document has embeddings
        has_embeddings = False
        embedding_count = 0
        
        for embedding_id, metadata in embedding_service.chunk_metadata.items():
            if metadata.get("doc_id") == doc_id:
                has_embeddings = True
                embedding_count += 1
        
        return {
            "doc_id": doc_id,
            "embeddings_ready": has_embeddings,
            "embedding_count": embedding_count,
            "can_search": has_embeddings
        }
        
    except Exception as e:
        logger.error(f"Error checking embedding status: {str(e)}")
        return {
            "doc_id": doc_id,
            "embeddings_ready": False,
            "embedding_count": 0,
            "can_search": False,
            "error": str(e)
        }