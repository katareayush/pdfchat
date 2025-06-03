import logging
import uuid
import pdfplumber
import io
import os
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        # In-memory storage (replace with database in production)
        self.extracted_documents: Dict[str, any] = {}
        
        # Create uploads directory if it doesn't exist
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    def extract_text_by_page(self, pdf_file_bytes: bytes) -> List[Dict]:
        """
        Extract text from PDF page by page
        Returns list of page dictionaries
        """
        extracted_pages = []
        
        try:
            logger.info("📄 Starting PDF text extraction...")
            
            with pdfplumber.open(io.BytesIO(pdf_file_bytes)) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"📄 Processing {total_pages} pages...")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        
                        # Clean and process text
                        if text and text.strip():
                            clean_text = text.strip()
                            extracted_pages.append({
                                "page_number": page_num,
                                "text_chunk": clean_text,
                                "char_count": len(clean_text)
                            })
                            logger.debug(f"✅ Page {page_num}: {len(clean_text)} chars")
                        else:
                            # Still record empty pages for completeness
                            extracted_pages.append({
                                "page_number": page_num,
                                "text_chunk": "",
                                "char_count": 0
                            })
                            logger.debug(f"⚠️ Page {page_num}: empty")
                            
                    except Exception as e:
                        logger.error(f"❌ Error processing page {page_num}: {str(e)}")
                        # Add empty page entry for failed pages
                        extracted_pages.append({
                            "page_number": page_num,
                            "text_chunk": "",
                            "char_count": 0
                        })
                        
        except Exception as e:
            logger.error(f"❌ PDF extraction error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error extracting PDF: {str(e)}")
            
        logger.info(f"✅ Text extraction completed: {len(extracted_pages)} pages processed")
        return extracted_pages
    
    def save_pdf_file(self, pdf_bytes: bytes, doc_id: str, filename: str) -> Path:
        """Save PDF file to uploads directory"""
        file_path = self.upload_dir / f"{doc_id}_{filename}"
        try:
            with open(file_path, "wb") as f:
                f.write(pdf_bytes)
            logger.info(f"✅ PDF saved: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"❌ Error saving PDF: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to save PDF file.")
    
    def process_pdf_without_embeddings(self, pdf_bytes: bytes, filename: str) -> any:
        """
        Process PDF: extract text and store document data (WITHOUT embeddings)
        This is the fast part that returns immediately to frontend
        """
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        logger.info(f"🆔 Generated doc ID: {doc_id}")
        
        # Extract text by page
        extracted_pages = self.extract_text_by_page(pdf_bytes)
        
        # Save PDF file
        self.save_pdf_file(pdf_bytes, doc_id, filename)
        
        # Convert to PageData objects
        from app.models.pdf_models import PageData, DocumentDetail
        page_objects = [PageData(**page) for page in extracted_pages]
        
        # Create document detail
        document_detail = DocumentDetail(
            doc_id=doc_id,
            filename=filename,
            total_pages=len(page_objects),
            pages=page_objects,
            total_chars=sum(page.char_count for page in page_objects),
            upload_time=datetime.utcnow()
        )
        
        # Store in memory
        self.extracted_documents[doc_id] = document_detail
        
        logger.info(f"✅ Document processed (no embeddings yet): {doc_id}")
        return document_detail
    
    def generate_embeddings_for_document(self, doc_id: str, pages: List[Dict]) -> Dict[str, int]:
        """
        Generate embeddings for a document's pages (CPU-intensive, runs in background)
        """
        try:
            logger.info(f"🤖 Starting embedding generation for doc {doc_id}")
            
            # Import here to avoid circular imports
            from app.services.embedding_service import embedding_service
            
            # Check if embedding service is available
            if embedding_service is None:
                logger.error("❌ Embedding service not available")
                return {}
            
            # Generate embeddings (this is the slow part)
            page_embeddings = embedding_service.add_document_embeddings(doc_id, pages)
            
            logger.info(f"✅ Embeddings generated for doc {doc_id}: {len(page_embeddings)} embeddings")
            return page_embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed for doc {doc_id}: {str(e)}")
            return {}  # Return empty dict instead of raising exception
    
    def get_document(self, doc_id: str) -> Optional[any]:
        """Get document by ID"""
        return self.extracted_documents.get(doc_id)
    
    def get_page(self, doc_id: str, page_number: int) -> Optional[any]:
        """Get specific page from document"""
        document = self.get_document(doc_id)
        if not document:
            return None
        
        for page in document.pages:
            if page.page_number == page_number:
                return page
        return None
    
    def list_documents(self) -> List[any]:
        """List all documents"""
        from app.models.pdf_models import DocumentResponse
        return [
            DocumentResponse(
                doc_id=doc_id,
                filename=doc.filename,
                total_pages=doc.total_pages,
                total_characters=doc.total_chars,
                pages_with_text=len([p for p in doc.pages if p.char_count > 0]),
                upload_time=doc.upload_time
            )
            for doc_id, doc in self.extracted_documents.items()
        ]
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document and associated file"""
        if doc_id not in self.extracted_documents:
            return False
        
        try:
            # Remove from memory
            document = self.extracted_documents.pop(doc_id)
            
            # Remove embeddings
            try:
                from app.services.embedding_service import embedding_service
                if embedding_service:
                    removed_count = embedding_service.remove_document_embeddings(doc_id)
                    logger.info(f"🗑️ Removed {removed_count} embeddings for document {doc_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not remove embeddings for {doc_id}: {str(e)}")
            
            # Remove file if it exists
            file_pattern = f"{doc_id}_*"
            for file_path in self.upload_dir.glob(file_pattern):
                file_path.unlink()
                logger.info(f"🗑️ Removed file: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting document {doc_id}: {str(e)}")
            return False
    
    def validate_pdf_file(self, file_size: int, filename: str) -> None:
        """Validate PDF file before processing"""
        # Check file extension
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Check if file is empty
        if not file_size:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Check file size (limit to 50MB)
        MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "26214400"))  # Default 25MB
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )

# Create a singleton instance
pdf_service = PDFService()