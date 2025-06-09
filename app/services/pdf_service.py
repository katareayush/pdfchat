import logging
import uuid
import pdfplumber
import io
import os
import re
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        self.extracted_documents: Dict[str, any] = {}
        
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    def extract_text_by_page(self, pdf_file_bytes: bytes) -> List[Dict]:
        """Extract text with improved chunking that preserves clause structure"""
        extracted_pages = []
        
        try:
            logger.info("Starting PDF text extraction with improved chunking...")
            
            with pdfplumber.open(io.BytesIO(pdf_file_bytes)) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing {total_pages} pages...")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        
                        if text and text.strip():
                            clean_text = text.strip()
                            
                            # Check if this page contains clause structures and is long
                            chunks = self._smart_chunk_page(clean_text, page_num)
                            
                            if chunks and len(chunks) > 1:
                                # Add each chunk as a separate "sub-page"
                                for i, chunk in enumerate(chunks):
                                    extracted_pages.append({
                                        "page_number": page_num,  # Keep original page number
                                        "text_chunk": chunk,
                                        "char_count": len(chunk),
                                        "chunk_id": f"{page_num}.{i+1}"  # Add chunk identifier
                                    })
                                    logger.debug(f"Page {page_num}.{i+1}: {len(chunk)} chars")
                            else:
                                # Single chunk for the whole page
                                extracted_pages.append({
                                    "page_number": page_num,
                                    "text_chunk": clean_text,
                                    "char_count": len(clean_text)
                                })
                                logger.debug(f"Page {page_num}: {len(clean_text)} chars")
                        else:
                            extracted_pages.append({
                                "page_number": page_num,
                                "text_chunk": "",
                                "char_count": 0
                            })
                            logger.debug(f"Page {page_num}: empty")
                            
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {str(e)}")
                        extracted_pages.append({
                            "page_number": page_num,
                            "text_chunk": "",
                            "char_count": 0
                        })
                        
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error extracting PDF: {str(e)}")
            
        logger.info(f"Text extraction completed: {len(extracted_pages)} chunks processed")
        return extracted_pages
    
    def _smart_chunk_page(self, text: str, page_num: int) -> List[str]:
        """Smart chunking that preserves clause and section structure"""
        
        # Don't chunk if text is reasonably short
        if len(text) <= 5000:  # Increased from 3000 to 5000
            return [text]
        
        # Look for clause/section patterns
        clause_pattern = r'^\s*(\d+(?:\.\d+)*)\s+([A-Z][^0-9\n]*)'
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_size = 4000  # Increased from 2500 to 4000
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line starts a new clause/section
            clause_match = re.match(clause_pattern, line_stripped)
            
            if clause_match and current_chunk and current_length > 1000:
                # Start new chunk at clause boundary
                chunk_text = '\n'.join(current_chunk)
                chunks.append(chunk_text)
                logger.debug(f"Created chunk at clause boundary: {clause_match.group(1)} (length: {len(chunk_text)})")
                current_chunk = [line]
                current_length = len(line)
            else:
                current_chunk.append(line)
                current_length += len(line) + 1
                
                # If chunk gets too large, split at sentence boundary
                if current_length > max_chunk_size:
                    # Find a good break point (sentence end)
                    chunk_text = '\n'.join(current_chunk)
                    last_sentence = chunk_text.rfind('. ')
                    
                    if last_sentence > len(chunk_text) * 0.7:  # Good break point found
                        chunks.append(chunk_text[:last_sentence + 1])
                        remaining = chunk_text[last_sentence + 2:].strip()
                        current_chunk = [remaining] if remaining else []
                        current_length = len(remaining) if remaining else 0
                        logger.debug(f"Split chunk at sentence boundary (length: {last_sentence + 1})")
                    else:
                        # No good break point, force split
                        chunks.append(chunk_text)
                        current_chunk = []
                        current_length = 0
                        logger.debug(f"Force split chunk (length: {len(chunk_text)})")
        
        # Add remaining content
        if current_chunk:
            final_chunk = '\n'.join(current_chunk)
            chunks.append(final_chunk)
        
        # Filter out very small chunks (merge with previous)
        filtered_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) < 200 and filtered_chunks:
                # Merge with previous chunk
                filtered_chunks[-1] += '\n' + chunk
                logger.debug(f"Merged small chunk with previous")
            else:
                filtered_chunks.append(chunk)
        
        # Log final chunking result
        if len(filtered_chunks) > 1:
            logger.info(f"Page {page_num} split into {len(filtered_chunks)} chunks")
            for i, chunk in enumerate(filtered_chunks):
                logger.debug(f"  Chunk {i+1}: {len(chunk)} chars")
        
        return filtered_chunks if len(filtered_chunks) > 1 else [text]
    
    def save_pdf_file(self, pdf_bytes: bytes, doc_id: str, filename: str) -> Path:
        file_path = self.upload_dir / f"{doc_id}_{filename}"
        try:
            with open(file_path, "wb") as f:
                f.write(pdf_bytes)
            logger.info(f"PDF saved: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving PDF: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to save PDF file.")
    
    def process_pdf_without_embeddings(self, pdf_bytes: bytes, filename: str) -> any:
        doc_id = str(uuid.uuid4())
        logger.info(f"Generated doc ID: {doc_id}")
        
        extracted_pages = self.extract_text_by_page(pdf_bytes)
        
        self.save_pdf_file(pdf_bytes, doc_id, filename)
        
        from app.models.pdf_models import PageData, DocumentDetail
        page_objects = [PageData(**page) for page in extracted_pages]
        
        document_detail = DocumentDetail(
            doc_id=doc_id,
            filename=filename,
            total_pages=len(page_objects),
            pages=page_objects,
            total_chars=sum(page.char_count for page in page_objects),
            upload_time=datetime.utcnow()
        )
        
        self.extracted_documents[doc_id] = document_detail
        
        logger.info(f"Document processed (no embeddings yet): {doc_id}")
        return document_detail
    
    def generate_embeddings_for_document(self, doc_id: str, pages: List[Dict]) -> Dict[str, int]:
        try:
            logger.info(f"Starting embedding generation for doc {doc_id}")
            
            from app.services.embedding_service import embedding_service
            
            if embedding_service is None:
                logger.error("Embedding service not available")
                return {}
            
            page_embeddings = embedding_service.add_document_embeddings(doc_id, pages)
            
            logger.info(f"Embeddings generated for doc {doc_id}: {len(page_embeddings)} embeddings")
            return page_embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed for doc {doc_id}: {str(e)}")
            return {}
    
    def get_document(self, doc_id: str) -> Optional[any]:
        return self.extracted_documents.get(doc_id)
    
    def get_page(self, doc_id: str, page_number: int) -> Optional[any]:
        document = self.get_document(doc_id)
        if not document:
            return None
        
        for page in document.pages:
            if page.page_number == page_number:
                return page
        return None
    
    def list_documents(self) -> List[any]:
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
        if doc_id not in self.extracted_documents:
            return False
        
        try:
            document = self.extracted_documents.pop(doc_id)
            
            try:
                from app.services.embedding_service import embedding_service
                if embedding_service:
                    removed_count = embedding_service.remove_document_embeddings(doc_id)
                    logger.info(f"Removed {removed_count} embeddings for document {doc_id}")
            except Exception as e:
                logger.warning(f"Could not remove embeddings for {doc_id}: {str(e)}")
            
            file_pattern = f"{doc_id}_*"
            for file_path in self.upload_dir.glob(file_pattern):
                file_path.unlink()
                logger.info(f"Removed file: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def validate_pdf_file(self, file_size: int, filename: str) -> None:
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        if not file_size:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "26214400"))
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )

pdf_service = PDFService()