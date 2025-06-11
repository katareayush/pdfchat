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
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
import fitz  

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
                            
                            needs_ocr = self._needs_ocr_fallback(clean_text, page_num)
                            
                            if needs_ocr:
                                logger.info(f"Page {page_num}: Poor quality text detected, applying OCR...")
                                ocr_text = self._apply_ocr_to_page(pdf_file_bytes, page_num - 1)  # 0-indexed
                                if ocr_text and len(ocr_text.strip()) > len(clean_text.strip()):
                                    logger.info(f"Page {page_num}: OCR improved text quality ({len(clean_text)} -> {len(ocr_text)} chars)")
                                    clean_text = ocr_text
                                else:
                                    logger.info(f"Page {page_num}: Keeping original text extraction")
                            
                            chunks = self._smart_chunk_page(clean_text, page_num)
                            
                            if chunks and len(chunks) > 1:
                                for i, chunk in enumerate(chunks):
                                    extracted_pages.append({
                                        "page_number": page_num,
                                        "text_chunk": chunk,
                                        "char_count": len(chunk),
                                        "chunk_id": f"{page_num}.{i+1}",
                                        "extraction_method": "ocr" if needs_ocr else "direct"  
                                    })
                                    logger.debug(f"Page {page_num}.{i+1}: {len(chunk)} chars")
                            else:
                                extracted_pages.append({
                                    "page_number": page_num,
                                    "text_chunk": clean_text,
                                    "char_count": len(clean_text),
                                    "extraction_method": "ocr" if needs_ocr else "direct"
                                })
                                logger.debug(f"Page {page_num}: {len(clean_text)} chars")
                        else:
                            logger.info(f"Page {page_num}: No text found, trying OCR...")
                            ocr_text = self._apply_ocr_to_page(pdf_file_bytes, page_num - 1)
                            
                            if ocr_text and ocr_text.strip():
                                logger.info(f"Page {page_num}: OCR recovered {len(ocr_text)} characters")
                                extracted_pages.append({
                                    "page_number": page_num,
                                    "text_chunk": ocr_text,
                                    "char_count": len(ocr_text),
                                    "extraction_method": "ocr"
                                })
                            else:
                                extracted_pages.append({
                                    "page_number": page_num,
                                    "text_chunk": "",
                                    "char_count": 0,
                                    "extraction_method": "none"
                                })
                                logger.debug(f"Page {page_num}: empty")
                            
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {str(e)}")
                        extracted_pages.append({
                            "page_number": page_num,
                            "text_chunk": "",
                            "char_count": 0,
                            "extraction_method": "error"
                        })
                        
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error extracting PDF: {str(e)}")
            
        logger.info(f"Text extraction completed: {len(extracted_pages)} chunks processed")
        return extracted_pages
    
    def _needs_ocr_fallback(self, extracted_text: str, page_num: int) -> bool:
        """
        Determine if OCR fallback is needed based on text quality
        """
        if not extracted_text or len(extracted_text.strip()) == 0:
            return True
        
        chars_count = len(extracted_text)
        if chars_count < 50:  
            return True
        
        readable_chars = sum(1 for c in extracted_text if c.isalnum() or c.isspace() or c in '.,!?;:-()[]{}')
        readable_ratio = readable_chars / chars_count if chars_count > 0 else 0
        
        if readable_ratio < 0.7:  
            logger.debug(f"Page {page_num}: Low readable ratio {readable_ratio:.2f}")
            return True
        
        words = extracted_text.split()
        if len(words) > 5:
            garbled_words = 0
            for word in words[:20]:  
                clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
                if len(clean_word) > 3:
                    consonant_pattern = re.search(r'[bcdfghjklmnpqrstvwxyz]{4,}', clean_word)
                    if consonant_pattern:
                        garbled_words += 1
            
            garbled_ratio = garbled_words / min(20, len(words))
            if garbled_ratio > 0.3: 
                logger.debug(f"Page {page_num}: High garbled word ratio {garbled_ratio:.2f}")
                return True
        
        artifact_patterns = [
            r'[^\w\s]{10,}',   
            r'\s{20,}',       
            r'(.)\1{10,}'     
        ]
        
        for pattern in artifact_patterns:
            if re.search(pattern, extracted_text):
                logger.debug(f"Page {page_num}: Found extraction artifacts")
                return True
        
        return False
    
    def _apply_ocr_to_page(self, pdf_bytes: bytes, page_index: int) -> str:
        """
        Apply OCR to a specific page using PyMuPDF + Tesseract
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            if page_index >= len(doc):
                logger.error(f"Page index {page_index} out of range")
                return ""
            
            page = doc[page_index]
            
            mat = fitz.Matrix(2.0, 2.0)  # 2x scaling for better OCR
            pix = page.get_pixmap(matrix=mat)
            
            img_data = pix.tobytes("ppm")
            image = Image.open(io.BytesIO(img_data))
            
            image = self._preprocess_image_for_ocr(image)
            
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;:()[]{}"-+=*/%@#$&'
            
            try:
                ocr_text = pytesseract.image_to_string(
                    image, 
                    config=custom_config,
                    lang='eng'
                )
                
                doc.close()
                return ocr_text.strip()
                
            except Exception as ocr_error:
                logger.error(f"Tesseract OCR failed: {str(ocr_error)}")
                try:
                    ocr_text = pytesseract.image_to_string(image)
                    doc.close()
                    return ocr_text.strip()
                except Exception as fallback_error:
                    logger.error(f"Fallback OCR also failed: {str(fallback_error)}")
                    doc.close()
                    return ""
                    
        except Exception as e:
            logger.error(f"OCR processing failed for page {page_index}: {str(e)}")
            return ""
    
    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results
        """
        try:
            if image.mode != 'L':
                image = image.convert('L')
            
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            width, height = image.size
            if width < 1000 or height < 1000:
                scale_factor = max(1000 / width, 1000 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return image  
    
    def _smart_chunk_page(self, text: str, page_num: int) -> List[str]:
        """Smart chunking that preserves clause and section structure - UNCHANGED"""
        
        if len(text) <= 5000:
            return [text]
        
        clause_pattern = r'^\s*(\d+(?:\.\d+)*)\s+([A-Z][^0-9\n]*)'
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_size = 4000
        
        for line in lines:
            line_stripped = line.strip()
            
            clause_match = re.match(clause_pattern, line_stripped)
            
            if clause_match and current_chunk and current_length > 1000:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(chunk_text)
                logger.debug(f"Created chunk at clause boundary: {clause_match.group(1)} (length: {len(chunk_text)})")
                current_chunk = [line]
                current_length = len(line)
            else:
                current_chunk.append(line)
                current_length += len(line) + 1
                
                if current_length > max_chunk_size:
                    chunk_text = '\n'.join(current_chunk)
                    last_sentence = chunk_text.rfind('. ')
                    
                    if last_sentence > len(chunk_text) * 0.7:
                        chunks.append(chunk_text[:last_sentence + 1])
                        remaining = chunk_text[last_sentence + 2:].strip()
                        current_chunk = [remaining] if remaining else []
                        current_length = len(remaining) if remaining else 0
                        logger.debug(f"Split chunk at sentence boundary (length: {last_sentence + 1})")
                    else:
                        chunks.append(chunk_text)
                        current_chunk = []
                        current_length = 0
                        logger.debug(f"Force split chunk (length: {len(chunk_text)})")
        
        if current_chunk:
            final_chunk = '\n'.join(current_chunk)
            chunks.append(final_chunk)
        
        filtered_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) < 200 and filtered_chunks:
                filtered_chunks[-1] += '\n' + chunk
                logger.debug(f"Merged small chunk with previous")
            else:
                filtered_chunks.append(chunk)
        
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