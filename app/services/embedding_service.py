import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
from pathlib import Path
from datetime import datetime
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunk_metadata = {}
        self.embedding_dim = 384
        self.next_embedding_id = 0
        self.model_loaded = False
        
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_client = None
        self.anthropic_enabled = False
        
        self.storage_dir = Path("data")
        self.storage_dir.mkdir(exist_ok=True)
        self.index_path = self.storage_dir / "faiss_index.bin"
        self.metadata_path = self.storage_dir / "chunk_metadata.json"
        
        self._initialize()
    
    def _initialize(self):
        try:
            self._load_index()
            self._initialize_anthropic()
        except Exception as e:
            logger.error(f"Error initializing embedding service: {str(e)}")
    
    def _initialize_anthropic(self):
        if not ANTHROPIC_AVAILABLE or not self.anthropic_api_key:
            logger.warning("Anthropic API key not found or module not available")
            return
        
        try:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            test_response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Test"}]
            )
            
            if test_response and test_response.content:
                self.anthropic_enabled = True
                logger.info("Anthropic API initialized successfully with Haiku")
                
        except Exception as e:
            logger.error(f"Error initializing Anthropic: {str(e)}")
            self.anthropic_enabled = False
    
    def _call_anthropic(self, prompt: str, max_tokens: int = 800, model: str = "claude-3-haiku-20240307") -> str:
        """Use Haiku for fast responses"""
        if not self.anthropic_enabled:
            return ""
        
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            if response and response.content and len(response.content) > 0:
                return response.content[0].text.strip()
            return ""
                
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            return ""
    
    def _ensure_model_loaded(self):
        if not self.model_loaded:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.model_loaded = True
                logger.info(f"Model loaded: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def _load_index(self):
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.chunk_metadata = json.load(f)
                
                if self.chunk_metadata:
                    max_id = max(int(k) for k in self.chunk_metadata.keys())
                    self.next_embedding_id = max_id + 1
                
                logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
            else:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Error loading/creating index: {str(e)}")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
    
    def _save_index(self):
        try:
            temp_index_path = self.index_path.with_suffix('.tmp')
            temp_metadata_path = self.metadata_path.with_suffix('.tmp')
            
            faiss.write_index(self.index, str(temp_index_path))
            with open(temp_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)
            
            temp_index_path.replace(self.index_path)
            temp_metadata_path.replace(self.metadata_path)
            
            logger.info(f"Index saved successfully ({self.index.ntotal} vectors)")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            self._ensure_model_loaded()
            
            if not texts:
                return np.array([]).reshape(0, self.embedding_dim)
            
            embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def add_document_embeddings(self, doc_id: str, pages: List[Dict]) -> Dict[str, int]:
        try:
            logger.info(f"Adding embeddings for document {doc_id}")
            
            non_empty_pages = [page for page in pages if page.get('text_chunk', '').strip()]
            
            if not non_empty_pages:
                logger.warning(f"No non-empty pages found for document {doc_id}")
                return {}
            
            texts = [page['text_chunk'] for page in non_empty_pages]
            embeddings = self.generate_embeddings(texts)
            
            if embeddings.size == 0:
                logger.warning(f"No embeddings generated for document {doc_id}")
                return {}
            
            self.index.add(embeddings.astype('float32'))
            
            page_to_embedding_id = {}
            for i, page in enumerate(non_empty_pages):
                embedding_id = self.next_embedding_id + i
                
                self.chunk_metadata[str(embedding_id)] = {
                    "doc_id": doc_id,
                    "page_number": page['page_number'],
                    "text_chunk": page['text_chunk'],
                    "char_count": page.get('char_count', len(page['text_chunk'])),
                    "added_at": datetime.now().isoformat()
                }
                
                page_to_embedding_id[page['page_number']] = embedding_id
            
            self.next_embedding_id += len(non_empty_pages)
            self._save_index()
            
            logger.info(f"Successfully added {len(non_empty_pages)} embeddings for document {doc_id}")
            return page_to_embedding_id
            
        except Exception as e:
            logger.error(f"Error adding document embeddings for {doc_id}: {str(e)}")
            return {}
    
    def _basic_search(self, query: str, top_k: int, score_threshold: float, doc_id: str = None) -> List[Dict]:
        try:
            if not self.model_loaded:
                self._ensure_model_loaded()
                
            if self.index.ntotal == 0:
                logger.warning("Index is empty - no documents have been indexed yet")
                return []
            
            query_embedding = self.generate_embeddings([query])
            
            if query_embedding.size == 0:
                return []
            
            search_k = min(top_k * 4, self.index.ntotal) if doc_id else min(top_k * 2, self.index.ntotal)
            scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            results = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < score_threshold:
                    continue
                    
                embedding_id = str(idx)
                if embedding_id in self.chunk_metadata:
                    metadata = self.chunk_metadata[embedding_id]
                    
                    if doc_id and metadata.get("doc_id") != doc_id:
                        continue
                    
                    results.append({
                        "page": metadata["page_number"],
                        "context": metadata["text_chunk"],
                        "doc_id": metadata["doc_id"],
                        "score": float(score),
                        "char_count": metadata.get("char_count", 0)
                    })
                    
                    if len(results) >= top_k:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Error in basic search: {str(e)}")
            return []
    
    def analyze_query_context(self, query: str, search_results: List[Dict]) -> Dict:
        """Analyze query context with Claude Haiku for speed"""
        if not self.anthropic_enabled or not search_results:
            return {
                "can_answer": len(search_results) > 0,
                "confidence": 0.5 if search_results else 0.0,
                "relevance": "unknown",
                "contextual_answer": None,
                "answer_type": "no_analysis"
            }
        
        try:
            combined_context = "\n\n".join([
                f"Page {result['page']}:\n{result['context']}"
                for result in search_results[:8]
            ])
            
            if len(combined_context) > 4000:
                combined_context = combined_context[:4000] + "..."
            
            relevance_prompt = f"""Analyze if the provided document content contains information that can answer the user's question.

Question: "{query}"

Document Content:
{combined_context}

Respond with:
- "RELEVANT" if the content directly answers the question
- "PARTIAL" if the content has some related information
- "NOT_RELEVANT" if the content has no meaningful connection

Response:"""
            
            relevance = self._call_anthropic(relevance_prompt, max_tokens=20)
            relevance = relevance.upper() if relevance else "NOT_RELEVANT"
            
            if relevance == "NOT_RELEVANT":
                return {
                    "can_answer": False,
                    "confidence": 0.1,
                    "relevance": "not_relevant",
                    "contextual_answer": "This information is not available in the document.",
                    "answer_type": "not_available"
                }
            
            answer_prompt = f"""Based on the document content provided, answer the user's question using only the information explicitly present.

Question: "{query}"

Document Content:
{combined_context}

Instructions:
1. Use ONLY information explicitly stated in the document
2. Quote exact text when possible
3. Include specific page numbers when referencing information
4. If information is incomplete, state what IS available and what is missing
5. Be thorough and accurate

Answer:"""
            
            contextual_answer = self._call_anthropic(answer_prompt, max_tokens=800)
            
            confidence = self._calculate_confidence(relevance, contextual_answer, search_results)
            answer_type = self._determine_answer_type(contextual_answer, relevance)
            
            return {
                "can_answer": relevance in ["RELEVANT", "PARTIAL"],
                "confidence": confidence,
                "relevance": relevance.lower(),
                "contextual_answer": contextual_answer,
                "answer_type": answer_type
            }
            
        except Exception as e:
            logger.error(f"Error in context analysis: {str(e)}")
            return {
                "can_answer": False,
                "confidence": 0.0,
                "relevance": "error",
                "contextual_answer": "Unable to analyze the document content.",
                "answer_type": "error"
            }
    
    def _calculate_confidence(self, relevance: str, answer: str, search_results: list) -> float:
        """Calculate confidence score"""
        confidence = 0.0
        
        if relevance == "RELEVANT":
            confidence += 0.8
        elif relevance == "PARTIAL":
            confidence += 0.5
        else:
            confidence += 0.2
        
        if search_results:
            avg_score = sum(r.get('score', 0) for r in search_results[:3]) / min(3, len(search_results))
            confidence += min(0.2, avg_score * 0.4)
        
        uncertainty_indicators = ["not clear", "unclear", "may be", "possibly", "unable to determine"]
        if any(indicator in answer.lower() for indicator in uncertainty_indicators):
            confidence *= 0.7
        
        if any(word in answer.lower() for word in ["page", "clause", "section"]) and any(char.isdigit() for char in answer):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _determine_answer_type(self, answer: str, relevance: str) -> str:
        """Determine answer type"""
        if relevance == "NOT_RELEVANT":
            return "not_available"
        elif relevance == "PARTIAL":
            return "partial"
        elif relevance == "RELEVANT":
            return "comprehensive"
        else:
            return "basic"
    
    def search_with_context_analysis(self, query: str, top_k: int = 5, doc_id: str = None) -> Dict:
        """Main search method with context analysis and performance tracking"""
        try:
            start_time = time.time()
            
            specific_clause = self._extract_clause_number(query)
            
            faiss_start = time.time()
            if specific_clause:
                clause_results = self._search_specific_clause(specific_clause, doc_id)
                if clause_results:
                    search_results = clause_results[:top_k]
                else:
                    search_results = self._basic_search(query, top_k, 0.25, doc_id)
            else:
                search_results = self._basic_search(query, top_k, 0.25, doc_id)
            faiss_time = (time.time() - faiss_start) * 1000
            
            claude_start = time.time()
            context_analysis = self.analyze_query_context(query, search_results)
            claude_time = (time.time() - claude_start) * 1000
            
            total_time = (time.time() - start_time) * 1000
            
            return {
                "query": query,
                "results": search_results,
                "total_found": len(search_results),
                "faiss_time_ms": round(faiss_time, 1),
                "claude_time_ms": round(claude_time, 1),
                "total_time_ms": round(total_time, 1),
                "llm_analysis": {
                    "status": "completed" if self.anthropic_enabled else "unavailable",
                    "relevance_check": context_analysis.get("relevance", "unknown"),
                    "anthropic_initialized": self.anthropic_enabled
                },
                "contextual_answer": context_analysis.get("contextual_answer", ""),
                "confidence_score": context_analysis.get("confidence", 0.0),
                "answer_type": context_analysis.get("answer_type", "basic"),
                "can_answer": context_analysis.get("can_answer", False),
                "anthropic_enabled": self.anthropic_enabled,
                "relevance": context_analysis.get("relevance", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Error in search with context analysis: {str(e)}")
            search_results = self._basic_search(query, top_k, 0.25, doc_id)
            return {
                "query": query,
                "results": search_results,
                "total_found": len(search_results),
                "llm_analysis": {"status": "error", "error": str(e)},
                "contextual_answer": "Unable to provide contextual analysis.",
                "confidence_score": 0.0,
                "answer_type": "error",
                "can_answer": len(search_results) > 0,
                "anthropic_enabled": False
            }
    
    def _extract_clause_number(self, query: str) -> str:
        """Extract clause numbers from query"""
        patterns = [
            r'(?:clause|section|paragraph|item|article|part|chapter)\s*(\d+(?:\.\d+)*)',
            r'(\d+\.\d+(?:\.\d+)*)',
            r'(?:^|\s)(\d{1,3})\s+(?:[A-Za-z]|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1)
        
        return ""
    
    def _search_specific_clause(self, clause_number: str, doc_id: str = None) -> List[Dict]:
        """Search for specific clauses"""
        try:
            clause_results = []
            
            for embedding_id, metadata in self.chunk_metadata.items():
                if doc_id and metadata.get("doc_id") != doc_id:
                    continue
                
                content = metadata.get("text_chunk", "")
                
                if clause_number in content and self._quick_clause_check(content, clause_number):
                    clause_results.append({
                        "page": metadata["page_number"],
                        "context": content,
                        "doc_id": metadata["doc_id"],
                        "score": 0.95,
                        "char_count": len(content)
                    })
            
            clause_results.sort(key=lambda x: x['page'])
            return clause_results[:5]
            
        except Exception as e:
            logger.error(f"Error in clause search: {str(e)}")
            return []
    
    def _quick_clause_check(self, content: str, clause_number: str) -> bool:
        """Quick clause validation"""
        content_lower = content.lower()
        number_lower = clause_number.lower()
        
        return (
            f"{number_lower} " in content_lower or
            f"clause {number_lower}" in content_lower or
            f"section {number_lower}" in content_lower or
            content_lower.startswith(number_lower + " ")
        )
    
    def search_similar(self, query: str, top_k: int = 5, score_threshold: float = 0.3, 
                      doc_id: str = None, use_claude_enhancement: bool = True) -> List[Dict]:
        try:
            if not self.model_loaded:
                self._ensure_model_loaded()
                
            if self.index.ntotal == 0:
                logger.warning("Index is empty - no documents have been indexed yet")
                return []
            
            return self._basic_search(query, top_k, score_threshold, doc_id)
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return self._basic_search(query, top_k, score_threshold, doc_id)
    
    def remove_document_embeddings(self, doc_id: str) -> int:
        try:
            embeddings_to_remove = []
            for embedding_id, metadata in self.chunk_metadata.items():
                if metadata.get("doc_id") == doc_id:
                    embeddings_to_remove.append(embedding_id)
            
            if not embeddings_to_remove:
                return 0
            
            for embedding_id in embeddings_to_remove:
                del self.chunk_metadata[embedding_id]
            
            self._rebuild_index()
            logger.info(f"Removed {len(embeddings_to_remove)} embeddings for document {doc_id}")
            return len(embeddings_to_remove)
            
        except Exception as e:
            logger.error(f"Error removing document embeddings: {str(e)}")
            return 0
    
    def _rebuild_index(self):
        try:
            if not self.chunk_metadata:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.next_embedding_id = 0
                self._save_index()
                return
            
            texts = [metadata.get("text_chunk", "") for metadata in self.chunk_metadata.values()]
            embeddings = self.generate_embeddings(texts)
            
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            if embeddings.size > 0:
                self.index.add(embeddings.astype('float32'))
            
            old_metadata = self.chunk_metadata.copy()
            self.chunk_metadata = {}
            self.next_embedding_id = 0
            
            for i, (old_id, metadata) in enumerate(old_metadata.items()):
                self.chunk_metadata[str(i)] = metadata
                self.next_embedding_id = i + 1
            
            self._save_index()
            logger.info(f"Index rebuilt with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.chunk_metadata = {}
            self.next_embedding_id = 0
    
    def get_stats(self) -> Dict:
        try:
            storage_size = 0
            if self.index_path.exists():
                storage_size = self.index_path.stat().st_size / (1024 * 1024)
                
            return {
                "total_embeddings": self.index.ntotal if self.index else 0,
                "embedding_dimension": self.embedding_dim,
                "model_name": self.model_name,
                "model_loaded": self.model_loaded,
                "unique_documents": len(set(
                    metadata.get("doc_id", "") 
                    for metadata in self.chunk_metadata.values()
                )),
                "storage_size_mb": storage_size,
                "metadata_count": len(self.chunk_metadata),
                "index_exists": self.index is not None,
                "anthropic_enabled": self.anthropic_enabled
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "total_embeddings": 0,
                "embedding_dimension": self.embedding_dim,
                "model_name": self.model_name,
                "model_loaded": self.model_loaded,
                "unique_documents": 0,
                "storage_size_mb": 0,
                "metadata_count": 0,
                "index_exists": False,
                "anthropic_enabled": False,
                "error": str(e)
            }
    
    def search_document_only(self, query: str, doc_id: str, top_k: int = 5, 
                           score_threshold: float = 0.3, use_claude: bool = True) -> List[Dict]:
        if not doc_id:
            raise ValueError("doc_id is required for document-specific search")
        
        return self.search_similar(query, top_k, score_threshold, doc_id, use_claude)
    
    def get_document_stats(self, doc_id: str) -> Dict:
        if not doc_id:
            return {"error": "doc_id is required"}
        
        doc_embeddings = 0
        doc_pages = set()
        total_chars = 0
        
        for embedding_id, metadata in self.chunk_metadata.items():
            if metadata.get("doc_id") == doc_id:
                doc_embeddings += 1
                doc_pages.add(metadata.get("page_number"))
                total_chars += metadata.get("char_count", 0)
        
        return {
            "doc_id": doc_id,
            "embeddings_count": doc_embeddings,
            "pages_with_embeddings": len(doc_pages),
            "total_characters": total_chars,
            "searchable": doc_embeddings > 0
        }
    
    def list_searchable_documents(self) -> List[Dict]:
        doc_stats = {}
        
        for embedding_id, metadata in self.chunk_metadata.items():
            doc_id = metadata.get("doc_id")
            if doc_id:
                if doc_id not in doc_stats:
                    doc_stats[doc_id] = {
                        "doc_id": doc_id,
                        "embeddings_count": 0,
                        "pages": set(),
                        "total_chars": 0
                    }
                
                doc_stats[doc_id]["embeddings_count"] += 1
                doc_stats[doc_id]["pages"].add(metadata.get("page_number"))
                doc_stats[doc_id]["total_chars"] += metadata.get("char_count", 0)
        
        searchable_docs = []
        for doc_id, stats in doc_stats.items():
            searchable_docs.append({
                "doc_id": doc_id,
                "embeddings_count": stats["embeddings_count"],
                "pages_with_embeddings": len(stats["pages"]),
                "total_characters": stats["total_chars"],
                "searchable": True
            })
        
        return searchable_docs

try:
    embedding_service = EmbeddingService()
    logger.info("Embedding service created successfully with Claude Haiku for speed")
except Exception as e:
    logger.error(f"Failed to create embedding service: {str(e)}")
    embedding_service = None