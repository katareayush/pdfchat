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

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.info("Gemini not available - install google-generativeai for enhanced features")

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunk_metadata = {}
        self.embedding_dim = 384
        self.next_embedding_id = 0
        self.model_loaded = False
        
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = None
        self.gemini_enabled = False
        
        self.storage_dir = Path("data")
        self.storage_dir.mkdir(exist_ok=True)
        self.index_path = self.storage_dir / "faiss_index.bin"
        self.metadata_path = self.storage_dir / "chunk_metadata.json"
        
        self._initialize_lazy()
        self._initialize_gemini()
    
    def _initialize_lazy(self):
        try:
            self._load_index()
        except Exception as e:
            logger.error(f"Error initializing embedding service: {str(e)}")
    
    def _initialize_gemini(self):
        if not GEMINI_AVAILABLE or not self.gemini_api_key:
            logger.info("Gemini API key not found or module not available")
            return
        
        try:
            genai.configure(api_key=self.gemini_api_key)
            
            model_names = [
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro'
            ]
            
            for model_name in model_names:
                try:
                    self.gemini_model = genai.GenerativeModel(model_name)
                    test_response = self.gemini_model.generate_content("Hello")
                    if test_response and test_response.text:
                        self.gemini_enabled = True
                        logger.info(f"Gemini API initialized successfully with model: {model_name}")
                        break
                except Exception as model_error:
                    logger.warning(f"Failed to initialize Gemini model {model_name}: {str(model_error)}")
                    continue
            
            if not self.gemini_enabled:
                logger.error("Failed to initialize any Gemini model")
                
        except Exception as e:
            logger.error(f"Error initializing Gemini: {str(e)}")
            self.gemini_enabled = False
    
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
                    "text_chunk": page['text_chunk'][:1000],
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
        if not self.gemini_enabled or not search_results:
            return {
                "can_answer": len(search_results) > 0,
                "confidence": 0.5 if search_results else 0.0,
                "relevance": "unknown",
                "contextual_answer": None,
                "answer_type": "no_analysis"
            }
        
        try:
            combined_context = "\n\n".join([
                f"Page {result['page']}: {result['context'][:300]}"
                for result in search_results[:5]
            ])
            
            relevance_prompt = f"""
            Analyze if the provided document content can answer the user's question.
            
            Question: "{query}"
            
            Document Content:
            {combined_context[:1200]}
            
            Respond with only:
            - "RELEVANT" if the content contains information that can answer the question
            - "NOT_RELEVANT" if the content does not contain relevant information  
            - "PARTIAL" if the content has some related information but cannot fully answer
            
            Response:"""
            
            relevance_response = self.gemini_model.generate_content(relevance_prompt)
            relevance = relevance_response.text.strip().upper() if relevance_response.text else "NOT_RELEVANT"
            
            logger.info(f"Relevance check for query '{query}': {relevance}")
            
            if relevance == "NOT_RELEVANT":
                return {
                    "can_answer": False,
                    "confidence": 0.1,
                    "relevance": "not_relevant",
                    "contextual_answer": "This information is not available in the document.",
                    "answer_type": "not_available"
                }
            
            answer_prompt = f"""
            Based on the document content provided, answer the user's question accurately and concisely.
            
            Question: "{query}"
            
            Document Content:
            {combined_context[:1500]}
            
            Guidelines:
            1. Only use information explicitly mentioned in the document
            2. If you cannot find a complete answer, say "The document provides partial information about this topic"
            3. Be specific and cite page numbers when possible
            4. Keep the answer under 200 words
            5. If the information is unclear or contradictory, mention that
            
            Answer:"""
            
            answer_response = self.gemini_model.generate_content(answer_prompt)
            contextual_answer = answer_response.text.strip() if answer_response.text else ""
            
            confidence = self._calculate_answer_confidence(relevance, contextual_answer, search_results)
            answer_type = self._determine_answer_type(contextual_answer, relevance, confidence)
            
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
                "contextual_answer": "Unable to analyze the document content at this time.",
                "answer_type": "error"
            }
    
    def _calculate_answer_confidence(self, relevance: str, answer: str, search_results: list) -> float:
        confidence = 0.0
        
        if relevance == "RELEVANT":
            confidence += 0.6
        elif relevance == "PARTIAL":
            confidence += 0.3
        else:
            confidence += 0.1
        
        if search_results:
            avg_score = sum(r.get('score', 0) for r in search_results[:3]) / min(3, len(search_results))
            confidence += min(0.3, avg_score * 0.5)
        
        vague_indicators = [
            "partial information", "not entirely clear", "may be", "possibly",
            "unable to determine", "not specified", "unclear"
        ]
        
        if any(indicator in answer.lower() for indicator in vague_indicators):
            confidence *= 0.7
        
        if "page" in answer.lower() and any(char.isdigit() for char in answer):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _determine_answer_type(self, answer: str, relevance: str, confidence: float) -> str:
        answer_lower = answer.lower()
        
        if confidence < 0.3 or relevance == "NOT_RELEVANT":
            return "not_available"
        elif "partial information" in answer_lower or relevance == "PARTIAL":
            return "partial"
        elif confidence > 0.7:
            return "comprehensive"
        else:
            return "basic"
    
    def search_with_context_analysis(self, query: str, top_k: int = 5, doc_id: str = None) -> Dict:
        try:
            search_results = self._basic_search(
                query=query,
                top_k=top_k,
                score_threshold=0.25,
                doc_id=doc_id
            )
            
            context_analysis = self.analyze_query_context(query, search_results)
            
            return {
                "query": query,
                "results": search_results,
                "total_found": len(search_results),
                "llm_analysis": {
                    "status": "completed" if self.gemini_enabled else "unavailable",
                    "relevance_check": context_analysis.get("relevance", "unknown")
                },
                "contextual_answer": context_analysis.get("contextual_answer"),
                "confidence_score": context_analysis.get("confidence", 0.0),
                "answer_type": context_analysis.get("answer_type", "basic"),
                "can_answer": context_analysis.get("can_answer", False),
                "gemini_enabled": self.gemini_enabled
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
                "gemini_enabled": False
            }
    
    def search_similar(self, query: str, top_k: int = 5, score_threshold: float = 0.3, 
                      doc_id: str = None, use_gemini_enhancement: bool = True) -> List[Dict]:
        try:
            if not self.model_loaded:
                self._ensure_model_loaded()
                
            if self.index.ntotal == 0:
                logger.warning("Index is empty - no documents have been indexed yet")
                return []
            
            if use_gemini_enhancement and self.gemini_enabled:
                return self._enhanced_search_with_gemini(query, top_k, score_threshold, doc_id)
            else:
                return self._basic_search(query, top_k, score_threshold, doc_id)
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return self._basic_search(query, top_k, score_threshold, doc_id)
    
    def _enhanced_search_with_gemini(self, query: str, top_k: int, score_threshold: float, doc_id: str = None) -> List[Dict]:
        try:
            basic_results = self._basic_search(query, top_k * 2, score_threshold, doc_id)
            
            if not basic_results:
                return basic_results
            
            enhanced_queries = self._generate_query_variations(query)
            all_results = {}
            
            for search_query in enhanced_queries[:3]:
                query_results = self._basic_search(search_query, top_k, score_threshold, doc_id)
                
                for result in query_results:
                    key = f"{result['doc_id']}_{result['page']}"
                    if key not in all_results or all_results[key]['score'] < result['score']:
                        all_results[key] = result
            
            final_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in enhanced search: {str(e)}")
            return self._basic_search(query, top_k, score_threshold, doc_id)
    
    def _generate_query_variations(self, query: str) -> List[str]:
        if not self.gemini_enabled:
            return [query]
        
        try:
            prompt = f"""
            Generate 2 alternative search queries for: "{query}"
            
            Make them:
            1. Use synonyms and related terms
            2. More specific or more general version
            
            Return only the alternative queries, one per line (no numbering):
            """
            
            response = self.gemini_model.generate_content(prompt)
            variations = [query]
            
            if response and response.text:
                lines = response.text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    line = re.sub(r'^\d+\.?\s*', '', line)
                    line = line.strip('"-')
                    
                    if line and line != query and len(line) > 3:
                        variations.append(line)
            
            return variations[:3]
            
        except Exception as e:
            logger.error(f"Error generating query variations: {e}")
            return [query]
    
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
                "gemini_enabled": self.gemini_enabled
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
                "gemini_enabled": False,
                "error": str(e)
            }
    
    def search_document_only(self, query: str, doc_id: str, top_k: int = 5, 
                           score_threshold: float = 0.3, use_gemini: bool = True) -> List[Dict]:
        if not doc_id:
            raise ValueError("doc_id is required for document-specific search")
        
        return self.search_similar(query, top_k, score_threshold, doc_id, use_gemini)
    
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
    logger.info("Embedding service created successfully")
except Exception as e:
    logger.error(f"Failed to create embedding service: {str(e)}")
    embedding_service = None