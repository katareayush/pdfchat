# 🔧 Enhanced Embedding Service - Drop-in replacement for your existing service
# app/services/embedding_service.py

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import pickle
import json
from pathlib import Path
from datetime import datetime
import logging
import os
import time
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Enhanced embedding service with hybrid search - drop-in replacement
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunk_metadata = {}  
        self.embedding_dim = 384  
        self.next_embedding_id = 0
        self.model_loaded = False
        
        # Enhanced search components (NEW)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.document_texts = []
        
        # Storage paths
        self.storage_dir = Path("data")
        self.storage_dir.mkdir(exist_ok=True)
        self.index_path = self.storage_dir / "faiss_index.bin"
        self.metadata_path = self.storage_dir / "chunk_metadata.json"
        self.tfidf_path = self.storage_dir / "tfidf_data.pkl"  # NEW
        
        # Enhanced search configuration (NEW)
        self.enhanced_search_enabled = True
        self.search_weights = {
            "semantic": 0.7,
            "lexical": 0.3,
            "fuzzy": 0.1
        }
        
        # Initialize lazily to avoid blocking startup
        self._initialize_lazy()
    
    def _initialize_lazy(self):
        """Initialize components lazily to avoid blocking server startup"""
        try:
            logger.info("🔄 Initializing enhanced embedding service...")
            self._load_index()
            self._load_tfidf_data()  # NEW
            logger.info("✅ Enhanced embedding service initialized (model will load on first use)")
        except Exception as e:
            logger.error(f"❌ Error initializing embedding service: {str(e)}")
    
    def _load_tfidf_data(self):
        """Load TF-IDF components for lexical search"""
        try:
            if self.tfidf_path.exists():
                logger.info("📂 Loading TF-IDF data...")
                with open(self.tfidf_path, 'rb') as f:
                    tfidf_data = pickle.load(f)
                    self.tfidf_vectorizer = tfidf_data.get('vectorizer')
                    self.tfidf_matrix = tfidf_data.get('matrix')
                    self.document_texts = tfidf_data.get('texts', [])
                logger.info(f"✅ Loaded TF-IDF data for {len(self.document_texts)} documents")
            else:
                self._initialize_tfidf()
        except Exception as e:
            logger.error(f"❌ Error loading TF-IDF data: {str(e)}")
            self._initialize_tfidf()
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True
        )
        self.tfidf_matrix = None
        self.document_texts = []
    
    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common encoding issues
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def _fuzzy_match_score(self, query: str, text: str) -> float:
        """Calculate fuzzy matching score"""
        query = query.lower().strip()
        text = text.lower()
        
        if not query or not text:
            return 0.0
        
        # Exact substring match
        if query in text:
            return 1.0
        
        # Word-level matching
        query_words = set(query.split())
        text_words = set(text.split())
        
        if not query_words:
            return 0.0
        
        # Calculate word overlap
        common_words = query_words.intersection(text_words)
        word_overlap = len(common_words) / len(query_words)
        
        # Partial word matching for longer words
        partial_matches = 0
        for q_word in query_words:
            if len(q_word) > 4:  # Only for longer words
                for t_word in text_words:
                    if q_word in t_word or t_word in q_word:
                        partial_matches += 0.5
                        break
        
        partial_score = min(partial_matches / len(query_words), 0.5)
        
        return word_overlap + partial_score
    
    def _lexical_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """TF-IDF based lexical search"""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None or len(self.document_texts) == 0:
            return []
        
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum threshold
                    embedding_id = str(idx)
                    if embedding_id in self.chunk_metadata:
                        metadata = self.chunk_metadata[embedding_id]
                        results.append({
                            "page": metadata["page_number"],
                            "context": metadata["text_chunk"],
                            "doc_id": metadata["doc_id"],
                            "score": float(similarities[idx]),
                            "char_count": metadata.get("char_count", 0),
                            "search_type": "lexical"
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in lexical search: {str(e)}")
            return []
    
    def _ensure_model_loaded(self):
        """Load model only when needed (lazy loading)"""
        if not self.model_loaded:
            try:
                logger.info(f"🤖 Loading sentence transformer model: {self.model_name}")
                start_time = time.time()
                
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.model_loaded = True
                
                load_time = time.time() - start_time
                logger.info(f"✅ Model loaded in {load_time:.2f}s. Embedding dimension: {self.embedding_dim}")
                
            except Exception as e:
                logger.error(f"❌ Error loading model: {str(e)}")
                raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def _load_index(self):
        """Load existing FAISS index or create a new one"""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                # Load existing index
                logger.info("📂 Loading existing FAISS index...")
                self.index = faiss.read_index(str(self.index_path))
                
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.chunk_metadata = json.load(f)
                
                # Update next_embedding_id
                if self.chunk_metadata:
                    max_id = max(int(k) for k in self.chunk_metadata.keys())
                    self.next_embedding_id = max_id + 1
                
                logger.info(f"✅ Loaded existing index with {self.index.ntotal} vectors")
            else:
                # Create new index
                logger.info(f"🆕 Creating new FAISS index with dimension {self.embedding_dim}")
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                
        except Exception as e:
            logger.error(f"❌ Error loading/creating index: {str(e)}")
            # Fallback: create new index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save with error handling
            temp_index_path = self.index_path.with_suffix('.tmp')
            temp_metadata_path = self.metadata_path.with_suffix('.tmp')
            temp_tfidf_path = self.tfidf_path.with_suffix('.tmp')
            
            # Write to temp files first
            faiss.write_index(self.index, str(temp_index_path))
            with open(temp_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)
            
            # Save TF-IDF data
            if self.tfidf_vectorizer and self.tfidf_matrix is not None:
                tfidf_data = {
                    'vectorizer': self.tfidf_vectorizer,
                    'matrix': self.tfidf_matrix,
                    'texts': self.document_texts
                }
                with open(temp_tfidf_path, 'wb') as f:
                    pickle.dump(tfidf_data, f)
            
            # Atomic move
            temp_index_path.replace(self.index_path)
            temp_metadata_path.replace(self.metadata_path)
            if temp_tfidf_path.exists():
                temp_tfidf_path.replace(self.tfidf_path)
            
            logger.info(f"💾 Enhanced index and metadata saved successfully ({self.index.ntotal} vectors)")
            
        except Exception as e:
            logger.error(f"❌ Error saving index: {str(e)}")
            # Clean up temp files
            for temp_file in [temp_index_path, temp_metadata_path, temp_tfidf_path]:
                if temp_file.exists():
                    temp_file.unlink()
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts with batching for better performance
        """
        try:
            self._ensure_model_loaded()  # Load model only when needed
            
            if not texts:
                return np.array([]).reshape(0, self.embedding_dim)
            
            logger.info(f"🔢 Generating embeddings for {len(texts)} texts...")
            start_time = time.time()
            
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Process in batches to avoid memory issues
            all_embeddings = []
            for i in range(0, len(processed_texts), batch_size):
                batch_texts = processed_texts[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(processed_texts) + batch_size - 1)//batch_size}")
                
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    normalize_embeddings=True,
                    show_progress_bar=False  # Disable progress bar to avoid console spam
                )
                all_embeddings.append(batch_embeddings)
            
            # Combine all batches
            embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([]).reshape(0, self.embedding_dim)
            
            generation_time = time.time() - start_time
            logger.info(f"✅ Generated {len(embeddings)} embeddings in {generation_time:.2f}s")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def add_document_embeddings(self, doc_id: str, pages: List[Dict]) -> Dict[str, int]:
        """
        Add embeddings for a document's pages to the index (ENHANCED VERSION)
        """
        try:
            logger.info(f"🤖 Adding enhanced embeddings for document {doc_id}")
            start_time = time.time()
            
            # Filter out empty pages
            non_empty_pages = [page for page in pages if page.get('text_chunk', '').strip()]
            
            if not non_empty_pages:
                logger.warning(f"⚠️ No non-empty pages found for document {doc_id}")
                return {}
            
            logger.info(f"📄 Processing {len(non_empty_pages)} non-empty pages")
            
            # Extract texts for embedding
            texts = [page['text_chunk'] for page in non_empty_pages]
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            if embeddings.size == 0:
                logger.warning(f"⚠️ No embeddings generated for document {doc_id}")
                return {}
            
            # Add to FAISS index
            logger.info(f"📊 Adding {len(embeddings)} embeddings to FAISS index...")
            self.index.add(embeddings.astype('float32'))
            
            # Update TF-IDF with new texts
            self.document_texts.extend(processed_texts)
            if len(self.document_texts) > 0:
                try:
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.document_texts)
                    logger.info(f"📊 Updated TF-IDF matrix with {len(self.document_texts)} documents")
                except Exception as e:
                    logger.warning(f"⚠️ TF-IDF update failed: {str(e)}")
            
            # Store metadata
            page_to_embedding_id = {}
            for i, page in enumerate(non_empty_pages):
                embedding_id = self.next_embedding_id + i
                
                # Store metadata
                self.chunk_metadata[str(embedding_id)] = {
                    "doc_id": doc_id,
                    "page_number": page['page_number'],
                    "text_chunk": page['text_chunk'][:1500],  # Store more context
                    "char_count": page.get('char_count', len(page['text_chunk'])),
                    "added_at": datetime.now().isoformat(),
                    "processed_text": processed_texts[i][:1000]  # Store processed version
                }
                
                page_to_embedding_id[page['page_number']] = embedding_id
            
            # Update next embedding ID
            self.next_embedding_id += len(non_empty_pages)
            
            # Save to disk
            self._save_index()
            
            total_time = time.time() - start_time
            logger.info(f"✅ Successfully added enhanced embeddings for document {doc_id} in {total_time:.2f}s")
            
            return page_to_embedding_id
            
        except Exception as e:
            logger.error(f"❌ Error adding document embeddings for {doc_id}: {str(e)}")
            # Don't raise exception - log error but continue
            return {}
    
    def search_similar(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict]:
        """
        ENHANCED: Search for similar chunks with hybrid semantic + lexical approach
        This method maintains the same signature as before for frontend compatibility
        """
        try:
            if not self.model_loaded:
                self._ensure_model_loaded()
                
            if self.index.ntotal == 0:
                logger.warning("⚠️ Index is empty - no documents have been indexed yet")
                return []
            
            logger.info(f"🔍 Enhanced search for: '{query[:50]}...' (top_k={top_k})")
            start_time = time.time()
            
            # If enhanced search is disabled, fall back to original semantic search
            if not self.enhanced_search_enabled:
                return self._original_semantic_search(query, top_k, score_threshold)
            
            # ENHANCED HYBRID SEARCH
            all_results = {}
            
            # 1. Semantic search
            semantic_results = self._semantic_search(query, top_k * 2)
            
            # 2. Lexical search
            lexical_results = self._lexical_search(query, top_k * 2)
            
            # 3. Combine and score results
            for result in semantic_results:
                key = f"{result['doc_id']}_{result['page']}"
                result['semantic_score'] = result['score']
                result['lexical_score'] = 0.0
                all_results[key] = result
            
            for result in lexical_results:
                key = f"{result['doc_id']}_{result['page']}"
                if key in all_results:
                    all_results[key]['lexical_score'] = result['score']
                else:
                    result['semantic_score'] = 0.0
                    result['lexical_score'] = result['score']
                    all_results[key] = result
            
            # 4. Calculate hybrid scores with fuzzy matching
            for result in all_results.values():
                semantic_score = result.get('semantic_score', 0)
                lexical_score = result.get('lexical_score', 0)
                
                # Normalize scores
                norm_semantic = min(semantic_score, 1.0)
                norm_lexical = min(lexical_score, 1.0)
                
                # Add fuzzy matching bonus
                fuzzy_score = self._fuzzy_match_score(query, result['context'])
                
                # Calculate final hybrid score
                hybrid_score = (
                    self.search_weights['semantic'] * norm_semantic +
                    self.search_weights['lexical'] * norm_lexical +
                    self.search_weights['fuzzy'] * fuzzy_score
                )
                
                result['score'] = hybrid_score
                result['fuzzy_score'] = fuzzy_score
                
                # Add position bonus for early query matches
                context_lower = result['context'].lower()
                query_lower = query.lower()
                position = context_lower.find(query_lower)
                if position != -1:
                    position_bonus = max(0, 0.1 * (1 - position / len(context_lower)))
                    result['score'] += position_bonus
            
            # 5. Sort and filter results
            sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
            filtered_results = [r for r in sorted_results if r['score'] >= score_threshold]
            final_results = filtered_results[:top_k]
            
            # 6. Clean up result format to match original API
            for result in final_results:
                # Remove extra fields that frontend doesn't expect
                result.pop('semantic_score', None)
                result.pop('lexical_score', None)
                result.pop('fuzzy_score', None)
                result.pop('search_type', None)
            
            search_time = time.time() - start_time
            logger.info(f"✅ Enhanced search found {len(final_results)} results in {search_time*1000:.1f}ms")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced search: {str(e)}")
            # Fallback to original search
            return self._original_semantic_search(query, top_k, score_threshold)
    
    def _semantic_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Pure semantic search using embeddings"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])
            
            if query_embedding.size == 0:
                return []
            
            # Search FAISS index
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, self.index.ntotal)
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                    
                embedding_id = str(idx)
                if embedding_id in self.chunk_metadata:
                    metadata = self.chunk_metadata[embedding_id]
                    results.append({
                        "page": metadata["page_number"],
                        "context": metadata["text_chunk"],
                        "doc_id": metadata["doc_id"],
                        "score": float(score),
                        "char_count": metadata.get("char_count", 0)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in semantic search: {str(e)}")
            return []
    
    def _original_semantic_search(self, query: str, top_k: int, score_threshold: float) -> List[Dict]:
        """Original semantic search for fallback"""
        try:
            query_embedding = self.generate_embeddings([query])
            
            if query_embedding.size == 0:
                logger.error("❌ Failed to generate query embedding")
                return []
            
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, self.index.ntotal)
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < score_threshold:
                    continue
                    
                embedding_id = str(idx)
                if embedding_id in self.chunk_metadata:
                    metadata = self.chunk_metadata[embedding_id]
                    results.append({
                        "page": metadata["page_number"],
                        "context": metadata["text_chunk"],
                        "doc_id": metadata["doc_id"],
                        "score": float(score),
                        "char_count": metadata.get("char_count", 0)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in original semantic search: {str(e)}")
            return []
    
    def remove_document_embeddings(self, doc_id: str) -> int:
        """
        Remove all embeddings for a document (rebuilds index)
        """
        try:
            logger.info(f"🗑️ Removing embeddings for document {doc_id}")
            
            # Find embeddings to remove
            embeddings_to_remove = []
            for embedding_id, metadata in self.chunk_metadata.items():
                if metadata.get("doc_id") == doc_id:
                    embeddings_to_remove.append(embedding_id)
            
            if not embeddings_to_remove:
                logger.info(f"ℹ️ No embeddings found for document {doc_id}")
                return 0
            
            # Remove from metadata
            for embedding_id in embeddings_to_remove:
                del self.chunk_metadata[embedding_id]
            
            # Rebuild index (FAISS doesn't support efficient deletion)
            self._rebuild_index()
            
            logger.info(f"✅ Removed {len(embeddings_to_remove)} embeddings for document {doc_id}")
            return len(embeddings_to_remove)
            
        except Exception as e:
            logger.error(f"❌ Error removing document embeddings: {str(e)}")
            return 0
    
    def _rebuild_index(self):
        """Rebuild FAISS index from current metadata"""
        try:
            logger.info("🔄 Rebuilding enhanced index...")
            start_time = time.time()
            
            if not self.chunk_metadata:
                # Create empty index
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.next_embedding_id = 0
                self.document_texts = []
                self.tfidf_matrix = None
                self._save_index()
                logger.info("✅ Created empty index")
                return
            
            # Extract all texts
            texts = []
            processed_texts = []
            for metadata in self.chunk_metadata.values():
                text = metadata.get("text_chunk", "")
                texts.append(text)
                processed_texts.append(metadata.get("processed_text", self._preprocess_text(text)))
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Create new index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            if embeddings.size > 0:
                self.index.add(embeddings.astype('float32'))
            
            # Rebuild TF-IDF
            self.document_texts = processed_texts
            if len(self.document_texts) > 0:
                try:
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.document_texts)
                except Exception as e:
                    logger.warning(f"⚠️ TF-IDF rebuild failed: {str(e)}")
                    self.tfidf_matrix = None
            
            # Update embedding IDs to be sequential
            old_metadata = self.chunk_metadata.copy()
            self.chunk_metadata = {}
            self.next_embedding_id = 0
            
            for i, (old_id, metadata) in enumerate(old_metadata.items()):
                self.chunk_metadata[str(i)] = metadata
                self.next_embedding_id = i + 1
            
            # Save rebuilt index
            self._save_index()
            
            rebuild_time = time.time() - start_time
            logger.info(f"✅ Enhanced index rebuilt with {self.index.ntotal} vectors in {rebuild_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error rebuilding index: {str(e)}")
            # Fallback: create empty index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.chunk_metadata = {}
            self.next_embedding_id = 0
    
    def get_stats(self) -> Dict:
        """Get statistics about the current index"""
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
                "enhanced_search_enabled": self.enhanced_search_enabled,  # NEW
                "tfidf_ready": self.tfidf_matrix is not None,  # NEW
                "search_weights": self.search_weights  # NEW
            }
        except Exception as e:
            logger.error(f"❌ Error getting stats: {str(e)}")
            return {
                "total_embeddings": 0,
                "embedding_dimension": self.embedding_dim,
                "model_name": self.model_name,
                "model_loaded": self.model_loaded,
                "unique_documents": 0,
                "storage_size_mb": 0,
                "metadata_count": 0,
                "index_exists": False,
                "enhanced_search_enabled": False,
                "tfidf_ready": False,
                "error": str(e)
            }
    
    def configure_search(self, **kwargs) -> Dict:
        """
        NEW: Configure search weights and parameters
        """
        updated = {}
        
        if 'semantic_weight' in kwargs:
            self.search_weights['semantic'] = float(kwargs['semantic_weight'])
            updated['semantic_weight'] = self.search_weights['semantic']
        
        if 'lexical_weight' in kwargs:
            self.search_weights['lexical'] = float(kwargs['lexical_weight'])
            updated['lexical_weight'] = self.search_weights['lexical']
        
        if 'fuzzy_weight' in kwargs:
            self.search_weights['fuzzy'] = float(kwargs['fuzzy_weight'])
            updated['fuzzy_weight'] = self.search_weights['fuzzy']
        
        if 'enhanced_search_enabled' in kwargs:
            self.enhanced_search_enabled = bool(kwargs['enhanced_search_enabled'])
            updated['enhanced_search_enabled'] = self.enhanced_search_enabled
        
        # Normalize weights
        total_weight = self.search_weights['semantic'] + self.search_weights['lexical']
        if total_weight > 1.0:
            self.search_weights['semantic'] /= total_weight
            self.search_weights['lexical'] /= total_weight
        
        logger.info(f"📝 Search configuration updated: {updated}")
        return self.search_weights
    
    def health_check(self) -> Dict:
        """Check if embedding service is healthy"""
        try:
            health_info = {
                "service": "enhanced_embedding_service",
                "status": "healthy",
                "model_loaded": self.model_loaded,
                "index_ready": self.index is not None,
                "total_embeddings": self.index.ntotal if self.index else 0,
                "storage_accessible": self.storage_dir.exists(),
                "enhanced_features": {
                    "tfidf_ready": self.tfidf_matrix is not None,
                    "enhanced_search_enabled": self.enhanced_search_enabled
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Test model loading if not already loaded
            if not self.model_loaded:
                try:
                    self._ensure_model_loaded()
                    health_info["model_test"] = "success"
                except Exception as e:
                    health_info["model_test"] = f"failed: {str(e)}"
                    health_info["status"] = "degraded"
            
            return health_info
            
        except Exception as e:
            return {
                "service": "enhanced_embedding_service", 
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Create singleton instance with error handling
try:
    embedding_service = EmbeddingService()
    logger.info("✅ Enhanced embedding service singleton created")
except Exception as e:
    logger.error(f"❌ Failed to create enhanced embedding service: {str(e)}")
    # Create a dummy service that will fail gracefully
    embedding_service = None