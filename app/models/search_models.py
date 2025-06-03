# Enhanced Search Models - app/models/search_models.py
# These models extend your existing ones while maintaining backward compatibility

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Keep your existing models exactly as they are for frontend compatibility
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    doc_id: Optional[str] = Field(default=None, description="Limit search to specific document")

class SearchResult(BaseModel):
    page: int
    context: str
    doc_id: str
    score: float
    char_count: int

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int
    search_time_ms: float

# Enhanced version of EmbeddingStats with new fields (backward compatible)
class EmbeddingStats(BaseModel):
    total_embeddings: int
    embedding_dimension: int
    model_name: str
    model_loaded: bool
    unique_documents: int
    storage_size_mb: float
    metadata_count: int
    index_exists: bool
    
    # NEW FIELDS (optional for backward compatibility)
    enhanced_search_enabled: Optional[bool] = None
    tfidf_ready: Optional[bool] = None
    search_weights: Optional[Dict[str, float]] = None
    error: Optional[str] = None

# NEW MODELS for enhanced features (these won't break existing frontend)

class EnhancedSearchRequest(BaseModel):
    """Extended search request with fine-tuning options"""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    score_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity score")
    doc_id: Optional[str] = Field(default=None, description="Limit search to specific document")
    
    # Enhanced parameters
    semantic_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Weight for semantic similarity")
    lexical_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Weight for lexical similarity")
    fuzzy_weight: Optional[float] = Field(default=None, ge=0.0, le=0.5, description="Weight for fuzzy matching")
    enhanced_mode: Optional[bool] = Field(default=True, description="Use enhanced search mode")

class EnhancedSearchResult(BaseModel):
    """Extended search result with detailed scoring"""
    page: int
    context: str
    doc_id: str
    score: float
    char_count: int
    
    # Enhanced scoring details (optional)
    semantic_score: Optional[float] = None
    lexical_score: Optional[float] = None
    fuzzy_score: Optional[float] = None
    search_type: Optional[str] = None

class EnhancedSearchResponse(BaseModel):
    """Extended search response with additional metadata"""
    query: str
    results: List[EnhancedSearchResult]
    total_found: int
    search_time_ms: float
    
    # Enhanced metadata
    search_mode: Optional[str] = "hybrid"
    weights_used: Optional[Dict[str, float]] = None
    tfidf_available: Optional[bool] = None

class SearchConfigRequest(BaseModel):
    """Request model for configuring search parameters"""
    semantic_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    lexical_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fuzzy_weight: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    enhanced_search_enabled: Optional[bool] = None

class SearchConfigResponse(BaseModel):
    """Response model for search configuration"""
    message: str
    config: Dict[str, float]
    enhanced_enabled: bool

class SearchHealthResponse(BaseModel):
    """Health check response for search service"""
    service: str
    status: str
    model_loaded: bool
    index_ready: bool
    total_embeddings: int
    storage_accessible: bool
    enhanced_features: Optional[Dict[str, Any]] = None
    timestamp: str
    error: Optional[str] = None

class SearchModeTestResponse(BaseModel):
    """Response for testing different search modes"""
    query: str
    test_time_ms: float
    results: Dict[str, List[Dict]]
    comparison: Dict[str, int]