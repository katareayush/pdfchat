# Enhanced Search Router - Minimal changes to your existing router
# app/api/search.py

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import time
from app.services.embedding_service import embedding_service
from app.models.search_models import SearchRequest, SearchResponse, SearchResult, EmbeddingStats

router = APIRouter()

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for similar content across all documents using enhanced semantic similarity
    NOTE: This endpoint maintains the same API but now uses enhanced hybrid search
    """
    try:
        start_time = time.time()
        
        # Perform enhanced search (automatically uses hybrid approach)
        results = embedding_service.search_similar(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        
        # Filter by document if specified
        if request.doc_id:
            results = [r for r in results if r["doc_id"] == request.doc_id]
        
        # Convert to response format
        search_results = [
            SearchResult(
                page=result["page"],
                context=result["context"],
                doc_id=result["doc_id"],
                score=result["score"],
                char_count=result["char_count"]
            )
            for result in results
        ]
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_found=len(search_results),
            search_time_ms=search_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/search", response_model=SearchResponse)
async def search_documents_get(
    query: str = Query(..., description="Search query text"),
    top_k: int = Query(default=5, description="Number of results to return", ge=1, le=50),
    score_threshold: float = Query(default=0.0, description="Minimum similarity score", ge=0.0, le=1.0),
    doc_id: Optional[str] = Query(default=None, description="Limit search to specific document")
):
    """
    Search for similar content using GET method (enhanced version)
    """
    search_request = SearchRequest(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        doc_id=doc_id
    )
    return await search_documents(search_request)

@router.get("/embeddings/stats", response_model=EmbeddingStats)
async def get_embedding_stats():
    """
    Get statistics about the current embedding index (enhanced with new metrics)
    """
    try:
        stats = embedding_service.get_stats()
        return EmbeddingStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# NEW ENDPOINTS FOR ENHANCED FEATURES (Optional - won't break existing frontend)

@router.post("/search/configure")
async def configure_search(
    semantic_weight: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Weight for semantic similarity"),
    lexical_weight: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Weight for lexical similarity"),
    fuzzy_weight: Optional[float] = Query(default=None, ge=0.0, le=0.5, description="Weight for fuzzy matching"),
    enhanced_search_enabled: Optional[bool] = Query(default=None, description="Enable/disable enhanced search")
):
    """
    NEW: Configure enhanced search parameters
    """
    try:
        config_params = {}
        
        if semantic_weight is not None:
            config_params['semantic_weight'] = semantic_weight
        if lexical_weight is not None:
            config_params['lexical_weight'] = lexical_weight
        if fuzzy_weight is not None:
            config_params['fuzzy_weight'] = fuzzy_weight
        if enhanced_search_enabled is not None:
            config_params['enhanced_search_enabled'] = enhanced_search_enabled
        
        if not config_params:
            # Return current configuration
            return {
                "message": "Current search configuration",
                "config": embedding_service.search_weights,
                "enhanced_enabled": embedding_service.enhanced_search_enabled
            }
        
        # Update configuration
        updated_config = embedding_service.configure_search(**config_params)
        
        return {
            "message": "Search configuration updated successfully",
            "config": updated_config,
            "enhanced_enabled": embedding_service.enhanced_search_enabled
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure search: {str(e)}")

@router.get("/search/config")
async def get_search_config():
    """
    NEW: Get current search configuration
    """
    try:
        return {
            "search_weights": embedding_service.search_weights,
            "enhanced_search_enabled": embedding_service.enhanced_search_enabled,
            "tfidf_ready": embedding_service.tfidf_matrix is not None,
            "total_documents": len(embedding_service.document_texts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get search config: {str(e)}")

@router.post("/search/test-modes")
async def test_search_modes(
    query: str = Query(..., description="Test query"),
    top_k: int = Query(default=3, ge=1, le=10, description="Results per mode")
):
    """
    NEW: Test different search modes to compare results
    """
    try:
        start_time = time.time()
        
        # Test semantic only
        embedding_service.enhanced_search_enabled = False
        semantic_results = embedding_service.search_similar(query, top_k, 0.0)
        
        # Test enhanced hybrid
        embedding_service.enhanced_search_enabled = True
        hybrid_results = embedding_service.search_similar(query, top_k, 0.0)
        
        # Test lexical only
        lexical_results = embedding_service._lexical_search(query, top_k)
        
        test_time_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "test_time_ms": test_time_ms,
            "results": {
                "semantic_only": semantic_results,
                "hybrid_enhanced": hybrid_results,
                "lexical_only": lexical_results
            },
            "comparison": {
                "semantic_count": len(semantic_results),
                "hybrid_count": len(hybrid_results),
                "lexical_count": len(lexical_results)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search mode test failed: {str(e)}")

@router.post("/embeddings/rebuild")
async def rebuild_embeddings():
    """
    Rebuild the entire embedding index (useful for maintenance)
    """
    try:
        embedding_service._rebuild_index()
        stats = embedding_service.get_stats()
        return {
            "message": "Enhanced embedding index rebuilt successfully",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")

@router.delete("/embeddings/clear")
async def clear_embeddings():
    """
    Clear all embeddings from the index
    """
    try:
        # Clear metadata
        embedding_service.chunk_metadata = {}
        embedding_service.next_embedding_id = 0
        embedding_service.document_texts = []
        embedding_service.tfidf_matrix = None
        
        # Create new empty index
        import faiss
        embedding_service.index = faiss.IndexFlatIP(embedding_service.embedding_dim)
        
        # Reinitialize TF-IDF
        embedding_service._initialize_tfidf()
        
        # Save empty index
        embedding_service._save_index()
        
        return {"message": "All embeddings and enhanced data cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear embeddings: {str(e)}")

@router.get("/search/health")
async def search_health_check():
    """
    NEW: Detailed health check for enhanced search features
    """
    try:
        health = embedding_service.health_check()
        return health
    except Exception as e:
        return {
            "service": "enhanced_search",
            "status": "unhealthy",
            "error": str(e)
        }