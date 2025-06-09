from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, Dict, Any
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)
router = APIRouter()
executor = ThreadPoolExecutor(max_workers=2)

@router.get("/search/enhanced")
async def enhanced_search_with_context_analysis(
    query: str = Query(..., description="Search query text"),
    top_k: int = Query(default=5, description="Number of results to return", ge=1, le=10),
    doc_id: Optional[str] = Query(default=None, description="Limit search to specific document"),
    background_tasks: BackgroundTasks = None
):
    try:
        start_time = time.time()
        
        if embedding_service is None:
            raise HTTPException(status_code=503, detail="Search service not available")
        
        enhanced_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            embedding_service.search_with_context_analysis,
            query,
            top_k,
            doc_id
        )
        
        search_time_ms = (time.time() - start_time) * 1000
        enhanced_result["search_time_ms"] = search_time_ms
        
        return enhanced_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced search failed: {str(e)}")

def filter_relevant_results(query: str, results: list, min_relevance_score: float = 0.25) -> list:
    if not results or not query:
        return results
    
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'what', 'how', 'when', 'where', 'why', 'who',
        'which', 'this', 'that', 'these', 'those', 'can', 'could', 'should', 'would'
    }
    
    query_words = [
        word.lower().strip() 
        for word in query.split() 
        if len(word) > 2 and word.lower() not in stop_words
    ]
    
    if not query_words:
        return [r for r in results if r.get('score', 0) > 0.4]
    
    filtered_results = []
    
    for result in results:
        context = result.get('context', '').lower()
        similarity_score = result.get('score', 0)
        
        exact_matches = sum(1 for word in query_words if word in context)
        
        phrase_bonus = 0
        if len(query_words) > 1:
            query_phrase = ' '.join(query_words)
            if query_phrase in context:
                phrase_bonus = 0.2
        
        length_bonus = min(0.1, len(result.get('context', '')) / 2000)
        
        word_match_ratio = exact_matches / len(query_words) if query_words else 0
        composite_score = (
            similarity_score + 
            (word_match_ratio * 0.3) + 
            phrase_bonus + 
            length_bonus
        )
        
        is_relevant = (
            similarity_score > 0.4 or
            (similarity_score > min_relevance_score and exact_matches >= 1) or
            (word_match_ratio >= 0.5 and similarity_score > 0.2) or
            (phrase_bonus > 0 and similarity_score > 0.3)
        )
        
        if is_relevant:
            result['word_matches'] = exact_matches
            result['match_ratio'] = word_match_ratio
            result['composite_score'] = composite_score
            result['phrase_match'] = phrase_bonus > 0
            filtered_results.append(result)
    
    filtered_results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
    
    final_results = [
        r for r in filtered_results 
        if r.get('composite_score', 0) > 0.35 or r.get('phrase_match', False)
    ]
    
    return final_results

@router.get("/search/context-check")
async def check_query_context(
    query: str = Query(..., description="Query to analyze"),
    doc_id: Optional[str] = Query(default=None, description="Document to search in")
):
    try:
        if embedding_service is None:
            raise HTTPException(status_code=503, detail="Search service not available")
        
        results = embedding_service._basic_search(
            query=query,
            top_k=3,
            score_threshold=0.2,
            doc_id=doc_id
        )
        
        if not results:
            return {
                "query": query,
                "can_answer": False,
                "reason": "No relevant content found"
            }
        
        if embedding_service.anthropic_enabled:
            combined_context = "\n".join([r['context'][:500] for r in results[:2]])
            
            prompt = f"""Can this content answer the question "{query}"?
            
Content: {combined_context}

Reply with: YES, NO, or PARTIAL"""
            
            try:
                response = embedding_service._call_anthropic(prompt, max_tokens=10)
                relevance = response.strip().upper() if response else "NO"
                
                can_answer = relevance in ["YES", "PARTIAL"]
                
                return {
                    "query": query,
                    "can_answer": can_answer,
                    "reason": f"LLM assessment: {relevance}",
                    "found_results": len(results)
                }
            except Exception:
                pass
        
        avg_score = sum(r['score'] for r in results) / len(results)
        can_answer = avg_score > 0.35
        
        return {
            "query": query,
            "can_answer": can_answer,
            "reason": f"Similarity-based assessment (avg score: {avg_score:.3f})",
            "found_results": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context check failed: {str(e)}")

@router.get("/search/debug")
async def debug_search(
    query: str = Query(..., description="Search query for debugging"),
    doc_id: Optional[str] = Query(default=None, description="Document to search in")
):
    try:
        if embedding_service is None:
            raise HTTPException(status_code=503, detail="Search service not available")
        
        all_results = embedding_service._basic_search(
            query=query,
            top_k=20,
            score_threshold=0.05,
            doc_id=doc_id
        )
        
        import re
        query_lower = query.lower()
        clause_numbers = re.findall(r'\b(\d+(?:\.\d+)*)\b', query)
        query_terms = [term.strip() for term in re.split(r'[^\w\.]', query_lower) if len(term.strip()) > 2]
        
        related_pages = []
        for embedding_id, metadata in embedding_service.chunk_metadata.items():
            if doc_id and metadata.get("doc_id") != doc_id:
                continue
            
            content = metadata.get("text_chunk", "").lower()
            
            matches = []
            for term in query_terms:
                if term in content:
                    matches.append(f"'{term}'")
            
            for number in clause_numbers:
                if number in content:
                    matches.append(f"number '{number}'")
            
            if matches:
                related_pages.append({
                    "page": metadata.get("page_number"),
                    "embedding_id": embedding_id,
                    "matches": matches,
                    "preview": metadata.get("text_chunk", "")[:200] + "..."
                })
        
        return {
            "query": query,
            "extracted_terms": query_terms,
            "extracted_numbers": clause_numbers,
            "search_results": [
                {
                    "page": r["page"],
                    "score": r["score"],
                    "preview": r["context"][:150] + "..."
                }
                for r in all_results[:10]
            ],
            "content_matches": related_pages,
            "total_embeddings": len(embedding_service.chunk_metadata),
            "pages_searched": len(set(r["page"] for r in all_results)) if all_results else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug search failed: {str(e)}")

@router.get("/search/pages")
async def list_searchable_pages(
    doc_id: Optional[str] = Query(default=None, description="Document to examine")
):
    try:
        if embedding_service is None:
            raise HTTPException(status_code=503, detail="Search service not available")
        
        pages_info = {}
        for embedding_id, metadata in embedding_service.chunk_metadata.items():
            if doc_id and metadata.get("doc_id") != doc_id:
                continue
                
            page_num = metadata.get("page_number")
            if page_num not in pages_info:
                pages_info[page_num] = {
                    "page": page_num,
                    "chunks": 0,
                    "total_chars": 0,
                    "detected_patterns": [],
                    "preview": ""
                }
            
            pages_info[page_num]["chunks"] += 1
            pages_info[page_num]["total_chars"] += metadata.get("char_count", 0)
            
            content = metadata.get("text_chunk", "")
            if not pages_info[page_num]["preview"]:
                pages_info[page_num]["preview"] = content[:300] + "..."
            
            import re
            numbered_patterns = re.findall(r'(\d+(?:\.\d+)*)\s+([A-Za-z][^0-9\n]{5,50})', content)
            for match in numbered_patterns:
                pattern_info = f"{match[0]} {match[1][:30]}..."
                if pattern_info not in pages_info[page_num]["detected_patterns"]:
                    pages_info[page_num]["detected_patterns"].append(pattern_info)
        
        return {
            "total_pages": len(pages_info),
            "pages": sorted(pages_info.values(), key=lambda x: x["page"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pages listing failed: {str(e)}")

@router.get("/search/fast")
async def fast_search(
    query: str = Query(..., description="Search query text"),
    top_k: int = Query(default=5, description="Number of results to return", ge=1, le=50),
    score_threshold: float = Query(default=0.3, description="Minimum similarity score", ge=0.0, le=1.0),
    doc_id: Optional[str] = Query(default=None, description="Limit search to specific document")
):
    try:
        start_time = time.time()
        
        if embedding_service is None:
            raise HTTPException(status_code=503, detail="Search service not available")
        
        results = embedding_service._basic_search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            doc_id=doc_id
        )
        
        filtered_results = filter_relevant_results(query, results)
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "results": filtered_results[:top_k],
            "total_found": len(filtered_results),
            "search_time_ms": search_time_ms,
            "enhancement_type": "relevance_filtered",
            "relevance_filtered": len(results) - len(filtered_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fast search failed: {str(e)}")