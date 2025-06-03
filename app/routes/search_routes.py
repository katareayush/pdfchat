from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, Dict, Any
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from app.services.embedding_service import embedding_service
from app.models.search_models import SearchRequest, SearchResponse, SearchResult, EmbeddingStats

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
        
        fast_results = embedding_service._basic_search(
            query=query,
            top_k=top_k * 2,
            score_threshold=0.25,
            doc_id=doc_id
        )
        
        relevant_results = filter_relevant_results(query, fast_results)
        
        search_time_ms = (time.time() - start_time) * 1000
        
        response = {
            "query": query,
            "results": relevant_results[:top_k],
            "total_found": len(relevant_results),
            "search_time_ms": search_time_ms,
            "llm_analysis": None,
            "contextual_answer": None
        }
        
        if (hasattr(embedding_service, 'gemini_enabled') and 
            embedding_service.gemini_enabled and 
            relevant_results):
            
            try:
                combined_context = "\n\n".join([
                    f"Page {result['page']}: {result['context'][:300]}"
                    for result in relevant_results[:5]
                ])
                
                llm_analysis = await generate_contextual_answer(
                    query, combined_context, relevant_results
                )
                
                response.update(llm_analysis)
                
            except Exception as e:
                logger.error(f"LLM analysis failed: {str(e)}")
                response["llm_analysis"] = {
                    "status": "failed",
                    "error": str(e)
                }
        else:
            response["llm_analysis"] = {
                "status": "unavailable",
                "reason": "Gemini not enabled or no results found"
            }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced search failed: {str(e)}")

async def generate_contextual_answer(query: str, combined_context: str, search_results: list) -> Dict[str, Any]:
    try:
        relevance_prompt = f"""
        Analyze if the provided document content can answer the user's question.
        
        Question: "{query}"
        
        Document Content:
        {combined_context[:1000]}
        
        Respond with only:
        - "RELEVANT" if the content contains information that can answer the question
        - "NOT_RELEVANT" if the content does not contain relevant information
        - "PARTIAL" if the content has some related information but cannot fully answer
        
        Response:"""
        
        relevance_response = embedding_service.gemini_model.generate_content(relevance_prompt)
        relevance = relevance_response.text.strip().upper() if relevance_response.text else "NOT_RELEVANT"
        
        logger.info(f"Relevance check for query '{query}': {relevance}")
        
        if relevance == "NOT_RELEVANT":
            return {
                "llm_analysis": {
                    "status": "completed",
                    "relevance_check": "not_relevant"
                },
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
        
        answer_response = embedding_service.gemini_model.generate_content(answer_prompt)
        contextual_answer = answer_response.text.strip() if answer_response.text else ""
        
        answer_type = determine_answer_type(contextual_answer, relevance)
        
        return {
            "llm_analysis": {
                "status": "completed",
                "relevance_check": relevance.lower(),
                "processing_time_ms": 0
            },
            "contextual_answer": contextual_answer,
            "answer_type": answer_type
        }
        
    except Exception as e:
        logger.error(f"Error generating contextual answer: {str(e)}")
        return {
            "llm_analysis": {
                "status": "error",
                "error": str(e)
            },
            "contextual_answer": "Unable to analyze the document content at this time.",
            "answer_type": "error"
        }

def determine_answer_type(answer: str, relevance: str) -> str:
    answer_lower = answer.lower()
    
    if relevance == "NOT_RELEVANT":
        return "not_available"
    elif "partial information" in answer_lower or relevance == "PARTIAL":
        return "partial"
    elif relevance == "RELEVANT":
        return "comprehensive"
    else:
        return "basic"

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
        
        if hasattr(embedding_service, 'gemini_enabled') and embedding_service.gemini_enabled:
            combined_context = "\n".join([r['context'][:200] for r in results[:2]])
            
            prompt = f"""
            Can this content answer the question "{query}"?
            
            Content: {combined_context}
            
            Reply with: YES, NO, or PARTIAL
            """
            
            try:
                response = embedding_service.gemini_model.generate_content(prompt)
                relevance = response.text.strip().upper() if response.text else "NO"
                
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