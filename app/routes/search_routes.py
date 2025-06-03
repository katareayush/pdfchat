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
    """
    Enhanced search: Fast results first, then LLM analysis for contextual answer
    """
    try:
        start_time = time.time()
        
        if embedding_service is None:
            raise HTTPException(status_code=503, detail="Search service not available")
        
        # Phase 1: Get fast search results immediately
        fast_results = embedding_service._basic_search(
            query=query,
            top_k=top_k * 2,  # Get more results for better context
            score_threshold=0.25,  # Lower threshold for broader search
            doc_id=doc_id
        )
        
        # Filter for relevance
        relevant_results = filter_relevant_results(query, fast_results)
        
        search_time_ms = (time.time() - start_time) * 1000
        
        # Prepare base response with immediate results
        response = {
            "query": query,
            "results": relevant_results[:top_k],
            "total_found": len(relevant_results),
            "search_time_ms": search_time_ms,
            "llm_analysis": None,
            "contextual_answer": None,
            "confidence_score": None
        }
        
        # Phase 2: LLM Analysis (if Gemini available)
        if (hasattr(embedding_service, 'gemini_enabled') and 
            embedding_service.gemini_enabled and 
            relevant_results):
            
            try:
                # Combine contexts from top results
                combined_context = "\n\n".join([
                    f"Page {result['page']}: {result['context'][:300]}"
                    for result in relevant_results[:5]
                ])
                
                # Generate contextual answer using LLM
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
    """
    Generate contextual answer using LLM with strict relevance checking
    """
    try:
        # First, check if the context is relevant to the query
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
                "confidence_score": 0.1,
                "answer_type": "not_available"
            }
        
        # Generate answer if relevant
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
        
        # Calculate confidence based on relevance and answer quality
        confidence = calculate_answer_confidence(relevance, contextual_answer, search_results)
        
        # Determine answer type
        answer_type = determine_answer_type(contextual_answer, relevance, confidence)
        
        return {
            "llm_analysis": {
                "status": "completed",
                "relevance_check": relevance.lower(),
                "processing_time_ms": 0  # Would track this in production
            },
            "contextual_answer": contextual_answer,
            "confidence_score": confidence,
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
            "confidence_score": 0.0,
            "answer_type": "error"
        }

def calculate_answer_confidence(relevance: str, answer: str, search_results: list) -> float:
    """
    Calculate confidence score for the generated answer
    """
    confidence = 0.0
    
    # Base confidence from relevance check
    if relevance == "RELEVANT":
        confidence += 0.6
    elif relevance == "PARTIAL":
        confidence += 0.3
    else:
        confidence += 0.1
    
    # Boost confidence based on search result scores
    if search_results:
        avg_score = sum(r.get('score', 0) for r in search_results[:3]) / min(3, len(search_results))
        confidence += min(0.3, avg_score * 0.5)
    
    # Reduce confidence for vague answers
    vague_indicators = [
        "partial information", "not entirely clear", "may be", "possibly",
        "unable to determine", "not specified", "unclear"
    ]
    
    if any(indicator in answer.lower() for indicator in vague_indicators):
        confidence *= 0.7
    
    # Boost confidence for specific answers with page references
    if "page" in answer.lower() and any(char.isdigit() for char in answer):
        confidence += 0.1
    
    return min(1.0, confidence)

def determine_answer_type(answer: str, relevance: str, confidence: float) -> str:
    """
    Categorize the type of answer provided
    """
    answer_lower = answer.lower()
    
    if confidence < 0.3 or relevance == "NOT_RELEVANT":
        return "not_available"
    elif "partial information" in answer_lower or relevance == "PARTIAL":
        return "partial"
    elif confidence > 0.7:
        return "comprehensive"
    else:
        return "basic"

def filter_relevant_results(query: str, results: list, min_relevance_score: float = 0.25) -> list:
    """
    Enhanced relevance filtering with multiple criteria
    """
    if not results or not query:
        return results
    
    # Extract meaningful words from query
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
        
        # Multiple relevance signals
        exact_matches = sum(1 for word in query_words if word in context)
        
        # Phrase matching (for multi-word queries)
        phrase_bonus = 0
        if len(query_words) > 1:
            query_phrase = ' '.join(query_words)
            if query_phrase in context:
                phrase_bonus = 0.2
        
        # Context length bonus (longer contexts often more informative)
        length_bonus = min(0.1, len(result.get('context', '')) / 2000)
        
        # Calculate composite relevance score
        word_match_ratio = exact_matches / len(query_words) if query_words else 0
        composite_score = (
            similarity_score + 
            (word_match_ratio * 0.3) + 
            phrase_bonus + 
            length_bonus
        )
        
        # More strict filtering criteria
        is_relevant = (
            # High semantic similarity
            similarity_score > 0.4 or
            # Good similarity with word matches
            (similarity_score > min_relevance_score and exact_matches >= 1) or
            # Strong word match coverage
            (word_match_ratio >= 0.5 and similarity_score > 0.2) or
            # Phrase match with decent similarity
            (phrase_bonus > 0 and similarity_score > 0.3)
        )
        
        if is_relevant:
            result['word_matches'] = exact_matches
            result['match_ratio'] = word_match_ratio
            result['composite_score'] = composite_score
            result['phrase_match'] = phrase_bonus > 0
            filtered_results.append(result)
    
    # Sort by composite relevance score
    filtered_results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
    
    # Final quality filter - remove very low quality results
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
    """
    Quick endpoint to check if a query can be answered by the document
    """
    try:
        if embedding_service is None:
            raise HTTPException(status_code=503, detail="Search service not available")
        
        # Get initial search results
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
                "reason": "No relevant content found",
                "confidence": 0.0
            }
        
        # Quick relevance check if Gemini available
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
                confidence = 0.8 if relevance == "YES" else (0.5 if relevance == "PARTIAL" else 0.2)
                
                return {
                    "query": query,
                    "can_answer": can_answer,
                    "reason": f"LLM assessment: {relevance}",
                    "confidence": confidence,
                    "found_results": len(results)
                }
            except Exception:
                pass
        
        # Fallback to similarity-based assessment
        avg_score = sum(r['score'] for r in results) / len(results)
        can_answer = avg_score > 0.35
        
        return {
            "query": query,
            "can_answer": can_answer,
            "reason": f"Similarity-based assessment (avg score: {avg_score:.3f})",
            "confidence": min(0.8, avg_score * 2),
            "found_results": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context check failed: {str(e)}")

# Keep existing fast search endpoint
@router.get("/search/fast")
async def fast_search(
    query: str = Query(..., description="Search query text"),
    top_k: int = Query(default=5, description="Number of results to return", ge=1, le=50),
    score_threshold: float = Query(default=0.3, description="Minimum similarity score", ge=0.0, le=1.0),
    doc_id: Optional[str] = Query(default=None, description="Limit search to specific document")
):
    """
    Super fast search without any AI enhancement - returns results in <20ms
    """
    try:
        start_time = time.time()
        
        if embedding_service is None:
            raise HTTPException(status_code=503, detail="Search service not available")
        
        # Force basic search without Gemini
        results = embedding_service._basic_search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            doc_id=doc_id
        )
        
        # Enhanced relevance filtering
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