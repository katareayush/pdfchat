from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import HTTPException
import os
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from app.routes.pdf_routes import router as pdf_router
from app.routes.search_routes import router as search_router

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080", 
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:5501",
    "http://127.0.0.1:5501",
    "http://localhost:5502",
    "http://127.0.0.1:5502",
    
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    
    "null",
]

if ENVIRONMENT == "production":
    CORS_ORIGINS.extend([
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ])

app = FastAPI(
    title="PDF RAG API",
    description="A FastAPI application for PDF upload, text extraction, and semantic search with RAG capabilities",
    version="1.0.0",
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path} from {request.client.host}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url.path} -> {response.status_code} ({process_time:.3f}s)")
    
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"❌ Global exception on {request.method} {request.url.path}: {str(exc)}", exc_info=True)
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "path": str(request.url.path)}
        )
    
    # Handle other exceptions
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if ENVIRONMENT == "development" else "Internal server error",
            "path": str(request.url.path)
        }
    )

# Include API routers
app.include_router(pdf_router, prefix="/api/v1", tags=["PDF Operations"])
app.include_router(search_router, prefix="/api/v1", tags=["Search & Embeddings"])

# Serve static files (frontend) if available
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting PDF RAG API...")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"CORS Origins: {CORS_ORIGINS[:3]}... (+{len(CORS_ORIGINS)-3} more)")
    
    # Test embedding service
    try:
        from app.services.embedding_service import embedding_service
        if embedding_service:
            health = embedding_service.health_check()
            logger.info(f"🤖 Embedding service: {health.get('status', 'unknown')}")
        else:
            logger.warning("⚠️ Embedding service not available")
    except Exception as e:
        logger.error(f"❌ Embedding service error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down PDF RAG API...")

@app.get("/")
async def root():
    """Root endpoint - serve frontend or API info"""
    frontend_file = Path("frontend/index.html")
    if frontend_file.exists():
        return FileResponse(frontend_file)
    
    return {
        "message": "PDF RAG API",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "status": "running",
        "docs": "/docs" if ENVIRONMENT != "production" else "disabled",
        "endpoints": {
            "health": "/health",
            "upload": "/api/v1/upload",
            "search": "/api/v1/search",
            "documents": "/api/v1/documents",
            "embedding_stats": "/api/v1/embeddings/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        health_info = {
            "status": "healthy",
            "service": "PDF RAG API",
            "version": "1.0.0",
            "environment": ENVIRONMENT,
            "timestamp": time.time(),
            "uptime": "unknown"  # TODO: implement uptime tracking
        }
        
        # Check embedding service health
        try:
            from app.services.embedding_service import embedding_service
            if embedding_service:
                embedding_health = embedding_service.health_check()
                health_info["embedding_service"] = embedding_health
            else:
                health_info["embedding_service"] = {"status": "unavailable"}
        except Exception as e:
            health_info["embedding_service"] = {"status": "error", "error": str(e)}
        
        # Check storage directories
        health_info["storage"] = {
            "uploads_dir": Path("uploads").exists(),
            "data_dir": Path("data").exists(),
            "frontend_dir": Path("frontend").exists()
        }
        
        return health_info
        
    except Exception as e:
        logger.error(f"❌ Health check error: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@app.get("/debug/cors")
async def debug_cors():
    """Debug CORS configuration"""
    return {
        "cors_origins": CORS_ORIGINS,
        "environment": ENVIRONMENT,
        "headers_info": "Check browser network tab for CORS headers",
        "note": "This endpoint helps debug CORS issues"
    }

# Serve frontend files for SPA routing
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve frontend for SPA routing (fallback)"""
    frontend_file = Path("frontend/index.html")
    if frontend_file.exists() and not full_path.startswith("api/"):
        return FileResponse(frontend_file)
    
    return JSONResponse(
        status_code=404,
        content={"detail": "Not found", "path": full_path}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=ENVIRONMENT == "development",
        log_level="info"
    )