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
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://localhost:8000",
    "null",
]

if ENVIRONMENT == "production":
    CORS_ORIGINS.extend([
        "https://testproject-production-b850.up.railway.app",
    ])

app = FastAPI(
    title="PDF RAG API",
    description="A FastAPI application for PDF upload, text extraction, and semantic search with RAG capabilities",
    version="1.0.0",
    docs_url="/docs",  
    redoc_url="/redoc"
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
    response = await call_next(request)
    process_time = time.time() - start_time
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "path": str(request.url.path)}
        )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if ENVIRONMENT == "development" else "Internal server error",
            "path": str(request.url.path)
        }
    )

app.include_router(pdf_router, prefix="/api/v1", tags=["PDF Operations"])
app.include_router(search_router, prefix="/api/v1", tags=["Search & Embeddings"])

@app.on_event("startup")
async def startup_event():
    for dir_path in ["uploads", "data", "models"]:
        Path(dir_path).mkdir(exist_ok=True)
    
    frontend_path = Path("frontend")
    if frontend_path.exists():
        try:
            app.mount("/static", StaticFiles(directory=frontend_path), name="static")
        except Exception:
            pass
    
    try:
        from app.services.embedding_service import embedding_service
        if embedding_service and hasattr(embedding_service, 'health_check'):
            embedding_service.health_check()
    except Exception:
        pass

@app.on_event("shutdown")
async def shutdown_event():
    pass

@app.get("/api/status")
async def api_status():
    return {
        "message": "PDF RAG API",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "status": "running",
        "docs": "/docs",
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
    health_info = {
        "status": "healthy",
        "service": "PDF RAG API",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "timestamp": time.time()
    }
    
    try:
        from app.services.embedding_service import embedding_service
        if embedding_service and hasattr(embedding_service, 'health_check'):
            embedding_health = embedding_service.health_check()
            health_info["embedding_service"] = embedding_health
        else:
            health_info["embedding_service"] = {"status": "available"}
    except Exception as e:
        health_info["embedding_service"] = {"status": "error", "error": str(e)}
    
    health_info["storage"] = {
        "uploads_dir": Path("uploads").exists(),
        "data_dir": Path("data").exists(),
        "frontend_dir": Path("display").exists()
    }
    
    return health_info

@app.get("/")
async def root():
    frontend_file = Path("display/index.html")
    if frontend_file.exists():
        return FileResponse(frontend_file)
    
    return {
        "message": "PDF RAG API",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "status": "running",
        "docs": "/docs",
        "api_status": "/api/status"
    }

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    if (full_path.startswith("api/") or 
        full_path.startswith("docs") or 
        full_path.startswith("redoc") or
        full_path == "health" or
        full_path.startswith("static/")):
        return JSONResponse(
            status_code=404,
            content={"detail": "Endpoint not found", "path": full_path}
        )
    
    frontend_file = Path("display/index.html")
    if frontend_file.exists():
        return FileResponse(frontend_file)
    
    return JSONResponse(
        status_code=404,
        content={"detail": "Page not found", "path": full_path}
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