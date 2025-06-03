import os
from pathlib import Path

class RailwayConfig:
    """Railway-specific configuration"""
    
    # Railway automatically sets PORT
    PORT = int(os.getenv("PORT", 8000))
    
    # Railway provides a persistent disk at /app
    DATA_DIR = Path("/app/data")
    UPLOAD_DIR = Path("/app/uploads") 
    MODEL_CACHE_DIR = Path("/app/models")
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    
    # Railway domain (will be set after deployment)
    RAILWAY_DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN", "localhost")
    
    # CORS for Railway
    CORS_ORIGINS = [
        f"https://{RAILWAY_DOMAIN}",
        f"https://www.{RAILWAY_DOMAIN}",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ]
    
    # Railway environment detection
    IS_RAILWAY = os.getenv("RAILWAY_ENVIRONMENT") is not None
    
    # Memory optimization for Railway (1GB limit on starter)
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB instead of 50MB
    
    @classmethod
    def get_settings(cls):
        return {
            "port": cls.PORT,
            "host": "0.0.0.0",
            "cors_origins": cls.CORS_ORIGINS,
            "data_dir": str(cls.DATA_DIR),
            "upload_dir": str(cls.UPLOAD_DIR),
            "max_file_size": cls.MAX_FILE_SIZE,
            "is_railway": cls.IS_RAILWAY
        }