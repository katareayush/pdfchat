import os
from pathlib import Path

class RailwayConfig:
    """Railway-specific configuration"""
    
    PORT = int(os.getenv("PORT", 8000))
    
    DATA_DIR = Path("/app/data")
    UPLOAD_DIR = Path("/app/uploads") 
    MODEL_CACHE_DIR = Path("/app/models")
    
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    
    RAILWAY_DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN", "localhost")
    
    CORS_ORIGINS = [
        f"https://{RAILWAY_DOMAIN}",
        f"https://www.{RAILWAY_DOMAIN}",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ]
    
    IS_RAILWAY = os.getenv("RAILWAY_ENVIRONMENT") is not None
    
    MAX_FILE_SIZE = 25 * 1024 * 1024  
    
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