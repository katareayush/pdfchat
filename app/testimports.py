try:
    import fastapi
    print("✅ FastAPI imported successfully")
    
    import uvicorn
    print("✅ Uvicorn imported successfully")
    
    import pdfplumber
    print("✅ PDFplumber imported successfully")
    
    import faiss
    print("✅ FAISS imported successfully")
    
    import chromadb
    print("✅ ChromaDB imported successfully")
    
    from sentence_transformers import SentenceTransformer
    print("✅ Sentence Transformers imported successfully")
    
    print("\n🎉 All packages installed correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")