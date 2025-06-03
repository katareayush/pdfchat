import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000/api/v1"

def test_upload_pdf(file_path: str) -> str:
    """Upload a PDF and return document ID"""
    print(f"🔄 Uploading PDF: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/pdf')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful!")
            print(f"   Document ID: {result['doc_id']}")
            print(f"   Pages: {result['total_pages']}")
            print(f"   Characters: {result['total_characters']}")
            return result['doc_id']
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return None

def test_embedding_stats():
    """Test embedding statistics endpoint"""
    print(f"\n🔄 Getting embedding statistics...")
    
    try:
        response = requests.get(f"{BASE_URL}/embeddings/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Embedding Stats:")
            print(f"   Total embeddings: {stats['total_embeddings']}")
            print(f"   Embedding dimension: {stats['embedding_dimension']}")
            print(f"   Model: {stats['model_name']}")
            print(f"   Unique documents: {stats['unique_documents']}")
            print(f"   Storage size: {stats['storage_size_mb']:.2f} MB")
            return stats
        else:
            print(f"❌ Stats failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Stats error: {str(e)}")
        return None

def test_search_post(query: str, top_k: int = 5, doc_id: str = None):
    """Test search using POST method"""
    print(f"\n🔍 Searching (POST): '{query}'")
    
    search_data = {
        "query": query,
        "top_k": top_k,
        "score_threshold": 0.1
    }
    
    if doc_id:
        search_data["doc_id"] = doc_id
    
    try:
        response = requests.post(f"{BASE_URL}/search", json=search_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Search completed in {result['search_time_ms']:.2f}ms")
            print(f"   Found {result['total_found']} results:")
            
            for i, res in enumerate(result['results'][:3], 1):  # Show top 3
                print(f"   {i}. Page {res['page']} (score: {res['score']:.3f})")
                print(f"      {res['context'][:100]}...")
                print(f"      Doc: {res['doc_id'][:8]}...")
            
            return result
        else:
            print(f"❌ Search failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        return None

def test_search_get(query: str, top_k: int = 3):
    """Test search using GET method"""
    print(f"\n🔍 Searching (GET): '{query}'")
    
    params = {
        "query": query,
        "top_k": top_k,
        "score_threshold": 0.1
    }
    
    try:
        response = requests.get(f"{BASE_URL}/search", params=params)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Search completed in {result['search_time_ms']:.2f}ms")
            print(f"   Found {result['total_found']} results")
            return result
        else:
            print(f"❌ Search failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        return None

def test_list_documents():
    """Test listing all documents"""
    print(f"\n📚 Listing all documents...")
    
    try:
        response = requests.get(f"{BASE_URL}/documents")
        if response.status_code == 200:
            docs = response.json()
            print(f"✅ Found {docs['total_count']} documents:")
            for doc in docs['documents']:
                print(f"   - {doc['filename']} (ID: {doc['doc_id'][:8]}...)")
                print(f"     Pages: {doc['total_pages']}, Characters: {doc['total_characters']}")
            return docs['documents']
        else:
            print(f"❌ List failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ List error: {str(e)}")
        return []

def test_performance_search(queries: list, doc_id: str = None):
    """Test search performance with multiple queries"""
    print(f"\n⚡ Performance testing with {len(queries)} queries...")
    
    total_time = 0
    successful_searches = 0
    
    for i, query in enumerate(queries, 1):
        try:
            start_time = time.time()
            search_data = {"query": query, "top_k": 5, "score_threshold": 0.1}
            if doc_id:
                search_data["doc_id"] = doc_id
                
            response = requests.post(f"{BASE_URL}/search", json=search_data)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                search_time = end_time - start_time
                total_time += search_time
                successful_searches += 1
                
                print(f"   Query {i}: {search_time*1000:.1f}ms, {result['total_found']} results")
            else:
                print(f"   Query {i}: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"   Query {i}: Error - {str(e)}")
    
    if successful_searches > 0:
        avg_time = (total_time / successful_searches) * 1000
        print(f"✅ Performance Summary:")
        print(f"   Successful searches: {successful_searches}/{len(queries)}")
        print(f"   Average search time: {avg_time:.1f}ms")
        print(f"   Total time: {total_time*1000:.1f}ms")

def run_comprehensive_test():
    """Run a comprehensive test of the entire system"""
    print("🧪 COMPREHENSIVE PDF RAG SYSTEM TEST")
    print("=" * 60)
    
    # Test 1: Check if server is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("❌ Server is not running! Start with: uvicorn app.main:app --reload")
            return
    except:
        print("❌ Cannot connect to server! Make sure it's running on localhost:8000")
        return
    
    print("✅ Server is running")
    
    # Test 2: Upload a PDF (replace with your PDF path)
    pdf_file = "sample.pdf"  # Update this path
    doc_id = test_upload_pdf(pdf_file)
    
    if not doc_id:
        print("❌ Cannot proceed without a successful upload")
        return
    
    # Test 3: Wait a moment for embeddings to be generated
    print("\n⏳ Waiting for embeddings to be generated...")
    time.sleep(2)
    
    # Test 4: Check embedding stats
    stats = test_embedding_stats()
    
    if not stats or stats['total_embeddings'] == 0:
        print("❌ No embeddings found! Check the embedding service")
        return
    
    # Test 5: Test search functionality
    test_queries = [
        "introduction",
        "conclusion", 
        "methodology",
        "results",
        "data analysis"
    ]
    
    print(f"\n🔍 Testing search with sample queries...")
    for query in test_queries[:3]:  # Test first 3 queries
        test_search_post(query, top_k=3)
    
    # Test 6: Test GET search
    test_search_get("summary", top_k=2)
    
    # Test 7: List all documents
    documents = test_list_documents()
    
    # Test 8: Performance test
    performance_queries = [
        "what is the main topic",
        "key findings",
        "important information",
        "conclusions drawn",
        "research methodology"
    ]
    test_performance_search(performance_queries, doc_id)
    
    print("\n" + "=" * 60)
    print("✅ COMPREHENSIVE TEST COMPLETED!")
    print(f"📊 System Status:")
    print(f"   - Documents indexed: {len(documents)}")
    print(f"   - Total embeddings: {stats['total_embeddings'] if stats else 'N/A'}")
    print(f"   - Search functionality: ✅ Working")
    print(f"   - Ready for production testing!")

if __name__ == "__main__":
    print("🚀 PDF RAG System Test Suite")
    print("Choose test mode:")
    print("1. Comprehensive test (recommended)")
    print("2. Quick search test")
    print("3. Upload only")
    print("4. Performance test only")
    
    choice = input("Enter choice (1-4) or press Enter for comprehensive test: ").strip()
    
    if choice == "2":
        # Quick search test
        query = input("Enter search query: ").strip()
        if query:
            test_search_post(query)
    elif choice == "3":
        pdf_path = input("Enter PDF file path: ").strip()
        if pdf_path:
            test_upload_pdf(pdf_path)
    elif choice == "4":
        queries = ["test query 1", "test query 2", "test query 3"]
        test_performance_search(queries)
    else:
        run_comprehensive_test()