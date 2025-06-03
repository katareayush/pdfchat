import requests
import json
from pathlib import Path

def test_upload_pdf(file_path: str, base_url: str = "http://localhost:8000"):
    """Test PDF upload functionality"""
    
    try:
        # Upload PDF
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/pdf')}
            response = requests.post(f"{base_url}/api/v1/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Upload successful!")
            print(f"Document ID: {result['doc_id']}")
            print(f"Total pages: {result['total_pages']}")
            print(f"Total characters: {result['total_characters']}")
            print(f"Pages with text: {result['pages_with_text']}")
            
            # Test getting document info
            doc_id = result['doc_id']
            doc_response = requests.get(f"{base_url}/api/v1/document/{doc_id}")
            if doc_response.status_code == 200:
                doc_data = doc_response.json()
                print(f"\n📄 First page preview:")
                if doc_data['pages'] and doc_data['pages'][0]['text_chunk']:
                    preview = doc_data['pages'][0]['text_chunk'][:200]
                    print(f"{preview}...")
                else:
                    print("No text found on first page")
            
            # Test document summary
            summary_response = requests.get(f"{base_url}/api/v1/document/{doc_id}/summary")
            if summary_response.status_code == 200:
                summary = summary_response.json()
                print(f"\n📊 Document Summary:")
                print(f"  Average chars per page: {summary['average_chars_per_page']:.0f}")
                print(f"  Empty pages: {summary['empty_pages']}")
                print(f"  Upload time: {summary['upload_time']}")
            
            return result['doc_id']
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            return None
            
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Make sure the server is running on localhost:8000")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def test_list_documents(base_url: str = "http://localhost:8000"):
    """Test listing all documents"""
    try:
        response = requests.get(f"{base_url}/api/v1/documents")
        if response.status_code == 200:
            docs = response.json()
            print(f"\n📚 Total documents: {docs['total_count']}")
            for doc in docs['documents']:
                print(f"  - {doc['filename']} ({doc['total_pages']} pages, {doc['total_characters']} chars)")
        else:
            print(f"❌ Failed to list documents: {response.status_code}")
    except Exception as e:
        print(f"❌ Error listing documents: {str(e)}")

def test_get_page(doc_id: str, page_number: int = 1, base_url: str = "http://localhost:8000"):
    """Test getting specific page"""
    try:
        response = requests.get(f"{base_url}/api/v1/document/{doc_id}/page/{page_number}")
        if response.status_code == 200:
            page_data = response.json()
            print(f"\n📃 Page {page_number} ({page_data['char_count']} characters):")
            if page_data['text_chunk']:
                preview = page_data['text_chunk'][:300]
                print(f"{preview}...")
            else:
                print("Empty page")
        else:
            print(f"❌ Failed to get page: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting page: {str(e)}")

if __name__ == "__main__":
    print("🧪 Testing PDF Upload & Extraction")
    print("=" * 50)
    
    # Replace with your PDF file path
    pdf_file = "sample.pdf"  # Update this path
    
    # Test upload
    doc_id = test_upload_pdf(pdf_file)
    
    if doc_id:
        # Test getting specific page
        test_get_page(doc_id, 1)
    
    # Test listing documents
    test_list_documents()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("Visit http://localhost:8000/docs for interactive API documentation")