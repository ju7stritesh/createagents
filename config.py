from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
LLAMA_PARSE_KEY = os.getenv('LLAMA_PARSE_KEY')

# Model Configurations
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"

# Document Processing Configurations
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Configuration
RETRIEVER_K = 4  # Number of documents to retrieve for each query

# Chroma Configuration
INDEX_PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "document_store"

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'xlsx', 'csv'}
    CHUNK_SIZE = CHUNK_SIZE
    CHUNK_OVERLAP = CHUNK_OVERLAP 