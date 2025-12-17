"""
Database Configuration for ShivYatra Tourism Vector Store
Configuration settings for ChromaDB vector database
"""

from pathlib import Path

# Database paths
DB_ROOT = Path(__file__).parent.parent
CHROMA_DB_PATH = DB_ROOT / "chromadb"
EMBEDDINGS_SOURCE_PATH = DB_ROOT.parent / "rag" / "embeddings"

# Collection settings
COLLECTION_NAME = "tourism_embeddings_minilm"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# ChromaDB settings
CHROMA_SETTINGS = {
    "allow_reset": True,
    "anonymized_telemetry": False,
}

# Batch processing settings
BATCH_SIZE = 100
MAX_RETRIES = 3

# Metadata schema for tourism chunks
METADATA_SCHEMA = {
    "chunk_id": "string",
    "city": "string", 
    "state": "string",
    "country": "string",
    "category": "string",
    "subcategory": "string",
    "price_range": "string",
    "has_contact": "boolean",
    "adventure_score": "int",
    "family_score": "int", 
    "solo_traveler_score": "int",
    "content_length": "int"
}

# Query settings
DEFAULT_QUERY_LIMIT = 10
SIMILARITY_THRESHOLD = 0.7

print("✅ Database configuration loaded")