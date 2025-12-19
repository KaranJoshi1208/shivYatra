"""
Database Configuration for Yatri Tourism Vector Store
Configuration settings for ChromaDB vector database
"""

from pathlib import Path

# Database paths
CONFIG_ROOT = Path(__file__).parent
APP_ROOT = CONFIG_ROOT.parent
PROJECT_ROOT = APP_ROOT.parent
CHROMA_DB_PATH = PROJECT_ROOT / "database"
EMBEDDINGS_SOURCE_PATH = PROJECT_ROOT / "data" / "embeddings"

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