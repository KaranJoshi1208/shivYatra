"""
ChromaDB Vector Database Initialization for ShivYatra Tourism Data
Creates and populates ChromaDB with tourism embeddings generated from all-MiniLM-L6-v2
"""

import json
import numpy as np
import chromadb
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import sys
import time

# Add config to path
sys.path.append(str(Path(__file__).parent.parent / "config"))
from db_config import *

class TourismVectorDB:
    """
    Tourism Vector Database Manager using ChromaDB
    Handles initialization, population, and basic querying of tourism embeddings
    """
    
    def __init__(self):
        """Initialize ChromaDB client and settings"""
        self.client = None
        self.collection = None
        self.embeddings_loaded = False
        
    def initialize_database(self) -> bool:
        """Initialize ChromaDB client and create collection"""
        try:
            print("🔄 Initializing ChromaDB...")
            
            # Create ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DB_PATH),
                settings=chromadb.Settings(**CHROMA_SETTINGS)
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "description": "Tourism data embeddings for ShivYatra",
                    "embedding_model": EMBEDDING_MODEL,
                    "dimensions": EMBEDDING_DIMENSIONS,
                    "created_at": str(time.time())
                }
            )
            
            print(f"✅ ChromaDB initialized successfully!")
            print(f"📁 Database path: {CHROMA_DB_PATH}")
            print(f"🗂️  Collection: {COLLECTION_NAME}")
            print(f"📊 Embedding dimensions: {EMBEDDING_DIMENSIONS}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize ChromaDB: {str(e)}")
            return False
    
    def load_embeddings_data(self) -> tuple:
        """Load embeddings and metadata from JSON file"""
        try:
            embeddings_file = EMBEDDINGS_SOURCE_PATH / "tourism_embeddings_all_MiniLM_L6_v2.json"
            
            if not embeddings_file.exists():
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
            
            print(f"📂 Loading embeddings from: {embeddings_file}")
            
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embedding_data = json.load(f)
            
            print(f"✅ Loaded {len(embedding_data):,} embedding entries")
            
            # Extract components for ChromaDB
            documents = []
            embeddings = []
            metadatas = []
            ids = []
            
            for entry in embedding_data:
                # Document content
                documents.append(entry['content'])
                
                # Embedding vectors
                embeddings.append(entry['embedding'])
                
                # Metadata (flattened for ChromaDB)
                metadata = {
                    "chunk_id": entry['chunk_id'],
                    "city": entry['metadata']['location']['city'],
                    "state": entry['metadata']['location']['state'], 
                    "country": entry['metadata']['location']['country'],
                    "category": entry['metadata']['classification']['category'],
                    "subcategory": entry['metadata']['classification']['subcategory'],
                    "price_range": entry['metadata']['practical_info'].get('price_range', 'unknown'),
                    "has_contact": entry['metadata']['practical_info'].get('has_contact', False),
                    "adventure_score": int(entry['metadata']['relevance_scores']['adventure']),
                    "family_score": int(entry['metadata']['relevance_scores']['family']),
                    "solo_traveler_score": int(entry['metadata']['relevance_scores']['solo_traveler']),
                    "content_length": len(entry['content'])
                }
                metadatas.append(metadata)
                
                # Unique IDs
                ids.append(entry['chunk_id'])
            
            self.embeddings_loaded = True
            
            print(f"📊 Prepared data summary:")
            print(f"   Documents: {len(documents):,}")
            print(f"   Embeddings: {len(embeddings):,} x {len(embeddings[0])}")
            print(f"   Metadata fields: {len(metadatas[0])}")
            print(f"   Sample metadata: {list(metadatas[0].keys())}")
            
            return documents, embeddings, metadatas, ids
            
        except Exception as e:
            print(f"❌ Failed to load embeddings: {str(e)}")
            return None, None, None, None
    
    def populate_database(self, documents: List[str], embeddings: List[List[float]], 
                         metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
        """Populate ChromaDB with embeddings in batches"""
        try:
            print(f"🚀 Populating ChromaDB with {len(documents):,} entries...")
            print(f"🔧 Batch size: {BATCH_SIZE}")
            
            # Clear existing data if any
            existing_count = self.collection.count()
            if existing_count > 0:
                print(f"🗑️  Clearing {existing_count} existing entries...")
                self.collection.delete(where={})
            
            # Add data in batches
            total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Populating database"):
                end_idx = min(i + BATCH_SIZE, len(documents))
                
                batch_documents = documents[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                batch_ids = ids[i:end_idx]
                
                # Add batch to collection
                self.collection.add(
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            # Verify population
            final_count = self.collection.count()
            
            print(f"✅ Database population complete!")
            print(f"📊 Total entries: {final_count:,}")
            print(f"🎯 Expected: {len(documents):,}")
            print(f"✓ Match: {final_count == len(documents)}")
            
            return final_count == len(documents)
            
        except Exception as e:
            print(f"❌ Failed to populate database: {str(e)}")
            return False
    
    def verify_database(self) -> bool:
        """Verify database integrity with sample queries"""
        try:
            print("🔍 Verifying database integrity...")
            
            # Check collection exists and has data
            count = self.collection.count()
            if count == 0:
                print("❌ Database is empty!")
                return False
            
            # Test basic query
            results = self.collection.query(
                query_texts=["adventure activities in mountains"],
                n_results=3
            )
            
            if not results or not results['documents']:
                print("❌ Query test failed!")
                return False
            
            print(f"✅ Database verification successful!")
            print(f"📊 Total entries: {count:,}")
            print(f"🔍 Sample query returned: {len(results['documents'][0])} results")
            print(f"📄 Sample result: {results['documents'][0][0][:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Database verification failed: {str(e)}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            count = self.collection.count()
            
            # Sample some entries to analyze
            sample_results = self.collection.get(limit=min(100, count), include=['metadatas'])
            
            if sample_results and sample_results['metadatas']:
                # Analyze categories
                categories = [meta['category'] for meta in sample_results['metadatas']]
                states = [meta['state'] for meta in sample_results['metadatas']]
                
                stats = {
                    "total_entries": count,
                    "sample_categories": list(set(categories)),
                    "sample_states": list(set(states)),
                    "collection_name": COLLECTION_NAME,
                    "embedding_model": EMBEDDING_MODEL,
                    "dimensions": EMBEDDING_DIMENSIONS
                }
            else:
                stats = {
                    "total_entries": count,
                    "collection_name": COLLECTION_NAME,
                    "embedding_model": EMBEDDING_MODEL,
                    "dimensions": EMBEDDING_DIMENSIONS
                }
                
            return stats
            
        except Exception as e:
            print(f"❌ Failed to get database stats: {str(e)}")
            return {}


def main():
    """Main function to initialize and populate ChromaDB"""
    print("🎯 CHROMADB VECTOR DATABASE INITIALIZATION")
    print("=" * 50)
    
    # Initialize database manager
    db_manager = TourismVectorDB()
    
    # Step 1: Initialize ChromaDB
    if not db_manager.initialize_database():
        print("❌ Failed to initialize database. Exiting.")
        return False
    
    # Step 2: Load embeddings data
    documents, embeddings, metadatas, ids = db_manager.load_embeddings_data()
    if not db_manager.embeddings_loaded:
        print("❌ Failed to load embeddings. Exiting.")
        return False
    
    # Step 3: Populate database
    if not db_manager.populate_database(documents, embeddings, metadatas, ids):
        print("❌ Failed to populate database. Exiting.")
        return False
    
    # Step 4: Verify database
    if not db_manager.verify_database():
        print("❌ Database verification failed. Exiting.")
        return False
    
    # Step 5: Display final statistics
    stats = db_manager.get_database_stats()
    if stats:
        print(f"\n📊 FINAL DATABASE STATISTICS")
        print("─" * 30)
        print(f"✅ Total entries: {stats['total_entries']:,}")
        print(f"🏷️  Collection: {stats['collection_name']}")
        print(f"🤖 Model: {stats['embedding_model']}")
        print(f"📐 Dimensions: {stats['dimensions']}")
        if 'sample_categories' in stats:
            print(f"📂 Categories: {', '.join(stats['sample_categories'][:5])}...")
            print(f"🗺️  States: {', '.join(stats['sample_states'][:5])}...")
    
    print(f"\n🎉 ChromaDB setup complete!")
    print(f"📁 Database location: {CHROMA_DB_PATH}")
    print(f"🚀 Ready for RAG queries and tourism recommendations!")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Vector database initialization successful!")
    else:
        print("\n❌ Vector database initialization failed!")
    exit(0 if success else 1)