"""ShivYatra RAG Pipeline Engine
Core RAG functionality connecting ChromaDB vector store with Ollama LLM
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
import requests
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent / "config"))
from rag_config import *

sys.path.append(str(VECTOR_DB_PATH / "config"))
from db_config import CHROMA_DB_PATH, COLLECTION_NAME

class ShivYatraRAG:
    """RAG Pipeline - Tourism Assistant with Vector Retrieval & LLM Generation"""
    def __init__(self):
        """Initialize RAG pipeline components"""
        self.vector_client = None
        self.collection = None
        self.embedding_model = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize all RAG components"""
        try:
            if not self._init_vector_store():
                return False
            if not self._init_embedding_model():
                return False
            if not self._test_ollama_connection():
                return False
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"RAG initialization failed: {str(e)}")
            return False
    
    def _init_vector_store(self) -> bool:
        """Initialize ChromaDB connection"""
        try:
            self.vector_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            self.collection = self.vector_client.get_collection(
                name=CHROMADB_CONFIG["collection_name"]
            )
            return True
        except Exception as e:
            print(f"Vector store connection failed: {str(e)}")
            return False
    
    def _init_embedding_model(self) -> bool:
        """Initialize sentence transformer model for query encoding"""
        try:
            self.embedding_model = SentenceTransformer(CHROMADB_CONFIG["embedding_model"])
            return True
        except Exception as e:
            print(f"Embedding model loading failed: {str(e)}")
            return False
    
    def _test_ollama_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{OLLAMA_CONFIG['base_url']}/api/tags", timeout=10)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if OLLAMA_CONFIG['model'] in model_names:
                    return True
                else:
                    print(f"Model {OLLAMA_CONFIG['model']} not found. Available: {model_names}")
                    return False
            else:
                print(f"Ollama server not responding: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Ollama connection failed: {str(e)}")
            print("Make sure Ollama is running: ollama serve")
            return False
    
    def retrieve_context(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context from vector store"""
        if not self.is_initialized:
            return []
        
        try:
            max_results = max_results or CHROMADB_CONFIG["max_results"]
            
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            context_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    similarity = 1 - results['distances'][0][i]
                    
                    if similarity >= RAG_SETTINGS["relevance_threshold"]:
                        context_docs.append({
                            'content': doc,
                            'metadata': results['metadatas'][0][i],
                            'similarity': round(similarity, 3),
                            'rank': i + 1
                        })
            
            return context_docs
            
        except Exception as e:
            print(f"Context retrieval failed: {str(e)}")
            return []
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using Ollama LLM with retrieved context"""
        try:
            if context_docs:
                context_text = self._format_context(context_docs)
                prompt = CONTEXT_PROMPT_TEMPLATE.format(
                    context=context_text,
                    question=query
                )
            else:
                prompt = f"{FALLBACK_PROMPT}\n\nUser question: {query}"
            
            response = self._call_ollama(prompt)
            return response
            
        except Exception as e:
            print(f"Response generation failed: {str(e)}")
            return "I'm having trouble generating a response right now. Please try again."
    
    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string"""
        formatted_context = []
        
        for doc in context_docs[:RAG_SETTINGS["max_context_chunks"]]:
            metadata = doc['metadata']
            content = doc['content']
            
            context_entry = f"""**{metadata['city']}, {metadata['state']}**
Category: {metadata['category']} → {metadata['subcategory']}
Budget: {metadata['price_range']}
{content}
---"""
            formatted_context.append(context_entry)
        
        return "\\n".join(formatted_context)
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for text generation"""
        try:
            payload = {
                "model": OLLAMA_CONFIG["model"],
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": OLLAMA_CONFIG["temperature"],
                    "num_predict": OLLAMA_CONFIG["max_tokens"]
                }
            }
            
            response = requests.post(
                f"{OLLAMA_CONFIG['base_url']}/api/generate",
                json=payload,
                timeout=OLLAMA_CONFIG["timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"LLM Error: {response.status_code}"
                
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat function - complete RAG pipeline"""
        start_time = time.time()
        
        if not self.is_initialized:
            return {
                "response": "RAG pipeline not initialized. Please restart the application.",
                "context_docs": [],
                "processing_time": 0,
                "error": "Not initialized"
            }
        
        try:
            context_docs = self.retrieve_context(query)
            response = self.generate_response(query, context_docs)
            
            processing_time = round(time.time() - start_time, 2)
            
            return {
                "response": response,
                "context_docs": context_docs,
                "processing_time": processing_time,
                "query": query,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "response": f"Error processing your question: {str(e)}",
                "context_docs": [],
                "processing_time": round(time.time() - start_time, 2),
                "error": str(e)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        status = {
            "initialized": self.is_initialized,
            "vector_store": False,
            "embedding_model": False,
            "ollama": False,
            "total_embeddings": 0
        }
        
        if self.is_initialized:
            # Check vector store
            try:
                status["total_embeddings"] = self.collection.count()
                status["vector_store"] = True
            except:
                pass
            
            # Check embedding model
            if self.embedding_model:
                status["embedding_model"] = True
            
            # Check Ollama
            try:
                response = requests.get(f"{OLLAMA_CONFIG['base_url']}/api/tags", timeout=5)
                status["ollama"] = response.status_code == 200
            except:
                pass
        
        return status


def create_rag_pipeline() -> ShivYatraRAG:
    """Factory function to create and initialize RAG pipeline"""
    rag = ShivYatraRAG()
    
    if rag.initialize():
        print("ShivYatra RAG Pipeline ready!")
        return rag
    else:
        print("Failed to initialize RAG pipeline")
        return None


if __name__ == "__main__":
    print("Testing ShivYatra RAG Pipeline...")
    
    rag = create_rag_pipeline()
    
    if rag:
        test_query = "What are the best adventure activities in Manali?"
        print(f"\nTest Query: {test_query}")
        
        result = rag.chat(test_query)
        print(f"\nResponse: {result['response']}")
        print(f"Processing time: {result['processing_time']}s")
        print(f"Context documents: {len(result['context_docs'])}")
        
        health = rag.get_health_status()
        print(f"\nSystem Health: {health}")
    else:
        print("RAG pipeline test failed!")