"""RAG Pipeline Configuration for Yatri Tourism Chatbot"""

from pathlib import Path

# Project paths
APP_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = APP_ROOT.parent
DATABASE_PATH = PROJECT_ROOT / "database"
DATA_PATH = PROJECT_ROOT / "data"

# Ollama LLM Configuration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "model": "qwen2.5:1.5b",
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 60,
    "stream": True
}

# ChromaDB Configuration
CHROMADB_CONFIG = {
    "collection_name": "tourism_embeddings_minilm",
    "similarity_threshold": 0.3,
    "max_results": 5,
    "embedding_model": "all-MiniLM-L6-v2"
}

# RAG Pipeline Settings
RAG_SETTINGS = {
    "context_window": 4000,
    "max_context_chunks": 5,
    "relevance_threshold": 0.3,
    "fallback_enabled": True,
    "include_metadata": True
}

# Gradio UI Configuration
UI_CONFIG = {
    "title": "🏔️ ShivYatra - AI Tourism Assistant",
    "description": "ShivYatra Professional Tourism Platform - Discover incredible destinations across India with AI-powered recommendations",
    "theme": "soft",
    "server_port": 7860,
    "server_name": "localhost",
    "share": True,
    "debug": False
}

# Prompt Templates
SYSTEM_PROMPT = """You are Yatri, an expert AI tourism assistant specializing in Indian travel destinations. 
You help travelers discover amazing places, plan trips, and provide detailed information about destinations across India.

Your knowledge is based on comprehensive tourism data from Indian destinations including places in Himachal Pradesh, 
Uttarakhand, Jammu & Kashmir, Ladakh, and many other incredible locations.

Guidelines:
- Provide helpful, accurate, and engaging travel advice
- Include practical information like budget, activities, and traveler suitability
- Be enthusiastic about Indian tourism while being honest about challenges
- If you don't have specific information, say so and provide general guidance
- Always prioritize traveler safety and responsible tourism
"""

CONTEXT_PROMPT_TEMPLATE = """Based on the following tourism information about Indian destinations:

{context}

Please answer the user's question: {question}

Provide a helpful and detailed response including specific recommendations, practical tips, and relevant details from the provided context.
"""

FALLBACK_PROMPT = """I don't have specific information about that particular query in my current database. However, as a tourism assistant for India, I can provide some general guidance about travel in India. What specific aspect of Indian travel would you like to know about?"""

# Response formatting
RESPONSE_CONFIG = {
    "max_response_length": 800,
    "include_sources": True,
    "format_markdown": True,
    "show_confidence": False
}