# 🏔️ ShivYatra Tourism Chatbot - RAG Pipeline

**AI-Powered Tourism Assistant for India** - Combining RAG (Retrieval-Augmented Generation) with Ollama LLM and ChromaDB vector store.

## 🎯 Overview

This is a complete RAG-powered chatbot system that helps users explore incredible destinations across India. It combines:

- **🦙 Ollama** (qwen2.5:1.5b) - Local LLM for response generation
- **📊 ChromaDB** - Vector database with 4,160+ tourism embeddings  
- **🎨 Gradio** - Beautiful web interface
- **🔍 Semantic Search** - Intelligent content retrieval

## 🏗️ Architecture

```
rag_pipeline/
├── app.py                    # Main application launcher ✅
├── config/
│   └── rag_config.py        # RAG pipeline configuration ✅
├── src/
│   └── rag_engine.py        # Core RAG logic ✅
├── ui/
│   └── chatbot_ui.py        # Gradio web interface ✅
├── requirements.txt         # Dependencies ✅
└── README.md               # This file ✅
```

## 🚀 Quick Start

### Prerequisites

1. **Ollama installed and running**:
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the model
   ollama pull qwen2.5:1.5b
   
   # Start Ollama server
   ollama serve
   ```

2. **Vector database initialized**:
   ```bash
   # Should already be done from vector_db setup
   ls ../vector_db/chromadb/  # Should contain chroma.sqlite3
   ```

### Launch Chatbot

```bash
cd /home/karan/shivYatra/rag_pipeline
python app.py
```

The chatbot will be available at: **http://localhost:7860**

## 🔧 Configuration

### RAG Settings (`config/rag_config.py`)

```python
# LLM Configuration
OLLAMA_CONFIG = {
    "model": "qwen2.5:1.5b",
    "temperature": 0.7,
    "max_tokens": 1000
}

# Vector Search
CHROMADB_CONFIG = {
    "similarity_threshold": 0.3,
    "max_results": 5
}

# UI Settings
UI_CONFIG = {
    "server_port": 7860,
    "theme": "soft"
}
```

## 💬 Features

### 🎯 **Smart Tourism Assistant**
- **Destination Discovery**: Find places in Himachal, Uttarakhand, J&K, Ladakh
- **Activity Recommendations**: Adventure, cultural, spiritual experiences
- **Budget Planning**: Options for all budget ranges
- **Personalized Suggestions**: Solo, family, adventure travelers

### 🔍 **Advanced RAG Capabilities**
- **Semantic Search**: Natural language understanding
- **Context-Aware**: Retrieves relevant tourism information
- **Source Attribution**: Shows information sources
- **Real-time Processing**: Fast response generation

### 🎨 **Beautiful Interface**
- **Responsive Design**: Works on all devices
- **Chat History**: Maintains conversation context
- **Example Questions**: Quick-start options
- **System Status**: Health monitoring
- **Rich Formatting**: Markdown support with emojis

## 🧪 Testing

### Manual Testing
```bash
# Test RAG engine directly
cd src
python rag_engine.py
```

### Example Queries
- *"What are the best adventure activities in Manali?"*
- *"I'm planning a family trip to Himachal Pradesh. Any suggestions?"*
- *"What are some budget-friendly places in Uttarakhand?"*
- *"Tell me about spiritual places in Haridwar"*
- *"I'm a solo traveler interested in trekking. Where should I go?"*

## 📊 System Requirements

### Minimum
- **RAM**: 4GB (for embeddings + LLM)
- **Storage**: 2GB free space
- **CPU**: Multi-core recommended
- **Network**: For Ollama downloads

### Recommended
- **RAM**: 8GB+ 
- **GPU**: For faster LLM inference (optional)
- **SSD**: For faster vector search

## 🔍 How It Works

### RAG Pipeline Flow

1. **User Query** → Gradio Interface
2. **Query Encoding** → SentenceTransformer embeddings
3. **Vector Search** → ChromaDB similarity search
4. **Context Retrieval** → Top-K relevant documents
5. **Prompt Engineering** → Format context + query
6. **LLM Generation** → Ollama (qwen2.5:1.5b) response
7. **Response Formatting** → Markdown + sources
8. **UI Display** → Gradio chat interface

### Data Flow
```
User Query → Embedding → Vector Search → Context Docs → LLM → Response
     ↓                        ↑                           ↑
   Gradio UI ←─── Format ←─── ChromaDB ←──── Prompt ←─────┘
```

## 🎛️ Customization

### Adding New Data Sources
1. Update vector database with new embeddings
2. Restart RAG pipeline to reload

### Changing LLM Model
```python
# In config/rag_config.py
OLLAMA_CONFIG["model"] = "llama2:7b"  # Or any other Ollama model
```

### UI Theming
```python
# In config/rag_config.py
UI_CONFIG["theme"] = "monochrome"  # Or "dark", "soft", etc.
```

## 🚨 Troubleshooting

### Common Issues

1. **"Ollama service not running"**
   ```bash
   ollama serve
   # In another terminal: ollama list (check models)
   ```

2. **"Vector database not found"**
   ```bash
   cd ../vector_db
   python scripts/initialize_db.py
   ```

3. **"Model not found"**
   ```bash
   ollama pull qwen2.5:1.5b
   ```

4. **Slow responses**
   - Reduce `max_results` in ChromaDB config
   - Use smaller LLM model
   - Check system resources

5. **Memory issues**
   - Reduce `max_tokens` in Ollama config
   - Close other applications
   - Restart the pipeline

### Debug Mode
```bash
# Enable debug output
export RAG_DEBUG=1
python app.py
```

## 📈 Performance Metrics

### Typical Performance
- **Query Processing**: 2-5 seconds
- **Vector Search**: <0.5 seconds  
- **LLM Generation**: 1-4 seconds
- **Memory Usage**: 1-2GB
- **Context Window**: 4000 tokens

### Optimization Tips
- **GPU Acceleration**: Use GPU-enabled Ollama
- **Model Quantization**: Use quantized models
- **Batch Processing**: For multiple queries
- **Caching**: Implement response caching

## 🔗 Integration Points

### API Integration
The RAG engine can be used programmatically:
```python
from src.rag_engine import create_rag_pipeline

rag = create_rag_pipeline()
result = rag.chat("Your query here")
```

### External Services
- **REST API**: Can be wrapped with FastAPI
- **Webhooks**: For chat platform integration
- **Streaming**: Real-time response streaming

## 📝 Development

### Code Structure
- **`rag_engine.py`**: Core RAG logic, stateless design
- **`chatbot_ui.py`**: Gradio interface, UI state management
- **`rag_config.py`**: Centralized configuration
- **`app.py`**: Application launcher with health checks

### Best Practices
- **Error Handling**: Graceful failure recovery
- **Logging**: Comprehensive system logging
- **Configuration**: Environment-based settings
- **Testing**: Automated testing framework

## 🛡️ Security Notes

- **Local Deployment**: Runs entirely on local machine
- **No Data Transmission**: Tourism data stays local
- **Privacy**: No external API calls for core functionality
- **Ollama Security**: Follow Ollama security guidelines

## 🗺️ Roadmap

### Planned Features
- [ ] **Multi-language Support**: Hindi, other regional languages
- [ ] **Advanced Filtering**: Date ranges, weather, seasons
- [ ] **Trip Planning**: Multi-day itinerary generation
- [ ] **Image Integration**: Destination photos and maps
- [ ] **Voice Interface**: Speech-to-text input
- [ ] **Mobile App**: React Native or Flutter
- [ ] **Analytics**: Usage analytics and insights
- [ ] **Real-time Updates**: Live data integration

### Technical Improvements
- [ ] **Caching Layer**: Redis for response caching
- [ ] **API Gateway**: REST API with authentication
- [ ] **Monitoring**: Prometheus + Grafana
- [ ] **Docker**: Containerization
- [ ] **CI/CD**: Automated testing and deployment

---

## 📞 Support

For issues or questions:
1. Check this README
2. Review configuration files
3. Test components individually
4. Check Ollama and ChromaDB logs

**Ready to explore India with AI assistance!** 🇮🇳✨