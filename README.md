# Yatri - AI Tourism Assistant

Local RAG-powered chatbot for Indian travel destinations using Ollama + ChromaDB.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed

## Build & Run

### 1. Clone & Install

```bash
git clone https://github.com/your-repo/shivYatra.git
cd shivYatra
pip install -r app/requirements.txt
```

### 2. Setup Ollama

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull qwen2.5:1.5b

# Start Ollama (keep running)
ollama serve
```

### 3. Run Application

```bash
python app/run.py
```

Open **http://localhost:5000** in your browser.

## Project Structure

```
shivYatra/
├── app/
│   ├── api/server.py        # Flask server
│   ├── core/rag_engine.py   # RAG pipeline
│   ├── config/              # Configuration
│   ├── web/templates/       # Chat UI
│   └── run.py               # Entry point
├── data/                    # Tourism data
├── database/                # ChromaDB vectors
└── notebooks/               # Data processing
```

## Configuration

Edit `app/config/rag_config.py`:

```python
OLLAMA_CONFIG = {
    "model": "qwen2.5:1.5b",    # Change LLM model
    "temperature": 0.7,
    "max_tokens": 1000
}
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Ollama not running | Run `ollama serve` in terminal |
| Model not found | Run `ollama pull qwen2.5:1.5b` |
| Port 5000 in use | Kill process: `lsof -ti:5000 \| xargs kill` |

## Tech Stack

- **LLM**: Ollama (qwen2.5:1.5b)
- **Vector DB**: ChromaDB
- **Embeddings**: all-MiniLM-L6-v2
- **Backend**: Flask
- **Frontend**: HTML/CSS/JS