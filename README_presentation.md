# ShivYatra AI Tourism Assistant — Executive Brief

A Retrieval-Augmented Generation (RAG) chatbot that delivers professional travel planning for Indian destinations (Himachal Pradesh, Uttarakhand, Jammu & Kashmir, Ladakh). Built for local, privacy-friendly execution with Ollama LLM + ChromaDB vector search and a Gradio web UI.

## Project Snapshot
- Purpose: AI travel consultation with curated Indian tourism knowledge.
- Stack: Python, Ollama (Qwen2.5:1.5B), ChromaDB, SentenceTransformers (all-MiniLM-L6-v2), Gradio UI.
- Data: 4,160+ destination chunks with metadata (city, state, category, budget cues, relevance scores).
- Footprint: Runs locally; no external paid APIs at runtime.

## Architecture (Layers)
1) UI (Gradio)
- Modern single-page chat UI (custom CSS) with sidebar, hero cards, chat pane, system status.

2) RAG Core
- Query → Embed (MiniLM) → Vector search (ChromaDB) → Context assembly → LLM generation (Ollama) → Response.

**RAG-Centric, Model-Agnostic Design**
- The LLM is a plug-in behind the retrieval layer. Swap `qwen2.5:1.5b` for any better/local/hosted model (e.g., Mistral, Llama, GPT via API) without changing retrieval, UI, or data.
- Gains from upgrades: higher quality generations, better reasoning, multilingual support—all while keeping the same curated context and guardrails.
- The retrieval pipeline (embeddings + ChromaDB) remains stable; only the `_call_ollama` (or equivalent) shim needs to point to the new provider.

3) Data Layer
- Vector DB: ChromaDB persistent store (tourism_embeddings_minilm collection).
- Embeddings: all-MiniLM-L6-v2; 4k+ documents.

4) LLM Layer
- Ollama local endpoint http://localhost:11434
- Model: qwen2.5:1.5b
- Usage: /api/generate (completion), /api/tags (health/model presence).

## Key Features
- Destination discovery and activity suggestions.
- Budget-aware guidance; traveler-type tailoring (solo, family, adventure).
- Context-grounded answers from curated corpus.
- Health/status panel; welcome prompts; example questions.
- Local-first: works offline after setup (except initial data prep).

## Repository Layout
- rag_pipeline/
  - app.py (launcher)
  - config/rag_config.py (LLM, vector, UI settings)
  - src/rag_engine.py (RAG core: retrieval + generation)
  - ui/chatbot_ui.py (Gradio interface)
- vector_db/
  - scripts/initialize_db.py (populate ChromaDB)
  - scripts/query_database.py (query/diagnose DB)
  - config/db_config.py (paths, collection settings)
- rag/ (notebooks for data extraction/embedding)
- optimized_chunks/, json_data/ (tourism data assets)

## Data & Embeddings
- Source: Cleaned tourism chunks (India-focused) via WikiVoyage scraping/processing notebooks (rag/).
- Embeddings: all-MiniLM-L6-v2 → stored in ChromaDB with metadata (location, category, subcategory, budget hints, relevance scores).

## Runtime Flow
1) User enters query in Gradio UI.
2) RAG engine retrieves top-k docs from ChromaDB (semantic search).
3) Context is formatted and sent to Ollama LLM.
4) LLM returns grounded answer; UI displays clean response.

## Config Highlights (rag_config.py)
- OLLAMA_CONFIG: base_url, model=qwen2.5:1.5b, temperature, max_tokens, timeout.
- CHROMADB_CONFIG: collection_name=tourism_embeddings_minilm, max_results=5, embedding_model=all-MiniLM-L6-v2.
- UI_CONFIG: title/description, port 7860, server_name localhost, share toggle, theme.

## Setup & Run (Local)
- Prereqs: Python 3.10+, Ollama installed and running; model pulled: `ollama pull qwen2.5:1.5b`.
- Install deps: `pip install -r rag_pipeline/requirements.txt` (and vector_db/requirements.txt if used separately).
- Initialize vector DB (once):
  - `cd vector_db && python scripts/initialize_db.py`
- Launch app:
  - `cd rag_pipeline && python app.py`
- Access UI: http://localhost:7860 (or generated gradio.live link if share=True).

## Operations & Health
- Vector DB health: collection.count(), verify queries in query_database.py.
- LLM health: GET /api/tags (rag_engine.py check).
- UI status: system status panel within the sidebar.

## Performance Notes
- Typical end-to-end: 2–5s (vector search <0.5s; LLM 1–4s).
- Context window: 4k tokens; max_context_chunks=5.
- Tune: reduce max_results or max_tokens for speed; lower temperature for determinism.

## Deployment Options
- Local with Gradio share (ephemeral gradio.live URL).
- LAN exposure: set server_name to 0.0.0.0 and use host IP:7860.
- Cloud/Custom domain: wrap with FastAPI + reverse proxy; deploy to Render/Railway/Vercel/HF Spaces.

## Limitations / Next Steps
- URL customization requires external hosting (Gradio share is random).
- No auth/roles; add if exposing publicly.
- Add analytics, caching, multilingual support, itinerary generation, and Dockerization for production.

## Contact / Ownership
- Project: ShivYatra AI Tourism Assistant
- Domain: Indian travel recommendations via RAG + local LLM
- Intended audience: Travel tech demos, PoCs, local-first assistants.
