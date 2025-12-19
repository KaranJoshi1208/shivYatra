"""
Yatri Travel Assistant - Flask Web Server
"""

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from rag_engine import create_rag_pipeline

# Template folder is in ../web/templates relative to this file
template_dir = Path(__file__).parent.parent / "web" / "templates"
static_dir = Path(__file__).parent.parent / "web" / "static"

app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
CORS(app)

# Initialize RAG pipeline
rag_pipeline = None


def init_rag():
    global rag_pipeline
    print("Initializing Yatri RAG pipeline...")
    rag_pipeline = create_rag_pipeline()
    if rag_pipeline:
        print("RAG pipeline ready")
    else:
        print("Failed to initialize RAG pipeline")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    global rag_pipeline
    if not rag_pipeline:
        return jsonify({"error": "Service unavailable", "response": ""}), 503

    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message", "response": ""}), 400

    try:
        result = rag_pipeline.chat(message)
        response_text = result.get("response", "")
        return jsonify({"response": response_text, "error": None})
    except Exception as e:
        return jsonify({"error": str(e), "response": ""}), 500


@app.route("/api/health")
def health():
    global rag_pipeline
    if not rag_pipeline:
        return jsonify({"status": "error", "message": "RAG not initialized"})

    health = rag_pipeline.get_health_status()
    return jsonify({
        "status": "ok" if health.get("initialized") else "error",
        "vector_store": health.get("vector_store", False),
        "embedding_model": health.get("embedding_model", False),
        "ollama": health.get("ollama", False),
        "total_embeddings": health.get("total_embeddings", 0),
    })


def main():
    init_rag()
    print("\n" + "=" * 50)
    print("Yatri Travel Assistant")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
