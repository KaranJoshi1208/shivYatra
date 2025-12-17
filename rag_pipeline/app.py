#!/usr/bin/env python3
"""
ShivYatra Tourism Chatbot - Main Application Launcher
Launch the complete RAG-powered tourism assistant with Gradio UI
"""

import sys
import os
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "ui"))
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "config"))

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'gradio',
        'chromadb',
        'sentence_transformers',
        'requests'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main application launcher"""
    print("ShivYatra Tourism Chatbot Launcher")
    print("=" * 50)
    
    print("Checking dependencies...")
    if not check_dependencies():
        return 1
    print("All dependencies available")
    
    print("Checking Ollama service...")
    if not check_ollama_service():
        print("Ollama service not running!")
        print("Start Ollama with: ollama serve")
        print("Make sure qwen2.5:1.5b is installed: ollama pull qwen2.5:1.5b")
        return 1
    print("Ollama service running")
    
    vector_db_path = PROJECT_ROOT.parent / "vector_db" / "chromadb"
    if not vector_db_path.exists():
        print("Vector database not found!")
        print(f"Expected at: {vector_db_path}")
        print("Run vector database initialization first")
        return 1
    print("Vector database found")
    
    print("Starting ShivYatra Tourism Chatbot...")
    print("UI will be available at: http://localhost:7860")
    print("Press Ctrl+C to stop")
    
    try:
        from chatbot_ui import main as launch_ui
        launch_ui()
    except KeyboardInterrupt:
        print("\nShutting down ShivYatra Chatbot...")
    except Exception as e:
        print(f"Error launching chatbot: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())