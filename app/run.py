#!/usr/bin/env python3
"""
Yatri Travel Assistant - Main Entry Point
"""

import sys
from pathlib import Path

# Add app to path
APP_ROOT = Path(__file__).parent
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "api"))
sys.path.insert(0, str(APP_ROOT / "core"))
sys.path.insert(0, str(APP_ROOT / "config"))


def check_dependencies():
    """Check if required packages are installed"""
    required = ["flask", "flask_cors", "chromadb", "sentence_transformers", "requests"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("All dependencies available")
    return True


def check_ollama():
    """Check if Ollama service is running"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("Ollama service running")
            return True
    except:
        pass
    print("WARNING: Ollama service not detected")
    return False


def check_database():
    """Check if vector database exists"""
    db_path = APP_ROOT.parent / "database"
    if db_path.exists() and any(db_path.iterdir()):
        print("Vector database found")
        return True
    print("WARNING: Vector database not found")
    return False


def main():
    """Main entry point"""
    print("\n" + "=" * 50)
    print("Yatri Travel Assistant")
    print("=" * 50)
    
    print("\nChecking dependencies...")
    if not check_dependencies():
        return
    
    print("\nChecking services...")
    check_ollama()
    check_database()
    
    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop\n")
    
    from server import app, init_rag
    init_rag()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
