#!/usr/bin/env python3
"""
Test script for ShivYatra Modern UI
"""

import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "config"))
sys.path.append(str(Path(__file__).parent / "ui" / "components"))

def test_imports():
    """Test all imports"""
    try:
        from rag_engine import create_rag_pipeline
        from rag_config import UI_CONFIG
        from ui.components import UIComponents
        from ui.chatbot_ui import ShivYatraChatbotUI
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_ui_creation():
    """Test UI creation without launching"""
    try:
        from ui.chatbot_ui import ShivYatraChatbotUI
        ui = ShivYatraChatbotUI()
        print("✅ UI class instantiated!")
        return True
    except Exception as e:
        print(f"❌ UI creation error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing ShivYatra Modern UI...")
    print("=" * 40)

    success = True
    success &= test_imports()
    success &= test_ui_creation()

    if success:
        print("\n🎉 All tests passed! Ready to launch.")
        print("Run: python ui/chatbot_ui.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")