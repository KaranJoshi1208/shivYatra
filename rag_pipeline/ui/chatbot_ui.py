"""
ShivYatra Tourism Chatbot - Gradio UI
Beautiful and interactive chatbot interface for tourism assistance
"""

import gradio as gr
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "config"))

from rag_engine import create_rag_pipeline, ShivYatraRAG
from rag_config import UI_CONFIG, RAG_SETTINGS

class ShivYatraChatbotUI:
    """
    Gradio-based chatbot UI for ShivYatra tourism assistant
    """
    
    def __init__(self):
        """Initialize chatbot UI"""
        self.rag_pipeline = None
        self.chat_history = []
        self.is_ready = False
        
    def initialize(self):
        """Initialize RAG pipeline"""
        print("Initializing ShivYatra Tourism Chatbot...")
        self.rag_pipeline = create_rag_pipeline()
        
        if self.rag_pipeline:
            self.is_ready = True
            print("Chatbot initialized successfully!")
        else:
            print("Failed to initialize chatbot")
        
        return self.is_ready
    
    def chat_fn(self, message: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Main chat function for Gradio interface"""
        if not self.is_ready or not self.rag_pipeline:
            error_response = "**System Error**: Chatbot not properly initialized. Please restart the application."
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})
            return history
        
        if not message.strip():
            return history
        
        history.append({"role": "user", "content": message})
        
        try:
            result = self.rag_pipeline.chat(message)
            response = self._format_response(result)
            history.append({"role": "assistant", "content": response})
            return history
            
        except Exception as e:
            error_response = f"**Error**: {str(e)}"
            history.append({"role": "assistant", "content": error_response})
            return history
    
    def _format_response(self, result: Dict[str, Any]) -> str:
        """Format RAG response with clean output"""
        response = result.get("response", "")
        
        # Return clean response without extra formatting
        return response.strip()
    
    def get_example_questions(self) -> List[List[str]]:
        """Get example questions for the interface"""
        return [
            ["What are the best adventure activities in Manali?"],
            ["I'm planning a family trip to Himachal Pradesh. Any suggestions?"],
            ["What are some budget-friendly places to visit in Uttarakhand?"],
            ["Tell me about temples and spiritual places in Haridwar"],
            ["What adventure sports can I do in Kullu?"],
            ["I'm a solo traveler interested in trekking. Where should I go?"],
            ["What are the best places to visit in Ladakh?"],
            ["Tell me about budget accommodation options in Manali"],
            ["What's the best time to visit Kashmir?"],
            ["I want to experience local culture in Himachal. Any recommendations?"]
        ]
    
    def clear_chat(self) -> List[Dict[str, str]]:
        """Clear chat history"""
        self.chat_history = []
        return []
    
    def get_system_status(self) -> str:
        """Get formatted system status"""
        if not self.rag_pipeline:
            return "**System Status**: Platform not initialized"
        
        health = self.rag_pipeline.get_health_status()
        
        status_parts = []
        status_parts.append("**Platform Status:**")
        status_parts.append(f"• Vector Database: {'ONLINE' if health['vector_store'] else 'OFFLINE'} ({health['total_embeddings']:,} entries)")
        status_parts.append(f"• Embedding Engine: {'ACTIVE' if health['embedding_model'] else 'INACTIVE'}")
        status_parts.append(f"• AI Language Model: {'CONNECTED' if health['ollama'] else 'DISCONNECTED'}")
        status_parts.append(f"• Overall Status: {'OPERATIONAL' if health['initialized'] else 'ERROR'}")
        
        return "\\n".join(status_parts)
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        # Custom CSS inspired by the provided reference UI
        custom_css = """
        :root {
            --bg: #f5f6fb;
            --panel: #ffffff;
            --border: #e5e7eb;
            --text: #111827;
            --muted: #6b7280;
            --accent: #22c55e;
            --accent-soft: #d1fae5;
            --card-shadow: 0 10px 30px rgba(0,0,0,0.06);
            --radius: 12px;
        }
        body, .gradio-container {
            background: var(--bg);
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
            color: var(--text);
        }
        .topbar {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px 16px;
            box-shadow: var(--card-shadow);
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 14px;
        }
        .topbar .logo {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 700;
            color: var(--text);
        }
        .layout {
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 16px;
        }
        .sidebar {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 16px;
            box-shadow: var(--card-shadow);
        }
        .sidebar h3 {
            margin: 0 0 12px 0;
            font-size: 1rem;
            color: var(--text);
        }
        .nav-list { list-style: none; padding: 0; margin: 0 0 18px 0; }
        .nav-item {
            display: flex; align-items: center; gap: 10px;
            padding: 10px 12px; border-radius: 10px;
            color: var(--text); cursor: default; border: 1px solid transparent;
        }
        .nav-item.active { background: #eef2ff; border-color: #e0e7ff; font-weight: 600; }
        .recent-card {
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 10px;
            background: #f9fafb;
        }
        .main-panel {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .hero {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 28px;
            box-shadow: var(--card-shadow);
            text-align: center;
        }
        .hero-title { font-size: 2rem; margin: 10px 0 6px 0; color: var(--text); }
        .hero-sub { color: var(--muted); margin: 0; }
        .cards {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }
        .feature-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            box-shadow: var(--card-shadow);
            text-align: left;
        }
        .chat-box {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px;
            box-shadow: var(--card-shadow);
        }
        .input-row { display: flex; gap: 10px; align-items: center; }
        .input-row .controls { display: flex; gap: 6px; }
        .input-row .controls button { border-radius: 10px; }
        .small-btn { padding: 8px 12px; border: 1px solid var(--border); background: #f9fafb; color: var(--muted); }
        .green-outline { border: 1px solid var(--accent); box-shadow: 0 0 0 2px var(--accent-soft); }
        .gradio-container .prose p { margin: 0; }
        """
        
        with gr.Blocks(title=UI_CONFIG["title"], css=custom_css, theme=UI_CONFIG["theme"]) as interface:
            # Top bar
            gr.HTML(
                """
                <div class="topbar">
                    <div class="logo">🌄 ShivYatra.AI</div>
                    <div style="flex:1"></div>
                    <div style="display:flex; gap:8px; align-items:center; color:#6b7280;">
                        <span>ShivYatra RAG v1.0</span>
                        <span style="width:1px;height:18px;background:#e5e7eb;"></span>
                        <span>Secure Session</span>
                    </div>
                </div>
                """
            )
            
            with gr.Row(elem_classes="layout"):
                # Sidebar
                with gr.Column(elem_classes="sidebar"):
                    gr.HTML(
                        """
                        <h3>AI Modules</h3>
                        <ul class="nav-list">
                            <li class="nav-item active">AI Chat</li>
                            <li class="nav-item">AI Video</li>
                            <li class="nav-item">AI Image</li>
                            <li class="nav-item">Documents</li>
                            <li class="nav-item">Community</li>
                            <li class="nav-item">History</li>
                        </ul>
                        <h3>Recent Chat</h3>
                        <div class="recent-card">Trip ideas for Spiti Valley</div>
                        <div class="recent-card">Family plan for Himachal</div>
                        <div class="recent-card">Solo trek in Uttarakhand</div>
                        <button class="small-btn" style="width:100%;">Show More</button>
                        <div style="margin-top:14px; border-top:1px solid var(--border); padding-top:12px; color:var(--muted);">
                            Settings · Help
                        </div>
                    """
                    )
                
                # Main content
                with gr.Column(elem_classes="main-panel"):
                    gr.HTML(
                        """
                        <div class="hero">
                            <div style="display:flex; justify-content:center;">
                                <div style="width:70px;height:70px;border-radius:50%; background: radial-gradient(circle at 30% 30%, #7ef3a2, #16a34a); box-shadow: 0 10px 30px rgba(34,197,94,0.35);"></div>
                            </div>
                            <h1 class="hero-title">Welcome to ShivYatra.AI</h1>
                            <p class="hero-sub">Share your travel goal and let the assistant handle the research, planning, and recommendations.</p>
                        </div>
                        """
                    )
                    
                    gr.HTML(
                        """
                        <div class="cards">
                            <div class="feature-card">
                                <div style="font-weight:700; margin-bottom:6px;">Productivity Boost</div>
                                <p style="color:var(--muted); margin:0;">Get quick, curated destination ideas with top picks and must-dos.</p>
                            </div>
                            <div class="feature-card">
                                <div style="font-weight:700; margin-bottom:6px;">User-Friendly Onboarding</div>
                                <p style="color:var(--muted); margin:0;">Share who you are and we tailor the itinerary for family, solo, or adventure.</p>
                            </div>
                            <div class="feature-card">
                                <div style="font-weight:700; margin-bottom:6px;">Voice-Activated Responses</div>
                                <p style="color:var(--muted); margin:0;">Hands-free travel planning with concise, reliable answers.</p>
                            </div>
                        </div>
                        """
                    )
                    
                    # Chat area
                    with gr.Column(elem_classes="chat-box"):
                        chatbot = gr.Chatbot(
                            label=None,
                            placeholder="Ask me anything about Indian travel...",
                            height=280,
                            show_label=False,
                            container=False
                        )
                        with gr.Row(elem_classes="input-row"):
                            msg_input = gr.Textbox(
                                placeholder="Ask me anything...",
                                lines=1,
                                scale=4,
                                elem_classes=["green-outline"]
                            )
                            send_btn = gr.Button("Send", variant="primary")
                        with gr.Row(elem_classes="controls"):
                            attach_btn = gr.Button("Attach", elem_classes=["small-btn"])
                            voice_btn = gr.Button("Voice Message", elem_classes=["small-btn"])
                            prompt_btn = gr.Button("Browse Prompts", elem_classes=["small-btn"])
                            clear_btn = gr.Button("Clear", elem_classes=["small-btn"])
                        status_display = gr.Markdown(self.get_system_status(), visible=False)

            # Event handlers
            def submit_message(message, history):
                if not message.strip():
                    return "", history
                updated_history = self.chat_fn(message, history)
                return "", updated_history

            send_btn.click(submit_message, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            msg_input.submit(submit_message, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            clear_btn.click(lambda: (self.clear_chat(), ""), outputs=[chatbot, msg_input])
            attach_btn.click(lambda x: (x, chatbot.value if hasattr(chatbot, "value") else []), inputs=msg_input, outputs=[msg_input, chatbot])
            voice_btn.click(lambda x: (x, chatbot.value if hasattr(chatbot, "value") else []), inputs=msg_input, outputs=[msg_input, chatbot])
            prompt_btn.click(lambda: ("Suggest a 3-day Himachal itinerary", chatbot.value if hasattr(chatbot, "value") else []), outputs=[msg_input, chatbot])

            def load_welcome():
                return [{
                    "role": "assistant",
                    "content": "Welcome to ShivYatra.AI — tell me your travel goal and I'll craft a plan."
                }]

            interface.load(load_welcome, outputs=chatbot)
        
        return interface
    
    def launch(self):
        """Launch the Gradio interface"""
        if not self.initialize():
            print("Platform initialization failed - unable to launch interface")
            return None
        
        print("Launching ShivYatra Professional Tourism Platform...")
        
        interface = self.create_interface()
        
        interface.launch(
            server_name=UI_CONFIG["server_name"],
            server_port=UI_CONFIG["server_port"],
            share=UI_CONFIG["share"],
            debug=UI_CONFIG["debug"],
            show_error=True,
            quiet=False,
            theme=UI_CONFIG["theme"]
        )


def main():
    """Main function to launch the platform"""
    print("Initializing ShivYatra Professional Tourism Platform...")
    
    chatbot_ui = ShivYatraChatbotUI()
    chatbot_ui.launch()


if __name__ == "__main__":
    main()