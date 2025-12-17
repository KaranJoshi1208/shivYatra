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
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chat-container {
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            border: 1px solid #e0e0e0;
        }
        .header {
            text-align: center;
            padding: 24px;
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: #ffffff;
            border-radius: 8px 8px 0 0;
            border-bottom: 2px solid #1a252f;
        }
        .status-panel {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 16px;
        }
        .professional-text {
            color: #2c3e50;
            line-height: 1.6;
        }
        """
        
        with gr.Blocks(
            title=UI_CONFIG["title"]
        ) as interface:
            
            # Header
            gr.HTML(f"""
            <div class="header">
                <h1 style="margin: 0; font-size: 2.2em; font-weight: 600; color: #ffffff;">ShivYatra Tourism Assistant</h1>
                <p style="margin: 8px 0 4px 0; font-size: 1.1em; color: #ecf0f1;">Professional AI-Powered Travel Consultation Platform</p>
                <p style="margin: 0; font-size: 0.95em; color: #bdc3c7;"><em>Advanced RAG Technology | Ollama LLM | Vector Database</em></p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=4):
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        label="Professional Travel Consultation",
                        placeholder="Welcome to ShivYatra Professional Tourism Consultation Platform",
                        height=500,
                        container=True,
                        show_label=False
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Enter your travel inquiry: destinations, activities, budget planning, or consultation requests...",
                            container=False,
                            scale=4,
                            lines=1
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                        clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                    
                    # Example questions
                    with gr.Accordion("Example Questions", open=False):
                        examples = gr.Examples(
                            examples=self.get_example_questions(),
                            inputs=[msg_input],
                            label="Click on any question to try it:"
                        )
                
                with gr.Column(scale=1):
                    # Status panel
                    with gr.Accordion("System Status", open=True):
                        status_display = gr.Markdown(
                            self.get_system_status(),
                            label="System Status"
                        )
                        refresh_btn = gr.Button("Refresh Status", variant="secondary")
                    
                    # Information panel
                    with gr.Accordion("About", open=False):
                        gr.Markdown("""
                        **ShivYatra Professional Tourism Platform**
                        
                        **Technical Specifications:**
                        • AI Model: Qwen2.5:1.5B via Ollama
                        • Knowledge Base: 4,160+ curated tourism entries
                        • Coverage: Himachal Pradesh, Uttarakhand, Jammu & Kashmir, Ladakh
                        
                        **Service Capabilities:**
                        • Destination Analysis & Recommendations
                        • Activity Planning & Consultation
                        • Budget Optimization Strategies
                        • Cultural Intelligence & Local Insights
                        • Professional Travel Advisory
                        
                        **System Requirements:**
                        Ensure Ollama service is active:
                        ```bash
                        ollama serve
                        ```
                        """)
            
            # Event handlers
            
            # Send message on button click or Enter key
            def submit_message(message, history):
                if not message.strip():
                    return "", history
                updated_history = self.chat_fn(message, history)
                return "", updated_history
            
            send_btn.click(
                submit_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            msg_input.submit(
                submit_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            # Clear chat
            def clear_and_reset():
                return self.clear_chat(), ""
            
            clear_btn.click(
                clear_and_reset,
                outputs=[chatbot, msg_input]
            )
            
            # Refresh status
            refresh_btn.click(
                self.get_system_status,
                outputs=status_display
            )
            
            # Welcome message
            def load_welcome():
                return [{
                    "role": "assistant",
                    "content": """
**Welcome to ShivYatra Professional Tourism Consultation Platform**

I am your dedicated AI travel consultant, specializing in comprehensive tourism solutions across India's premier destinations.
How can I assist you in planning your next unforgettable journey?
                    """
                }]
            
            interface.load(
                load_welcome,
                outputs=chatbot
            )
        
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