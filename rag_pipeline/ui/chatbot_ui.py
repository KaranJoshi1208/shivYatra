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
        print("🚀 Initializing ShivYatra Tourism Chatbot...")
        self.rag_pipeline = create_rag_pipeline()
        
        if self.rag_pipeline:
            self.is_ready = True
            print("✅ Chatbot initialized successfully!")
        else:
            print("❌ Failed to initialize chatbot")
        
        return self.is_ready
    
    def chat_fn(self, message: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Main chat function for Gradio interface
        
        Args:
            message: User input message
            history: Chat history as list of message dictionaries
        
        Returns:
            Updated chat history
        """
        if not self.is_ready or not self.rag_pipeline:
            error_response = "🚨 **System Error**: Chatbot not properly initialized. Please restart the application."
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})
            return history
        
        if not message.strip():
            return history
        
        # Add user message to history
        history.append({"role": "user", "content": message})
        
        try:
            # Get RAG response
            result = self.rag_pipeline.chat(message)
            
            # Format response with metadata
            response = self._format_response(result)
            
            # Add assistant response to history
            history.append({"role": "assistant", "content": response})
            
            return history
            
        except Exception as e:
            error_response = f"❌ **Error**: {str(e)}"
            history.append({"role": "assistant", "content": error_response})
            return history
    
    def _format_response(self, result: Dict[str, Any]) -> str:
        """Format RAG response with metadata and sources"""
        response = result.get("response", "")
        context_docs = result.get("context_docs", [])
        processing_time = result.get("processing_time", 0)
        
        # Main response
        formatted_response = f"{response}"
        
        # Add sources if available
        if context_docs and len(context_docs) > 0:
            formatted_response += "\\n\\n---\\n**📚 Sources:**\\n"
            
            for i, doc in enumerate(context_docs[:3], 1):  # Show top 3 sources
                metadata = doc['metadata']
                similarity = doc['similarity']
                
                source_info = f"{i}. **{metadata['city']}, {metadata['state']}** ({metadata['category']}) - Relevance: {similarity:.2f}"
                formatted_response += f"\\n{source_info}"
        
        # Add processing time
        formatted_response += f"\\n\\n*⏱️ Response time: {processing_time}s*"
        
        return formatted_response
    
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
            return "❌ **System Status**: Not initialized"
        
        health = self.rag_pipeline.get_health_status()
        
        status_parts = []
        status_parts.append("**🔧 System Status:**")
        status_parts.append(f"• Vector Store: {'✅' if health['vector_store'] else '❌'} ({health['total_embeddings']:,} embeddings)")
        status_parts.append(f"• Embedding Model: {'✅' if health['embedding_model'] else '❌'}")
        status_parts.append(f"• Ollama LLM: {'✅' if health['ollama'] else '❌'}")
        status_parts.append(f"• Overall: {'🟢 Ready' if health['initialized'] else '🔴 Error'}")
        
        return "\\n".join(status_parts)
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .chat-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px 15px 0 0;
        }
        """
        
        with gr.Blocks(
            title=UI_CONFIG["title"]
        ) as interface:
            
            # Header
            gr.HTML(f"""
            <div class="header">
                <h1>🏔️ ShivYatra Tourism Assistant</h1>
                <p>Your AI-powered guide for exploring incredible destinations in India</p>
                <p><em>Powered by RAG + Ollama + ChromaDB</em></p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=4):
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        label="Chat with ShivYatra Assistant",
                        placeholder="Ask me anything about traveling in India! 🇮🇳",
                        height=500,
                        container=True,
                        show_label=False
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask about destinations, activities, budget tips, or anything related to Indian tourism...",
                            container=False,
                            scale=4,
                            lines=1
                        )
                        send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                        clear_btn = gr.Button("Clear 🗑️", variant="secondary", scale=1)
                    
                    # Example questions
                    with gr.Accordion("💡 Example Questions", open=False):
                        examples = gr.Examples(
                            examples=self.get_example_questions(),
                            inputs=[msg_input],
                            label="Click on any question to try it:"
                        )
                
                with gr.Column(scale=1):
                    # Status panel
                    with gr.Accordion("📊 System Status", open=True):
                        status_display = gr.Markdown(
                            self.get_system_status(),
                            label="System Status"
                        )
                        refresh_btn = gr.Button("Refresh Status 🔄", variant="secondary")
                    
                    # Information panel
                    with gr.Accordion("ℹ️ About", open=False):
                        gr.Markdown("""
                        **ShivYatra Tourism Assistant**
                        
                        🔹 **AI Model**: Qwen2.5:1.5B via Ollama
                        🔹 **Knowledge Base**: 4,160+ tourism chunks
                        🔹 **Regions Covered**: Himachal, Uttarakhand, J&K, Ladakh, and more
                        🔹 **Capabilities**: 
                           - Destination recommendations
                           - Activity suggestions
                           - Budget planning
                           - Cultural insights
                           - Practical travel tips
                        
                        **🚨 Note**: Make sure Ollama is running:
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
👋 **Welcome to ShivYatra - Your AI Tourism Assistant!**

I'm here to help you explore the incredible destinations across India! Here's what I can help you with:

🏔️ **Destinations**: Discover amazing places in Himachal Pradesh, Uttarakhand, Kashmir, Ladakh, and more
🎯 **Activities**: Adventure sports, cultural experiences, spiritual journeys, family fun
💰 **Budget Planning**: Find options that fit your budget - from budget-friendly to luxury
🎒 **Travel Types**: Solo travel, family trips, adventure tours, spiritual journeys
📍 **Local Insights**: Hidden gems, local culture, practical tips, and authentic experiences

**Just ask me anything!** For example:
- "What are the best places for adventure sports in Himachal?"
- "I have ₹20,000 budget for a family trip. Where should we go?"
- "Tell me about spiritual places in Uttarakhand"

Let's start planning your incredible Indian adventure! 🇮🇳✨
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
            print("❌ Cannot launch chatbot - initialization failed")
            return None
        
        print(f"🚀 Launching ShivYatra Chatbot UI...")
        
        interface = self.create_interface()
        
        # Launch with configuration
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
    """Main function to launch the chatbot"""
    print("🏔️ Starting ShivYatra Tourism Chatbot...")
    
    chatbot_ui = ShivYatraChatbotUI()
    chatbot_ui.launch()


if __name__ == "__main__":
    main()