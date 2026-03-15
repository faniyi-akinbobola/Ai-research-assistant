import gradio as gr
from answer import answer_question

def format_response(result: dict) -> str:
    """
    Format the answer with sources for Gradio display.
    """
    answer = result["answer"]
    sources = result["sources"]
    
    # Build formatted response
    response = f"{answer}\n\n"
    response += "---\n\n"
    response += f"📚 **Sources** ({result['source_count']} references):\n\n"
    
    for i, source in enumerate(sources, 1):
        response += f"{i}. **Page {source['page']}**\n"
        response += f"   _{source['content_preview']}_\n\n"
    
    return response

def chat_wrapper(message: str, history: list) -> str:
    """
    Wrapper function for Gradio ChatInterface.
    Converts Gradio 6 history format to our format.
    """
    # Gradio 6 history format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    # Convert to list of (user_msg, assistant_msg) tuples
    converted_history = []
    if history:
        user_msg = None
        for msg in history:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                elif msg.get("role") == "assistant" and user_msg is not None:
                    converted_history.append((user_msg, msg.get("content", "")))
                    user_msg = None
    
    try:
        # Get answer with sources
        result = answer_question(message, converted_history)
        
        # Format for display
        return format_response(result)
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease check your internet connection and try again."

def launch_ui():
    """
    Launch the Gradio interface.
    """
    interface = gr.ChatInterface(
        fn=chat_wrapper,
        title="🔬 AI Research Assistant",
        description="Ask questions about the 'Attention Is All You Need' paper. I'll provide answers with sources!",
        examples=[
            "What is the Transformer architecture?",
            "Explain the self-attention mechanism",
            "What are the key results of this paper?",
            "How does multi-head attention work?"
        ],
        flagging_mode="never"
    )
    
    interface.launch(
        inbrowser=True,
        share=False,  # Set to True if you want a public link
    )