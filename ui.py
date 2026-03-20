import gradio as gr
from answer import answer_question

def format_response(result: dict) -> str:
    """
    Format the answer with sources for Gradio display.
    """
    answer = result["answer"]
    sources = result["sources"]

    # No sources for greetings/small talk
    if not sources:
        return answer
    
    # Build formatted response with clean citations
    response = f"{answer}\n\n"
    response += "---\n\n"
    response += "**Sources:**\n\n"
    
    for source in sources:
        filename = source['source'].replace(".pdf", "").replace("-", " ").title()
        response += f"- 📄 {filename} — Page {source['page']}\n"
    
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
                role = msg.get("role")
                # content can be str or list in Gradio 6 multimodal
                content = msg.get("content", "")
                content_str = content if isinstance(content, str) else str(content)

                if role == "user":
                    user_msg = content_str
                elif role == "assistant" and user_msg is not None:
                    converted_history.append((user_msg, content_str))
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