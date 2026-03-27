from typing import Generator
import os
import gradio as gr
from answer import answer_question, NOT_IN_PAPER
from ingest import ingest_pdf

def hide_sources_for_response(answer: str) -> bool:
    """Returns True if sources should be hidden for this response."""
    return NOT_IN_PAPER.lower() in answer.lower()

def format_sources(sources: list) -> str:
    """
    Format sources block to append after the streamed answer.
    """
    if not sources:
        return ""

    sources_block = "\n\n---\n\n**Sources:**\n\n"
    for source in sources:
        filename = source['source'].replace(".pdf", "").replace("-", " ").title()
        sources_block += f"- {filename} — Page {source['page']}\n"

    return sources_block

def get_existing_papers() -> list[str]:
    """
    Read paper filenames already stored in the vector DB.
    Returns a deduplicated list of basenames.
    """
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", max_retries=3)
        vectorstore = Chroma(persist_directory="./data/vector_db", embedding_function=embeddings)
        existing = vectorstore.get()
        sources = existing.get("metadatas", [])
        seen = set()
        papers = []
        for m in sources:
            if m:
                name = os.path.basename(m.get("source", ""))
                if name and name not in seen:
                    seen.add(name)
                    papers.append(name)
        return papers
    except Exception:
        return []

def launch_ui():
    """
    Launch the Gradio Blocks interface with file upload and chat.
    """
    
    # Check DB for papers already loaded from previous sessions
    existing_papers = get_existing_papers()
    chat_ready = len(existing_papers) > 0

    with gr.Blocks(title="AI Research Assistant") as demo:

        # --- State: pre-populate from existing DB ---
        uploaded_papers = gr.State(existing_papers)

        # --- Header ---
        gr.Markdown("# AI Research Assistant")
        gr.Markdown("Upload one or more research papers (PDF), then ask questions about them.")

        # --- Upload Section ---
        with gr.Row():
            file_upload = gr.File(
                label="Upload research paper(s)",
                file_types=[".pdf"],
                file_count="multiple",
            )

        upload_status = gr.Textbox(
            label="Upload status",
            interactive=False,
            value=f"{len(existing_papers)} paper(s) already loaded from previous session." if chat_ready else "",
            placeholder="No papers uploaded yet.",
        )

        papers_list = gr.Textbox(
            label="Loaded papers",
            interactive=False,
            value=", ".join(existing_papers) if existing_papers else "",
            placeholder="None",
        )

        # --- Chat Section ---
        chatbot = gr.Chatbot()

        with gr.Row():
            message_input = gr.Textbox(
                placeholder="Ask a question about the uploaded papers..." if chat_ready else "Upload a paper first, then ask your question...",
                label="Your question",
                scale=9,
                interactive=chat_ready,
            )
            send_btn = gr.Button("Send", scale=1, interactive=chat_ready)

        # --- Upload Handler ---
        def handle_upload(files, current_papers):
            if not files:
                return (
                    "No papers uploaded yet." if not current_papers else "\n".join([f"Loaded: {p}" for p in current_papers]),
                    ", ".join(current_papers) or "None",
                    current_papers,
                    gr.update(interactive=bool(current_papers)),
                    gr.update(interactive=bool(current_papers)),
                )

            status_lines = []
            for file in files:
                # Gradio may pass a string path or a file object
                file_path = file.name if hasattr(file, "name") else str(file)
                if not file_path or not os.path.exists(file_path):
                    continue

                result = ingest_pdf(file_path)
                filename = result.get("filename", os.path.basename(file_path))

                if result["success"]:
                    if filename not in current_papers:
                        current_papers.append(filename)
                    status_lines.append(
                        f"Ingested: {filename} ({result['pages']} pages, {result['chunks']} chunks)"
                    )
                else:
                    status_lines.append(f"Skipped {filename}: {result['error']}")

            status = "\n".join(status_lines) if status_lines else "No new papers to process."
            papers_display = ", ".join(current_papers) if current_papers else "None"
            chat_active = len(current_papers) > 0

            return (
                status,
                papers_display,
                current_papers,
                gr.update(interactive=chat_active, placeholder="Ask a question about the uploaded papers..." if chat_active else "Upload a paper first..."),
                gr.update(interactive=chat_active),
            )

        file_upload.upload(
            fn=handle_upload,
            inputs=[file_upload, uploaded_papers],
            outputs=[upload_status, papers_list, uploaded_papers, message_input, send_btn],
        )

        # --- Chat Handler ---
        def extract_text_content(content) -> str:
            """Safely extract plain text from Gradio 6 content, which may be a string or a list of content dicts."""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Gradio 6 multimodal format: [{"type": "text", "text": "..."}, ...]
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                return "".join(parts)
            return str(content)

        def handle_message(message, history, current_papers):
            if not message.strip():
                yield history, ""
                return

            # Convert Gradio messages history to (user, assistant) tuples for answer_question
            converted_history = []
            user_msg = None
            for msg in history:
                if isinstance(msg, dict):
                    role = msg.get("role")
                    content_str = extract_text_content(msg.get("content", ""))
                    if role == "user":
                        user_msg = content_str
                    elif role == "assistant" and user_msg is not None:
                        converted_history.append((user_msg, content_str))
                        user_msg = None

            # Append user message immediately
            history = history + [{"role": "user", "content": message}]
            yield history, ""

            # Stream assistant response
            assistant_text = ""
            for result in answer_question(message, converted_history):
                partial_answer = result["answer"]
                final_sources = result["sources"]
                is_done = result["done"]

                if is_done:
                    hide_sources = not final_sources or hide_sources_for_response(partial_answer)
                    assistant_text = partial_answer if hide_sources else partial_answer + format_sources(final_sources)
                else:
                    assistant_text = partial_answer

                yield history + [{"role": "assistant", "content": assistant_text}], ""

        send_btn.click(
            fn=handle_message,
            inputs=[message_input, chatbot, uploaded_papers],
            outputs=[chatbot, message_input],
        )

        message_input.submit(
            fn=handle_message,
            inputs=[message_input, chatbot, uploaded_papers],
            outputs=[chatbot, message_input],
        )

    demo.launch(inbrowser=True, share=False)
