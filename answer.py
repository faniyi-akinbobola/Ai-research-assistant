from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

MODEL = "gpt-5-mini"  

# Load the EXISTING vector database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", max_retries=3, request_timeout=30)
vectorstore = Chroma(
    persist_directory="./data/vector_db",
    embedding_function=embeddings
)

# Create retriever to search the vector database
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize the LLM
llm = ChatOpenAI(model=MODEL, temperature=0.2, max_retries=3, request_timeout=30) 

SYSTEM_PROMPT_TEMPLATE = """
You are an expert AI Research Assistant specialized in analyzing and explaining academic papers in machine learning, deep learning, and artificial intelligence.

## Your Role:
- Provide clear, accurate explanations of research papers and their concepts
- Break down complex technical ideas into understandable explanations
- Reference specific sections, equations, or findings from the papers
- Maintain academic rigor while being accessible

## Guidelines:
1. **Base answers strictly on the provided context** from the research paper
2. If information is not in the context, clearly state: "This information is not covered in the provided paper"
3. When explaining technical concepts, provide both high-level intuition and technical details
4. Reference page numbers or sections when citing specific information
5. Use clear structure with headers, lists, and examples when helpful
6. If a question is ambiguous, ask for clarification

## Context from Research Paper:
{context}

Answer the user's question based on this context. Be precise, thorough, and cite specific parts of the paper when relevant.
"""

CONVERSATIONAL_PROMPT = """
You are a friendly AI Research Assistant. Respond naturally and concisely to greetings and small talk.
Keep responses short and warm. Do not reference any research papers unless specifically asked.
"""

# --- Query Classifier ---
GREETINGS = {
    "hi", "hello", "hey", "howdy", "sup", "yo", "hiya", "greetings",
    "hey there", "hi there", "hello there", "good morning", "good afternoon",
    "good evening", "good night", "morning", "whats up", "what's up",
    "wassup", "heya", "howdy"
}
SMALL_TALK = {
    "how are you", "how are you doing", "how is it going", "how do you do",
    "what are you", "who are you", "what can you do", "what do you do",
    "thanks", "thank you", "thank you so much", "cheers",
    "bye", "goodbye", "see you", "take care",
    "ok", "okay", "cool", "nice", "great", "awesome"
}

def classify_query(question: str) -> str:
    """
    Classify the query into: 'greeting', 'small_talk', or 'knowledge_query'
    """
    # Strip all punctuation and whitespace, not just trailing
    normalized = re.sub(r"[^\w\s]", "", question.strip().lower()).strip()

    # Exact match first
    if normalized in GREETINGS:
        return "greeting"

    if normalized in SMALL_TALK:
        return "small_talk"

    # Partial match: short queries that start with a greeting word
    words = normalized.split()
    if len(words) <= 4 and words[0] in {"hi", "hey", "hello", "good", "morning", "howdy", "yo"}:
        return "greeting"

    # Partial match for small talk phrases
    if any(phrase in normalized for phrase in SMALL_TALK):
        return "small_talk"

    return "knowledge_query"

#fetch the content from the vector database
def answer_question(question: str, history: list = []) -> dict:
    """
    Answer questions based on the research paper context.
    
    Args:
        question: User's question
        history: Conversation history from Gradio (list of [user_msg, bot_msg] pairs)
    
    Returns:
        dict with answer, sources, and source_count
    """
    try:
        # --- Query Classifier Layer ---
        query_type = classify_query(question)

        # --- Handle non-knowledge queries without RAG ---
        if query_type in ("greeting", "small_talk"):
            messages = [
                SystemMessage(content=CONVERSATIONAL_PROMPT),
                HumanMessage(content=question)
            ]
            response = llm.invoke(messages)
            return {
                "answer": response.content,
                "sources": [],
                "source_count": 0
            }

        # --- RAG Trigger Layer: only for knowledge queries ---
        relevant_chunks = retriever.invoke(question)
        context_parts = []
        seen_sources = set()
        clean_sources = []

        for chunk in relevant_chunks:
            # Get clean filename instead of full path
            raw_source = chunk.metadata.get("source", "unknown source")
            filename = os.path.basename(raw_source)
            page = chunk.metadata.get("page", "unknown page")
            # Pages are 0-indexed in PyPDF, make it human-readable
            readable_page = page + 1 if isinstance(page, int) else page
            source_key = (filename, readable_page)

            context_parts.append(
                f"Source: {filename}, Page: {readable_page}\n{chunk.page_content}"
            )

            # Add to clean sources (deduplicated, max 2)
            if source_key not in seen_sources and len(clean_sources) < 2:
                seen_sources.add(source_key)
                clean_sources.append({
                    "source": filename,
                    "page": readable_page,
                    "content_preview": chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
                })
        
        context = "\n\n".join(context_parts)
        
        # Create system prompt with context
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        
        messages = [SystemMessage(content=system_prompt)]

        # Add previous conversation turns
        if history:
            for exchange in history:
                if len(exchange) >= 2:
                    user_msg, assistant_msg = exchange[0], exchange[1]
                    if user_msg:
                        # user_msg can be str or list in Gradio 6
                        user_text = user_msg if isinstance(user_msg, str) else str(user_msg)
                        messages.append(HumanMessage(content=user_text))
                    if assistant_msg:
                        # assistant_msg can be str or list in Gradio 6
                        assistant_text = assistant_msg if isinstance(assistant_msg, str) else str(assistant_msg)
                        # Strip the sources block to avoid LLM copying it into answers
                        clean_assistant_msg = assistant_text.split("\n\n---\n\n**Sources:**")[0].strip()
                        messages.append(AIMessage(content=clean_assistant_msg))

        # Add current question
        messages.append(HumanMessage(content=question))
        
        # Get answer from the LLM
        response = llm.invoke(messages)
        
        return {
            "answer": response.content,
            "sources": clean_sources,
            "source_count": len(clean_sources)
        }
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error in answer_question: {error_detail}")
        
        return {
            "answer": f"Error: {str(e)}\n\nPlease try again or check your connection.",
            "sources": [],
            "source_count": 0
        }



