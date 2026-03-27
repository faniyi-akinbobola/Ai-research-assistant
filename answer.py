from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from typing import Generator
import os
import re
import time
from metrics import log_query

# Load environment variables
load_dotenv()

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

MODEL = "gpt-4o-mini"

# Load the EXISTING vector database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", max_retries=3, request_timeout=30)

def get_retriever():
    """
    Load a fresh retriever each call so newly ingested papers are always visible.
    """
    vectorstore = Chroma(
        persist_directory="./data/vector_db",
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize the LLM
llm = ChatOpenAI(model=MODEL, temperature=0.2, max_retries=3, request_timeout=30, streaming=True)

# One exact phrase used everywhere — in the prompt AND in the checker
NOT_IN_PAPER = "This information is not covered in the uploaded papers."

SYSTEM_PROMPT_TEMPLATE = """
You are an expert AI Research Assistant. Your sole purpose is to answer questions based strictly on the research paper excerpts provided below.

## Rules:
1. Only use information from the provided context to answer questions.
2. If the answer cannot be found in the context, respond with exactly this phrase and nothing else:
   "This information is not covered in the uploaded papers."
3. Never suggest alternatives, offer general knowledge, or answer outside the provided context.
4. When answering, cite the specific page number from the context.
5. Be precise and thorough. Use headers and lists where helpful.

## Context from Uploaded Papers:
The following excerpts are from one or more uploaded research papers.
Each excerpt is labeled with its source filename and page number.

{context}

Answer the user's question using only the context above.
"""

CONVERSATIONAL_PROMPT = """
You are a friendly assistant. The user is greeting you or making small talk.
Respond warmly and briefly in one or two sentences.
Do not mention research papers, documents, or your capabilities unless directly asked.
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
def answer_question(question: str, history: list = []) -> Generator[dict, None, None]:
    """
    Stream answers based on the research paper context.

    Args:
        question: User's question
        history: Conversation history from Gradio (list of [user_msg, bot_msg] pairs)

    Yields:
        dict with partial answer, sources, source_count, and done flag
    """
    query_type = "knowledge_query"
    start_time = time.time()

    try:
        start_time = time.time()

        # --- Query Classifier Layer ---
        query_type = classify_query(question)

        # --- Handle non-knowledge queries without RAG ---
        if query_type in ("greeting", "small_talk"):
            messages = [
                SystemMessage(content=CONVERSATIONAL_PROMPT),
                HumanMessage(content=question)
            ]
            partial_answer = ""
            for chunk in llm.stream(messages):
                partial_answer += chunk.content
                yield {
                    "answer": partial_answer,
                    "sources": [],
                    "source_count": 0,
                    "done": False
                }
            yield {
                "answer": partial_answer,
                "sources": [],
                "source_count": 0,
                "done": True
            }
            log_query(
                question=question,
                query_type=query_type,
                response_time_ms=(time.time() - start_time) * 1000,
                sources_count=0,
                sources_shown=False
            )
            return

        # --- RAG Trigger Layer: only for knowledge queries ---
        retriever = get_retriever()
        relevant_chunks = retriever.invoke(question)
        context_parts = []
        seen_sources = set()
        clean_sources = []

        for chunk in relevant_chunks:
            raw_source = chunk.metadata.get("source", "unknown source")
            filename = os.path.basename(raw_source)
            page = chunk.metadata.get("page", "unknown page")
            readable_page = page + 1 if isinstance(page, int) else page
            source_key = (filename, readable_page)

            context_parts.append(
                f"Source: {filename}, Page: {readable_page}\n{chunk.page_content}"
            )

            if source_key not in seen_sources and len(clean_sources) < 2:
                seen_sources.add(source_key)
                clean_sources.append({
                    "source": filename,
                    "page": readable_page,
                    "content_preview": chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
                })

        context = "\n\n".join(context_parts)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        messages = [SystemMessage(content=system_prompt)]

        # Add previous conversation turns
        if history:
            for exchange in history:
                if len(exchange) >= 2:
                    user_msg, assistant_msg = exchange[0], exchange[1]
                    if user_msg:
                        user_text = user_msg if isinstance(user_msg, str) else str(user_msg)
                        messages.append(HumanMessage(content=user_text))
                    if assistant_msg:
                        assistant_text = assistant_msg if isinstance(assistant_msg, str) else str(assistant_msg)
                        clean_assistant_msg = assistant_text.split("\n\n---\n\n**Sources:**")[0].strip()
                        messages.append(AIMessage(content=clean_assistant_msg))

        messages.append(HumanMessage(content=question))

        # --- Response Generator Layer: stream the response ---
        partial_answer = ""
        for chunk in llm.stream(messages):
            partial_answer += chunk.content
            yield {
                "answer": partial_answer,
                "sources": clean_sources,
                "source_count": len(clean_sources),
                "done": False
            }

        # Final yield signals streaming is complete
        not_in_paper = NOT_IN_PAPER.lower() in partial_answer.lower()
        sources_shown = bool(clean_sources) and not not_in_paper

        yield {
            "answer": partial_answer,
            "sources": clean_sources,
            "source_count": len(clean_sources),
            "done": True
        }

        log_query(
            question=question,
            query_type=query_type,
            response_time_ms=(time.time() - start_time) * 1000,
            sources_count=len(clean_sources),
            sources_shown=sources_shown
        )

    except Exception as e:
        import traceback
        print(f"Error in answer_question: {traceback.format_exc()}")
        log_query(
            question=question,
            query_type=query_type,
            response_time_ms=(time.time() - start_time) * 1000,
            sources_count=0,
            sources_shown=False,
            error=True
        )
        yield {
            "answer": f"Error: {str(e)}\n\nPlease try again or check your connection.",
            "sources": [],
            "source_count": 0,
            "done": True
        }



