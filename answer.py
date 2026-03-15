from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

MODEL = "gpt-4o"  

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
        # Retrieve relevant chunks from the vector database
        relevant_chunks = retriever.invoke(question)
        context_parts = []
        seen_sources = set()
        clean_sources = []

        for chunk in relevant_chunks:
            source = chunk.metadata.get("source", "unknown source")
            page = chunk.metadata.get("page", "unknown page")
            source_key = (source, page)

            context_parts.append(
                f"Source: {source}, Page: {page}\n{chunk.page_content}"
            )

            # Add to clean sources (deduplicated)
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                clean_sources.append({
                    "source": source,
                    "page": page,
                    "content_preview": chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content
                })
        
        context = "\n\n".join(context_parts)
        
        # Create system prompt with context
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        
        messages = [SystemMessage(content=system_prompt)]

        # Add previous conversation turns
        # Gradio history format: [[user_msg, bot_msg], [user_msg, bot_msg], ...]
        if history:
            for exchange in history:
                if len(exchange) >= 2:  # Safety check
                    user_msg, assistant_msg = exchange[0], exchange[1]
                    if user_msg:  # Only add if not None
                        messages.append(HumanMessage(content=user_msg))
                    if assistant_msg:  # Only add if not None
                        messages.append(AIMessage(content=assistant_msg))

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
        # Return error in expected format
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error in answer_question: {error_detail}")
        
        return {
            "answer": f"❌ Error: {str(e)}\n\nPlease try again or check your connection.",
            "sources": [],
            "source_count": 0
        }



