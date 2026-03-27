import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from pypdf import PdfReader

# Load environment variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in the environment variables. Please set it before running the script.")

def ingest_pdf(uploaded_pdf_path: str):
    #validate the uploaded file path
    if not os.path.exists(uploaded_pdf_path):
        print(f"Error: PDF file not found at path: {uploaded_pdf_path}")
        return {
            "success": False,
            "error": f"PDF file not found at path: {uploaded_pdf_path}"
        }

    #check if the paper already exists in the database
    filename = os.path.basename(uploaded_pdf_path)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", max_retries=3)
    vectorstore = Chroma(persist_directory="./data/vector_db", embedding_function=embeddings)
    existing = vectorstore.get()
    sources = existing.get("metadatas", [])
    already_exists = any(
        os.path.basename(m.get("source", "")) == filename
        for m in sources
        if m
    )
    if already_exists:
        print("Paper already exists in the database, skipping ingestion...")
        return {
            "success": False,
            "error": "Paper already exists in the database"
        }

    #load in the pdf file
    try:
        loader = PyPDFLoader(uploaded_pdf_path)
        documents = loader.load()
    except Exception as e:
        # Some PDFs have font encoding issues — fall back to raw pypdf extraction
        print(f"Standard extraction failed ({e}), using fallback extractor...")
        reader = PdfReader(uploaded_pdf_path)
        documents = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
            except Exception:
                text = ""
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"source": uploaded_pdf_path, "page": i}
                ))

    if not documents:
        return {
            "success": False,
            "error": "Could not extract any text from this PDF."
        }

    print(f"Loaded {len(documents)} pages")


    #convert the pdf file into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")


    #vectorize the chunks
    # Specify the model for embeddings
    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        max_retries=3
        ) 

    #store the vectorized chunks in a vector database
    print("Creating vector store (this may take a few minutes)...")
    try:
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings, 
            persist_directory="./data/vector_db")
        
        print(f"Successfully processed {len(chunks)} chunks")
        print(f"Vector store created with {vectorstore._collection.count()} documents")
        return{
            "success": True,
            "filename": os.path.basename(uploaded_pdf_path),
            "pages": len(documents),
            "chunks": len(chunks)
        }
    except Exception as e:
        print(f"Error creating vector store: {e}")
        print("Please check your internet connection and API key")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    ingest_pdf("data/raw/attention-is-all-you-need.pdf")