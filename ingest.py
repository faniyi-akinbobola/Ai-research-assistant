import os
import shutil
from langchain_community.document_loaders import PyPDFLoader  # Fixed import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in the environment variables. Please set it before running the script.")

# Clear existing vector database
vector_db_path = "./data/vector_db"
if os.path.exists(vector_db_path):
    print(f"🗑️  Clearing existing vector database...")
    shutil.rmtree(vector_db_path)
    os.makedirs(vector_db_path)

#load in the pdf file
PDF_PATH = "data/raw/attention-is-all-you-need.pdf"  
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} pages")


#convert the pdf file into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")


#vectorize the chunks
print("Creating embeddings...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    max_retries=3
    )  # Specify the model for embeddings

#store the vectorized chunks in a vector database
print("Creating vector store (this may take a few minutes)...")
try:
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings, 
        persist_directory="./data/vector_db")
    
    print(f"Successfully processed {len(chunks)} chunks")
    print(f"Vector store created with {vectorstore._collection.count()} documents")
except Exception as e:
    print(f"Error creating vector store: {e}")
    print("Please check your internet connection and API key")
