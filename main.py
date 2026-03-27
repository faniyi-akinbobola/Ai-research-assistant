import os
from dotenv import load_dotenv
from ui import launch_ui


def main():
    """
    Main entry point for the AI Research Assistant application.
    """
    print("="*70)
    print("🔬 AI Research Assistant - Starting...")
    print("="*70)
    
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        print("Please set your OpenAI API key in the .env file")
        return
    
    print("✅ Environment variables loaded")
    
    # Check if vector database exists
    vector_db_path = "./data/vector_db"
    if not os.path.exists(vector_db_path) or not os.listdir(vector_db_path):
        print("\n⚠️  Vector database not found!")
        print("Please run the ingestion process first:")
        print("   uv run python ingest.py")
        return
    
    print("✅ Vector database found")
    
    # Launch the Gradio UI
    print("\n🚀 Launching Gradio interface...")
    launch_ui()

if __name__ == "__main__":
    main()