import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key exists: {api_key is not None}")
print(f"API Key starts with: {api_key[:10] if api_key else 'None'}...")

try:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="test"
    )
    print("✅ API connection successful!")
    print(f"Embedding dimension: {len(response.data[0].embedding)}")
except Exception as e:
    print(f"❌ API connection failed: {e}")