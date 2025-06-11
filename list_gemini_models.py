import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file (if you use one)
load_dotenv()

# Get your Google API Key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")

# Configure the genai library with your API key
genai.configure(api_key=GOOGLE_API_KEY)

print("Listing available Google Generative AI models for your API key...")

try:
    for m in genai.list_models():
        # Only show models that support text generation (often called 'generateContent')
        # and are not just for embeddings (which you already have working)
        if "generateContent" in m.supported_generation_methods:
            print(f"Name: {m.name}")
            print(f"  Display Name: {m.display_name}")
            print(f"  Description: {m.description}")
            print(f"  Supported Generation Methods: {m.supported_generation_methods}")
            print("-" * 30)

except Exception as e:
    print(f"An error occurred while listing models: {e}")
    print("Please ensure your GOOGLE_API_KEY is correct and you have access to the Generative Language API.")

print("\n--- Model Listing Complete ---")
print("Copy the 'Name' (e.g., 'models/gemini-1.0-pro') of a suitable model and use it in your query_knowledge_base.py script.")