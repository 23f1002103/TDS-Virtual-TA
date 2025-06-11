import os
import base64
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Remove ChatGoogleGenerativeAI from here
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from pinecone import Pinecone

load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
GOOGLE_LLM_MODEL = "models/gemini-1.5-flash-002"

# Ensure API key and Pinecone keys are set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")
if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT or not PINECONE_INDEX_NAME:
    raise ValueError("Pinecone environment variables not set. Please set PINECONE_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_INDEX_NAME.")


CUSTOM_QA_PROMPT = PromptTemplate(
    template="""You are a helpful assistant for university students. Your task is to answer questions based *only* on the provided context and any image provided.
If the context or image does not contain enough information to answer the question, or if you cannot find the answer, please politely state "I am sorry, but I do not have enough information in my knowledge base or the provided image to answer that question." Do not try to make up an answer.

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"],
)

# _qa_chain = None # This will now be initialized by the returned retriever and LLM from startup_event
# _llm = None      # LLM moved to api.py startup
_retriever = None # Keep this global for now, or pass it around

def initialize_rag_retriever_only(): # Renamed the function
    global _retriever

    if _retriever is not None: # Check if retriever is already initialized
        return _retriever

    print(f"⏳ Initializing Google embedding model: {GOOGLE_EMBEDDING_MODEL}...")
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
        print("✅ Google Embedding model loaded.")
    except Exception as e:
        raise RuntimeError(f"Error initializing Google Embedding model: {e}. Check GOOGLE_API_KEY and internet connection.")

    print(f"⏳ Connecting to Pinecone index: {PINECONE_INDEX_NAME}...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings_model
        )
        
        print("✅ Connected to Pinecone.")
    except Exception as e:
        raise RuntimeError(f"Error connecting to Pinecone: {e}. Ensure API keys and index name are correct, and the index exists.")

    _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return _retriever # Only return the retriever now


def query_rag_system(question: str, image_data_base64: str = None, qa_chain_instance=None): # Added qa_chain_instance
    """
    Queries the RAG system with a given question and optional image.
    Requires an initialized qa_chain_instance to be passed.
    """
    if qa_chain_instance is None:
        raise RuntimeError("RAG chain (qa_chain_instance) not provided to query_rag_system.")

    if image_data_base64:
        print("Image attachment detected in query_rag_system. Note: Current RAG chain primarily handles text retrieval.")
        pass

    print(f"Querying RAG system with question: {question}")

    response = qa_chain_instance.invoke({"query": question}) # Use the passed instance

    answer = response.get("result")
    source_documents = response.get("source_documents", [])

    return {
        "answer": answer,
        "source_documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in source_documents]
    }