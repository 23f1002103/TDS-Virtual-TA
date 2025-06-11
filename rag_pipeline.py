import os
import base64
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pinecone import Pinecone
import json # Import json for serialization utility

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

_retriever = None

def initialize_rag_retriever_only():
    global _retriever

    if _retriever is not None:
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
    
    return _retriever


def serialize_value(value):
    """
    Recursively converts problematic types (like protobuf Repeated) to serializable forms.
    This helps ensure metadata is JSON-compatible.
    """
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    
    # Heuristic for protobuf Repeated and other non-standard iterable types
    # This specifically targets objects that behave like lists but aren't standard Python lists
    if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
        try:
            return [serialize_value(item) for item in list(value)]
        except TypeError: # If it's iterable but cannot be converted to list (e.g., generator)
            return str(value)
    
    # More direct check for protobuf objects if 'DESCRIPTOR' attribute is present
    # and the type name suggests it's a Repeated field
    if hasattr(value, 'DESCRIPTOR') and 'Repeated' in str(type(value)):
        return [serialize_value(item) for item in list(value)]

    # Attempt to convert to string if it's a type that's generally not JSON serializable
    # but doesn't fit the above categories (e.g., custom objects, specific enums etc.)
    try:
        # Try a round-trip through JSON to confirm serializability for common types
        # This will fail if the value itself isn't directly serializable
        json.dumps(value)
        return value
    except (TypeError, OverflowError): # OverflowError for very large ints not fitting JSON spec
        return str(value) # Fallback to string representation

    # If it passes all checks, return as is
    return value


def query_rag_system(question: str, image_data_base64: str = None, qa_chain_instance=None):
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

    response = qa_chain_instance.invoke({"query": question})

    answer = response.get("result")
    source_documents = response.get("source_documents", [])

    # Process source_documents to ensure metadata is fully serializable
    cleaned_source_documents = []
    for doc in source_documents:
        cleaned_source_documents.append({
            "page_content": doc.page_content,
            "metadata": serialize_value(doc.metadata) # Apply the serialization function
        })

    return {
        "answer": answer,
        "source_documents": cleaned_source_documents
    }