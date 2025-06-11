import os
import base64
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import Chroma # <--- REMOVE THIS
from langchain_pinecone import PineconeVectorStore # <--- ADD THIS (or for your chosen DB)
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# REMOVE VECTOR_DB_DIR = "chroma_db"

# New Pinecone (or other vector DB) specific configurations
# You'll get these from your Pinecone dashboard
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # e.g., "gcp-starter" or "us-west-2"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") # The name of your Pinecone index

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

_qa_chain = None
_llm = None

def initialize_rag_components():
    global _qa_chain, _llm

    if _qa_chain is not None:
        return _qa_chain, _llm

    print(f"⏳ Initializing Google embedding model: {GOOGLE_EMBEDDING_MODEL}...")
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
        print("✅ Google Embedding model loaded.")
    except Exception as e:
        raise RuntimeError(f"Error initializing Google Embedding model: {e}. Check GOOGLE_API_KEY and internet connection.")

    print(f"⏳ Connecting to Pinecone index: {PINECONE_INDEX_NAME}...")
    try:
        # Initialize Pinecone
        # The LangChain Pinecone integration typically handles connecting if env vars are set
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings_model,
            environment=PINECONE_ENVIRONMENT # If your LangChain Pinecone integration needs this
        )
        # You might add a check if the index is empty, or if it can connect
        # For Pinecone, you'd usually create and populate the index beforehand.
        print("✅ Connected to Pinecone.")
    except Exception as e:
        raise RuntimeError(f"Error connecting to Pinecone: {e}. Ensure API keys and index name are correct, and the index exists.")

    print(f"⏳ Initializing Google LLM: {GOOGLE_LLM_MODEL}...")
    try:
        _llm = ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL, temperature=0.2)
        print("✅ Google LLM loaded.")
    except Exception as e:
        raise RuntimeError(f"Error initializing Google LLM: {e}. Check GOOGLE_API_KEY and internet connection.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    _qa_chain = RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_QA_PROMPT}
    )

    return _qa_chain, _llm, retriever

# ... (rest of query_rag_system and if __name__ == "__main__" remains similar,
#      but the __main__ part for creating the index would be a separate script)