import os
import base64
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Import the Pinecone client to ensure environment variables are picked up if needed
from pinecone import Pinecone # ADD THIS LINE

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

_qa_chain = None
_llm = None
_retriever = None

def initialize_rag_components():
    global _qa_chain, _llm, _retriever

    if _qa_chain is not None:
        return _qa_chain, _llm, _retriever

    print(f"⏳ Initializing Google embedding model: {GOOGLE_EMBEDDING_MODEL}...")
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
        print("✅ Google Embedding model loaded.")
    except Exception as e:
        raise RuntimeError(f"Error initializing Google Embedding model: {e}. Check GOOGLE_API_KEY and internet connection.")

    print(f"⏳ Connecting to Pinecone index: {PINECONE_INDEX_NAME}...")
    try:
        # Initialize Pinecone client first to ensure env vars are picked up by Pinecone
        # The Pinecone class will automatically look for PINECONE_API_KEY and PINECONE_ENVIRONMENT
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT) # Initialize Pinecone client
        
        # Now pass the initialized embedding model and index name to PineconeVectorStore
        # The 'environment' argument is no longer needed here if Pinecone client is initialized globally or picked up
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings_model # REMOVE ', environment=PINECONE_ENVIRONMENT' from this line
        )
        
        print("✅ Connected to Pinecone.")
    except Exception as e:
        raise RuntimeError(f"Error connecting to Pinecone: {e}. Ensure API keys and index name are correct, and the index exists.")

    print(f"⏳ Initializing Google LLM: {GOOGLE_LLM_MODEL}...")
    try:
        _llm = ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL, temperature=0.2)
        print("✅ Google LLM loaded.")
    except Exception as e:
        raise RuntimeError(f"Error initializing Google LLM: {e}. Check GOOGLE_API_KEY and internet connection.")

    _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    _qa_chain = RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_QA_PROMPT}
    )

    return _qa_chain, _llm, _retriever


def query_rag_system(question: str, image_data_base64: str = None):
    """
    Queries the RAG system with a given question and optional image.
    """
    global _qa_chain, _llm, _retriever

    if _qa_chain is None:
        _qa_chain, _llm, _retriever = initialize_rag_components()

    if image_data_base64:
        print("Image attachment detected in query_rag_system. Note: Current RAG chain primarily handles text retrieval.")
        pass

    print(f"Querying RAG system with question: {question}")

    response = _qa_chain.invoke({"query": question})

    answer = response.get("result")
    source_documents = response.get("source_documents", [])

    return {
        "answer": answer,
        "source_documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in source_documents]
    }