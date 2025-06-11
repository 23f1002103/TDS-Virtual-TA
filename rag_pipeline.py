import os
import base64
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
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
_retriever = None # Added to store retriever instance

def initialize_rag_components():
    global _qa_chain, _llm, _retriever

    if _qa_chain is not None:
        return _qa_chain, _llm, _retriever # Return retriever too

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

    _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    _qa_chain = RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_retriever, # Use _retriever here
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_QA_PROMPT}
    )

    return _qa_chain, _llm, _retriever # Return retriever here as well


def query_rag_system(question: str, image_data_base64: str = None): # Parameter name changed
    """
    Queries the RAG system with a given question and optional image.
    """
    global _qa_chain, _llm, _retriever

    if _qa_chain is None:
        _qa_chain, _llm, _retriever = initialize_rag_components() # Initialize if not already

    # This part needs to be carefully integrated if your LLM needs the image
    # for answering. A standard RetrievalQA chain typically works on text context.
    # For now, we'll ensure the function passes the question to the RAG chain.
    # If the LLM itself is multimodal and you want to pass the image directly to it,
    # you might need to modify _qa_chain or call _llm.invoke() directly with
    # multimodal input.
    if image_data_base64:
        print("Image attachment detected in query_rag_system. Note: Current RAG chain primarily handles text retrieval.")
        # If you needed to truly use the image with the LLM in a multimodal way,
        # you'd restructure how _llm is called within the chain or directly.
        # Example for direct LLM call with image (if _llm supports it):
        # image_message = {
        #     "type": "image_url",
        #     "image_url": {"url": f"data:image/jpeg;base64,{image_data_base64}"}
        # }
        # text_message = {"type": "text", "text": question}
        # response_from_llm = _llm.invoke([HumanMessage(content=[text_message, image_message])])
        # return {"answer": response_from_llm.content, "source_documents": []} # No source docs from direct LLM call
        pass # For now, we continue with text-based RAG

    print(f"Querying RAG system with question: {question}")

    # Use the initialized RetrievalQA chain to get an answer
    response = _qa_chain.invoke({"query": question})

    answer = response.get("result")
    source_documents = response.get("source_documents", [])

    return {
        "answer": answer,
        "source_documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in source_documents]
    }

# This section is typically for local testing or initial index population
# if __name__ == "__main__":
#     # This part should ideally be in a separate script (e.g., create_index.py)
#     # if you need to run it only once to set up your Pinecone index.
#     # It's not part of the FastAPI application's runtime.

#     # Example of how you might test initialize and query locally:
#     print("Running local test for RAG pipeline...")
#     qa_chain_test, llm_test, retriever_test = initialize_rag_components()
#     print("RAG components initialized.")

#     test_question = "What is the capital of France?"
#     test_response = query_rag_system(question=test_question)
#     print(f"Question: {test_question}")
#     print(f"Answer: {test_response['answer']}")
#     print("Source Documents:", test_response['source_documents'])