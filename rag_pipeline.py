import os
import base64 # Import for image handling
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage # NEW: For multi-modal input
from dotenv import load_dotenv

# Load environment variables (useful for local testing)
load_dotenv()

# --- Configuration ---
VECTOR_DB_DIR = "chroma_db"
GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
GOOGLE_LLM_MODEL = "models/gemini-1.5-flash-002"

# Ensure API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")

# Custom Prompt Template
CUSTOM_QA_PROMPT = PromptTemplate(
    template="""You are a helpful assistant for university students. Your task is to answer questions based *only* on the provided context and any image provided.
If the context or image does not contain enough information to answer the question, or if you cannot find the answer, please politely state "I am sorry, but I do not have enough information in my knowledge base or the provided image to answer that question." Do not try to make up an answer.

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"],
)

# Global variable to hold the initialized RAG chain (or its components)
_qa_chain = None
_llm = None # We'll need the direct LLM instance for multi-modal

def initialize_rag_components():
    """
    Initializes and returns the RAG components.
    This function should be called ONCE at application startup.
    """
    global _qa_chain, _llm

    if _qa_chain is not None: # Already initialized
        return _qa_chain, _llm

    print(f"⏳ Initializing Google embedding model: {GOOGLE_EMBEDDING_MODEL}...")
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
        print("✅ Google Embedding model loaded.")
    except Exception as e:
        raise RuntimeError(f"Error initializing Google Embedding model: {e}. Check GOOGLE_API_KEY and internet connection.")

    print(f"⏳ Loading ChromaDB from {VECTOR_DB_DIR}...")
    try:
        vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings_model)
        if vectorstore._collection.count() == 0:
            raise ValueError("ChromaDB is empty. Please ensure the knowledge base has been created successfully.")
        print("✅ ChromaDB loaded.")
    except Exception as e:
        raise RuntimeError(f"Error loading ChromaDB: {e}. Ensure the knowledge base has been created by running 'create_knowledge_base.py'.")

    print(f"⏳ Initializing Google LLM: {GOOGLE_LLM_MODEL}...")
    try:
        _llm = ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL, temperature=0.2)
        print("✅ Google LLM loaded.")
    except Exception as e:
        raise RuntimeError(f"Error initializing Google LLM: {e}. Check GOOGLE_API_KEY and internet connection.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # The RetrievalQA chain itself might not directly handle image inputs alongside context easily.
    # We'll use the LLM directly for combining text context and image later.
    # For now, let's keep qa_chain for text-only retrieval.
    _qa_chain = RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_QA_PROMPT}
    )

    return _qa_chain, _llm, retriever # Return retriever as well for direct use

async def query_rag_system(question: str, image_data_base64: str = None):
    """
    Queries the RAG system with a question and optional image.
    Returns the answer and source documents.
    """
    if _qa_chain is None or _llm is None:
        raise RuntimeError("RAG components not initialized. Call initialize_rag_components() first.")

    # --- Step 1: Retrieve relevant text documents based on the text question ---
    # Note: Retrieval is usually text-based. Images augment the LLM's understanding.
    retriever = _qa_chain.retriever # Get the retriever instance
    retrieved_docs = retriever.get_relevant_documents(question)

    # Combine retrieved documents into a single context string
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # --- Step 2: Prepare the input for the multi-modal LLM ---
    messages = []
    if image_data_base64:
        try:
            # Decode base64 image data
            image_bytes = base64.b64decode(image_data_base64)
            # Add image part
            messages.append(HumanMessage(
                content=[
                    {"type": "text", "text": f"Question: {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{image_data_base64}"}}, # Assuming webp, adjust as needed
                    {"type": "text", "text": f"Context for answer: {context_text}"}
                ]
            ))
        except Exception as e:
            print(f"Warning: Could not decode or process image: {e}. Proceeding with text only.")
            # Fallback to text-only if image processing fails
            messages.append(HumanMessage(
                content=[
                    {"type": "text", "text": f"Question: {question}"},
                    {"type": "text", "text": f"Context for answer: {context_text}"}
                ]
            ))
    else:
        # If no image, just text content
        messages.append(HumanMessage(
            content=[
                {"type": "text", "text": f"Question: {question}"},
                {"type": "text", "text": f"Context for answer: {context_text}"}
            ]
        ))

    # Apply the prompt template logic manually for multi-modal
    # The CUSTOM_QA_PROMPT is designed for text-only.
    # For multi-modal, we format the messages directly.
    # The core prompt instruction "answer based *only* on the provided context and any image"
    # is embedded in the way we structure the HumanMessage.

    try:
        llm_response = await _llm.ainvoke(messages) # Use ainvoke for async compatibility with FastAPI
        answer = llm_response.content
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        answer = "I am sorry, but I encountered an issue while generating an answer."

    # Return answer and source documents for link extraction
    return {
        "answer": answer,
        "source_documents": retrieved_docs
    }


# This __name__ == "__main__" block is for *local testing* of the RAG pipeline directly
if __name__ == "__main__":
    # Initialize RAG components once
    try:
        _, _, _ = initialize_rag_components() # We don't need the return values here for this simple test
        print("RAG system initialized for local testing.")
    except Exception as e:
        print(f"Failed to initialize RAG system for local testing: {e}")
        exit(1)

    print("\nReady to answer questions from your knowledge base (local testing)!")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nYour question: ")
        if not query.strip():
            print("Please enter a question.")
            continue
        if query.lower() == 'exit':
            break

        # For local testing, we don't have an easy way to pass images via CLI
        # So, this will run as text-only
        print("Searching and generating answer (text-only for CLI)...")
        try:
            response = _qa_chain.invoke({"query": query}) # Use the text-only qa_chain for simple CLI testing

            print("\n--- Answer ---")
            if "I am sorry, but I do not have enough information" in response["result"]:
                print(response["result"])
            elif not response["result"].strip():
                 print("I am sorry, but I could not generate an answer based on the retrieved information.")
            else:
                print(response["result"])

            print("\n--- Sources ---")
            if response["source_documents"]:
                for i, doc in enumerate(response["source_documents"]):
                    print(f"Source {i+1}:")
                    if doc.metadata.get("title"):
                        print(f"  Title: {doc.metadata['title']}")
                    if doc.metadata.get("url"):
                        print(f"  URL: {doc.metadata['url']}")
                    elif doc.metadata.get("source"):
                        print(f"  Source: {doc.metadata['source']}")
                    print(f"  Content (excerpt): {doc.page_content[:200]}...")
                    print("-" * 20)
            else:
                print("No relevant source documents were retrieved to help answer this question.")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please check your API key, internet connection, or try a different query.")